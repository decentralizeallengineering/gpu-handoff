"""gpu-handoff command line: precheck, to-host, to-vm."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import probe

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2
VFIO = "vfio-pci"


class HandoffError(Exception):
    """A step failed; the caller may roll back."""


class Refused(Exception):
    """A precondition failed before anything was changed."""


# --------------------------------------------------------------------------
# Backend: the only place that touches the machine. Tests replace it.
# --------------------------------------------------------------------------
class RealBackend:
    def write(self, path: str, value: str) -> None:
        with open(path, "w") as handle:
            handle.write(value)

    def run(self, argv: list[str], timeout: float = 60) -> tuple[int, str]:
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, str(exc)
        return done.returncode, (done.stdout or "") + (done.stderr or "")


class EventLog:
    """Append-only JSONL event log. Disabled for dry runs."""

    def __init__(self, path: Path | None):
        self.path = path

    def emit(self, event: str, **fields) -> None:
        if self.path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"warning: event log not written: {exc}", file=sys.stderr)


@dataclass
class Ctx:
    sysfs: Path
    proc: Path
    xml_dir: Path
    dry_run: bool
    log: EventLog
    backend: object
    virsh: list[str]
    sleep: object = time.sleep
    clock: object = time.monotonic
    steps: list[str] = field(default_factory=list)

    def say(self, text: str) -> None:
        print(("[dry-run] " if self.dry_run else "") + text)
        self.steps.append(text)

    def write(self, path: Path | str, value: str) -> None:
        self.say(f"write {path} <- {value!r}")
        if self.dry_run:
            return
        self.log.emit("write", path=str(path), value=value)
        try:
            self.backend.write(str(path), value)
        except OSError as exc:
            self.log.emit("write_failed", path=str(path), error=str(exc))
            raise HandoffError(f"write to {path} failed: {exc}") from exc

    def run(
        self, argv: list[str], timeout: float = 60, check: bool = True
    ) -> tuple[int, str]:
        self.say("run " + " ".join(shlex.quote(a) for a in argv))
        if self.dry_run:
            return 0, ""
        self.log.emit("run", argv=argv)
        rc, out = self.backend.run(argv, timeout)
        if rc != 0:
            self.log.emit("run_failed", argv=argv, rc=rc, output=out.strip()[-400:])
            if check:
                raise HandoffError(f"{argv[0]} exited {rc}: {out.strip()[-200:]}")
        return rc, out

    def domstate(self, vm: str) -> str:
        rc, out = self.backend.run([*self.virsh, "domstate", vm], 30)
        return out.strip() if rc == 0 else "unknown"

    # sysfs shortcuts
    def dev(self, addr: str) -> Path:
        return probe.dev_path(self.sysfs, addr)

    def driver(self, addr: str) -> str:
        return probe.current_driver(self.sysfs, addr)


# --------------------------------------------------------------------------
# precheck
# --------------------------------------------------------------------------
def build_checks(ctx: Ctx, addr: str, vm: str | None) -> list[tuple[str, bool, str]]:
    sysfs = ctx.sysfs
    rows: list[tuple[str, bool, str]] = []
    if not probe.device_exists(sysfs, addr):
        return [("device present", False, f"{addr} not found under {sysfs}")]
    rows.append(("device present", True, addr))

    group = probe.group_report(sysfs, addr)
    if not group.members:
        rows.append(
            ("iommu group clean", False, "no IOMMU group (is the IOMMU enabled?)")
        )
    elif group.clean:
        rows.append(("iommu group clean", True, "members: " + ", ".join(group.members)))
    else:
        rows.append(
            (
                "iommu group clean",
                False,
                "foreign members: "
                + ", ".join(group.foreign)
                + " (all: "
                + ", ".join(group.members)
                + ")",
            )
        )

    drv = probe.current_driver(sysfs, addr)
    rows.append(("current driver", drv in probe.KNOWN_DRIVERS, drv))

    vms = probe.vms_referencing(ctx.xml_dir, addr)
    if vm is not None:
        ok = vm in vms
        detail = (
            f"{vm} references it"
            if ok
            else f"{vm} does not reference it (found: {', '.join(vms) or 'none'})"
        )
    else:
        ok = bool(vms)
        detail = ", ".join(vms) if vms else "no VM XML references this device"
    if not vms:
        locked = probe.unreadable_xml(ctx.xml_dir)
        if locked:
            # libvirt keeps domain XML root-only; say so instead of a false FAIL.
            ok = None
            detail = (f"cannot read {locked} VM XML file(s) in {ctx.xml_dir}; "
                      "run as root or point GPU_HANDOFF_XML_DIR at readable copies")
    rows.append(("vm references device", ok, detail))

    holders = probe.device_holders(ctx.proc)
    if holders:
        detail = "; ".join(f"pid {p} ({c}) {n}" for p, c, n in holders)
    else:
        detail = "no process holds /dev/nvidia*"
    rows.append(("no host process holds gpu", not holders, detail))

    methods = probe.reset_methods(sysfs, addr)
    if methods is None:
        rows.append(("reset method", False, "reset_method file missing"))
    elif not methods:
        rows.append(("reset method", False, "device lists no reset method"))
    else:
        rows.append(("reset method", True, methods))

    siblings = probe.audio_siblings(sysfs, addr)
    if not siblings:
        rows.append(("audio siblings", True, "none"))
    else:
        mismatched = [
            s
            for s in siblings
            if (probe.current_driver(sysfs, s) == VFIO) != (drv == VFIO)
        ]
        detail = ", ".join(f"{s} ({probe.current_driver(sysfs, s)})" for s in siblings)
        if mismatched:
            detail += " - driver differs from the GPU function"
        rows.append(("audio siblings", not mismatched, detail))
    return rows


def render_table(rows: list[tuple[str, bool | None, str]]) -> str:
    """ok is True (PASS), False (FAIL) or None (WARN: could not check)."""
    width = max(len(r[0]) for r in rows)
    lines = [f"{'CHECK':<{width}}  RESULT  DETAIL"]
    for name, ok, detail in rows:
        word = "WARN" if ok is None else ("PASS" if ok else "FAIL")
        lines.append(f"{name:<{width}}  {word:<6}  {detail}")
    return "\n".join(lines)


def cmd_precheck(ctx: Ctx, args) -> int:
    rows = build_checks(ctx, args.addr, args.vm)
    print(render_table(rows))
    return EXIT_OK if all(ok is not False for _, ok, _ in rows) else EXIT_FAIL


# --------------------------------------------------------------------------
# shared helpers for the mutating commands
# --------------------------------------------------------------------------
def gate(ctx: Ctx, addr: str, vm: str) -> list[str]:
    """Refuse unless the device, its group and the VM reference look sane."""
    if not probe.device_exists(ctx.sysfs, addr):
        raise Refused(f"{addr} not found under {ctx.sysfs}")
    group = probe.group_report(ctx.sysfs, addr)
    if not group.members:
        raise Refused("device has no IOMMU group")
    if not group.clean:
        raise Refused(
            "IOMMU group is not clean, foreign members: " + ", ".join(group.foreign)
        )
    if vm not in probe.vms_referencing(ctx.xml_dir, addr):
        raise Refused(f"VM {vm!r} has no PCI hostdev for {addr}")
    return probe.movable_functions(ctx.sysfs, addr)


def unbind(ctx: Ctx, fn: str) -> None:
    ctx.write(ctx.dev(fn) / "driver" / "unbind", fn)


def bind_vfio(ctx: Ctx, fn: str) -> None:
    ctx.write(ctx.dev(fn) / "driver_override", VFIO)
    ctx.write(ctx.sysfs / "bus" / "pci" / "drivers" / VFIO / "bind", fn)


def bind_host(ctx: Ctx, fn: str, driver: str | None) -> None:
    """Hand fn to a host driver: a named one, or whatever the kernel probes."""
    override = ctx.dev(fn) / "driver_override"
    if driver:
        ctx.write(override, driver)
        ctx.write(ctx.sysfs / "bus" / "pci" / "drivers" / driver / "bind", fn)
        ctx.write(override, "\n")
    else:
        ctx.write(override, "\n")
        ctx.write(ctx.sysfs / "bus" / "pci" / "drivers_probe", fn)


def wait_for(ctx: Ctx, vm: str, want: str, timeout: float, interval: float) -> bool:
    deadline = ctx.clock() + timeout
    while True:
        if ctx.domstate(vm) == want:
            return True
        if ctx.clock() >= deadline:
            return False
        ctx.sleep(interval)


@contextlib.contextmanager
def handoff_lock(ctx: Ctx, path: Path):
    if ctx.dry_run:
        ctx.say(f"lock {path}")
        yield
        return
    with contextlib.ExitStack() as stack:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(open(path, "a+"))
        except OSError as exc:
            raise Refused(f"cannot open lock file {path}: {exc}") from exc
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused(f"another handoff holds the lock {path}") from None
        yield


# --------------------------------------------------------------------------
# to-host
# --------------------------------------------------------------------------
def cmd_to_host(ctx: Ctx, args) -> int:
    addr, vm = args.addr, args.vm
    funcs = gate(ctx, addr, vm)
    on_vfio = [f for f in funcs if ctx.driver(f) == VFIO]
    if not on_vfio:
        print(f"nothing in the group of {addr} is bound to {VFIO}; already on the host")
        return EXIT_OK

    ctx.log.emit("to-host_start", addr=addr, vm=vm, functions=on_vfio)
    stopped = False
    try:
        state = "running" if ctx.dry_run else ctx.domstate(vm)
        if state == "shut off":
            ctx.say(f"vm {vm} already shut off")
        else:
            ctx.run([*ctx.virsh, "shutdown", vm])
            if ctx.dry_run:
                ctx.say(
                    f"wait {' '.join(ctx.virsh)} domstate {vm} == 'shut off' (timeout {args.timeout:g}s)"
                )
            elif not wait_for(ctx, vm, "shut off", args.timeout, args.poll_interval):
                raise Refused(
                    f"VM {vm} did not shut off within {args.timeout:g}s; nothing was unbound "
                    "(shut it down yourself or raise --timeout)"
                )
        stopped = True
        for fn in on_vfio:
            unbind(ctx, fn)
        for fn in on_vfio:
            bind_host(ctx, fn, args.host_driver if fn == addr else None)
        verify_host(ctx, addr, on_vfio, args)
    except Refused:
        ctx.log.emit("to-host_refused", addr=addr, vm=vm)
        raise
    except HandoffError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        ctx.log.emit("to-host_failed", addr=addr, error=str(exc))
        if stopped:
            rollback_to_vfio(ctx, on_vfio)
            print(
                f"rolled back: functions rebound to {VFIO}; VM {vm} left shut off",
                file=sys.stderr,
            )
        return EXIT_FAIL
    ctx.log.emit("to-host_done", addr=addr, vm=vm)
    print(f"{addr} is on the host" if not ctx.dry_run else "dry run complete")
    return EXIT_OK


def verify_host(ctx: Ctx, addr: str, funcs: list[str], args) -> None:
    for fn in funcs:
        if ctx.dry_run:
            ctx.say(f"verify driver of {fn} is a host driver (not {VFIO}, not none)")
            continue
        drv = ctx.driver(fn)
        if drv in (VFIO, "none"):
            raise HandoffError(f"{fn} ended on driver {drv!r}, expected a host driver")
        if fn == addr and args.host_driver and drv != args.host_driver:
            raise HandoffError(f"{fn} ended on {drv!r}, expected {args.host_driver!r}")
        ctx.log.emit("verified", function=fn, driver=drv)
    if args.probe:
        ctx.run(["nvidia-smi", "-L"], timeout=30)


def rollback_to_vfio(ctx: Ctx, funcs: list[str]) -> None:
    """Best effort: every error is logged and the next function is still tried."""
    ctx.log.emit("rollback_start", target=VFIO, functions=funcs)
    for fn in funcs:
        try:
            if ctx.driver(fn) == VFIO:
                continue
            if ctx.driver(fn) != "none":
                unbind(ctx, fn)
            bind_vfio(ctx, fn)
        except HandoffError as exc:
            print(f"rollback of {fn} failed: {exc}", file=sys.stderr)
            ctx.log.emit("rollback_failed", function=fn, error=str(exc))
    ctx.log.emit("rollback_done", target=VFIO)


# --------------------------------------------------------------------------
# to-vm
# --------------------------------------------------------------------------
def cmd_to_vm(ctx: Ctx, args) -> int:
    addr, vm = args.addr, args.vm
    funcs = gate(ctx, addr, vm)

    holders = probe.device_holders(ctx.proc)
    if holders:
        lines = [f"  pid {p} ({c}) holds {n}" for p, c, n in holders]
        raise Refused("host processes hold the GPU, refusing:\n" + "\n".join(lines))
    if not (ctx.sysfs / "bus" / "pci" / "drivers" / VFIO).is_dir():
        raise Refused(f"{VFIO} driver is not loaded (load the vfio-pci module first)")
    if not ctx.dry_run and ctx.domstate(vm) == "running":
        raise Refused(f"VM {vm} is already running")

    original = {f: ctx.driver(f) for f in funcs}
    ctx.log.emit("to-vm_start", addr=addr, vm=vm, original=original)
    try:
        for fn in funcs:
            if original[fn] not in (VFIO, "none"):
                unbind(ctx, fn)
        for fn in funcs:
            if original[fn] != VFIO:
                bind_vfio(ctx, fn)
        verify_vfio(ctx, funcs)
        ctx.run([*ctx.virsh, "start", vm])
        if ctx.dry_run:
            ctx.say(f"verify domstate {vm} == 'running'")
        elif not wait_for(ctx, vm, "running", args.timeout, args.poll_interval):
            raise HandoffError(f"VM {vm} is not running after start")
    except HandoffError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        ctx.log.emit("to-vm_failed", addr=addr, error=str(exc))
        rollback_to_host(ctx, original)
        print("rolled back: functions returned to their host drivers", file=sys.stderr)
        return EXIT_FAIL
    ctx.log.emit("to-vm_done", addr=addr, vm=vm)
    print(f"{addr} is in VM {vm}" if not ctx.dry_run else "dry run complete")
    return EXIT_OK


def verify_vfio(ctx: Ctx, funcs: list[str]) -> None:
    for fn in funcs:
        if ctx.dry_run:
            ctx.say(f"verify driver of {fn} is {VFIO}")
        elif ctx.driver(fn) != VFIO:
            raise HandoffError(f"{fn} is on {ctx.driver(fn)!r}, expected {VFIO}")


def rollback_to_host(ctx: Ctx, original: dict[str, str]) -> None:
    ctx.log.emit("rollback_start", target="host", original=original)
    for fn, was in original.items():
        if was == VFIO:
            continue
        try:
            if ctx.driver(fn) == VFIO:
                unbind(ctx, fn)
            if ctx.driver(fn) == "none":
                bind_host(ctx, fn, None)
        except HandoffError as exc:
            print(f"rollback of {fn} failed: {exc}", file=sys.stderr)
            ctx.log.emit("rollback_failed", function=fn, error=str(exc))
    ctx.log.emit("rollback_done", target="host")


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def default_lock_path() -> Path:
    for base in (Path("/run/lock"), Path(tempfile.gettempdir())):
        if os.access(base, os.W_OK):
            return base / "gpu-handoff.lock"
    return Path(tempfile.gettempdir()) / "gpu-handoff.lock"


def default_log_path() -> Path:
    state = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return Path(state) / "gpu-handoff" / "events.jsonl"


def addr_arg(text: str) -> str:
    try:
        return probe.normalize_addr(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "addr",
        type=addr_arg,
        metavar="PCI_ADDR",
        help="GPU function, e.g. 0000:01:00.0",
    )
    common.add_argument("--sysfs-root", default=env("GPU_HANDOFF_SYSFS_ROOT", "/sys"))
    common.add_argument("--proc-root", default=env("GPU_HANDOFF_PROC_ROOT", "/proc"))
    common.add_argument(
        "--xml-dir",
        default=env("GPU_HANDOFF_XML_DIR", "/etc/libvirt/qemu"),
        help="directory of libvirt domain XML files",
    )

    mutating = argparse.ArgumentParser(add_help=False)
    mutating.add_argument("--vm", required=True, help="libvirt domain name")
    mutating.add_argument(
        "--dry-run",
        action="store_true",
        help="print every write and command, change nothing",
    )
    mutating.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the VM state change",
    )
    mutating.add_argument("--poll-interval", type=float, default=2.0)
    mutating.add_argument("--lock-file", default=env("GPU_HANDOFF_LOCK"))
    mutating.add_argument("--log-file", default=env("GPU_HANDOFF_LOG"))

    parser = argparse.ArgumentParser(prog="gpu-handoff", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precheck", parents=[common], help="read-only readiness table")
    p.add_argument("--vm", help="also require this VM to reference the device")
    p = sub.add_parser("to-host", parents=[common, mutating], help="VM to host")
    p.add_argument(
        "--host-driver",
        help="bind the GPU function to this driver instead of letting the kernel probe",
    )
    p.add_argument(
        "--probe", action="store_true", help="also run 'nvidia-smi -L' after binding"
    )
    sub.add_parser("to-vm", parents=[common, mutating], help="host to VM")
    return parser


def main(argv: list[str] | None = None, backend=None, sleep=None, clock=None) -> int:
    args = build_parser().parse_args(argv)
    dry = getattr(args, "dry_run", False)
    log_path = None
    if args.cmd != "precheck" and not dry:
        log_path = Path(args.log_file) if args.log_file else default_log_path()
    ctx = Ctx(
        sysfs=Path(args.sysfs_root),
        proc=Path(args.proc_root),
        xml_dir=Path(args.xml_dir),
        dry_run=dry,
        log=EventLog(log_path),
        backend=backend or RealBackend(),
        virsh=shlex.split(os.environ.get("GPU_HANDOFF_VIRSH", "virsh")),
        sleep=sleep or time.sleep,
        clock=clock or time.monotonic,
    )
    if args.cmd == "precheck":
        return cmd_precheck(ctx, args)
    handler = cmd_to_host if args.cmd == "to-host" else cmd_to_vm
    try:
        lock = Path(args.lock_file) if args.lock_file else default_lock_path()
        with handoff_lock(ctx, lock):
            return handler(ctx, args)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        ctx.log.emit("refused", reason=str(exc))
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
