"""Fake sysfs/proc/libvirt tree and a fake backend that emulates driver binding.

The layout follows the kernel's sysfs ABI: bus/pci/devices/<addr> with class,
vendor, device, reset_method, driver_override, a driver symlink into
bus/pci/drivers and an iommu_group symlink into kernel/iommu_groups.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

GPU = "0000:01:00.0"
AUDIO = "0000:01:00.1"
BRIDGE = "0000:00:01.1"
OTHER = "0000:02:00.0"
VM = "guest"


def add_device(
    sysfs: Path,
    addr: str,
    cls: str,
    vendor: str,
    driver: str | None,
    group: int,
    reset: str | None = "flr bus",
    device: str = "0x0000",
) -> Path:
    """One PCI function. reset=None leaves out the reset_method file."""
    d = sysfs / "bus/pci/devices" / addr
    d.mkdir(parents=True)
    (d / "class").write_text(cls + "\n")
    (d / "vendor").write_text(vendor + "\n")
    (d / "device").write_text(device + "\n")
    if reset is not None:
        (d / "reset_method").write_text(reset + "\n")
    (d / "driver_override").write_text("(null)\n")
    os.symlink(f"../../../../kernel/iommu_groups/{group}", d / "iommu_group")
    gdir = sysfs / "kernel/iommu_groups" / str(group) / "devices"
    gdir.mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../../../bus/pci/devices/{addr}", gdir / addr)
    if driver:
        link_driver(sysfs, addr, driver)
    return d


def link_driver(sysfs: Path, addr: str, driver: str) -> None:
    (sysfs / "bus/pci/drivers" / driver).mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../drivers/{driver}", sysfs / "bus/pci/devices" / addr / "driver")


def add_module(sysfs: Path, name: str, taint: str = "") -> Path:
    """A loaded module: /sys/module/<name> with its taint flags."""
    d = sysfs / "module" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "taint").write_text(taint + "\n")
    return d


def add_holder(proc: Path, pid: int, comm: str, *targets: str) -> None:
    """A process with one open fd per target, e.g. '/dev/nvidia0'."""
    fd = proc / str(pid) / "fd"
    fd.mkdir(parents=True)
    for n, target in enumerate(targets, start=3):
        os.symlink(target, fd / str(n))
    (proc / str(pid) / "comm").write_text(comm + "\n")


def domain_xml(name: str, addrs=(GPU,), managed: str | None = "no") -> str:
    """A libvirt domain with one PCI hostdev per address; managed=None omits it."""
    attr = "" if managed is None else f" managed='{managed}'"
    hostdevs = []
    for addr in addrs:
        domain, bus, rest = addr.split(":")
        slot, fn = rest.split(".")
        hostdevs.append(
            f"    <hostdev mode='subsystem' type='pci'{attr}>\n"
            f"      <source><address domain='0x{domain}' bus='0x{bus}' "
            f"slot='0x{slot}' function='0x{fn}'/></source>\n"
            "    </hostdev>\n"
        )
    return (
        "<domain type='kvm'>\n"
        f"  <name>{name}</name>\n"
        "  <devices>\n" + "".join(hostdevs) + "  </devices>\n</domain>\n"
    )


def build_tree(
    root: Path,
    driver: str = "vfio-pci",
    dirty: bool = False,
    holder: bool = False,
    reset: str | None = "flr bus",
) -> dict[str, Path]:
    """An NVIDIA GPU + audio function behind a root port, all in IOMMU group 7."""
    sysfs, proc, xml = root / "sys", root / "proc", root / "xml"
    sysfs.mkdir()
    xml.mkdir()
    proc.mkdir()
    add_device(sysfs, GPU, "0x030000", "0x10de", driver, 7, reset, "0x2704")
    add_device(sysfs, AUDIO, "0x040300", "0x10de", driver, 7, "flr bus", "0x22bb")
    add_device(sysfs, BRIDGE, "0x060400", "0x1022", "pcieport", 7, None, "0x1483")
    if dirty:
        add_device(sysfs, OTHER, "0x010802", "0x144d", "nvme", 7, "flr", "0xa808")
    (sysfs / "bus/pci/drivers_probe").write_text("")
    (sysfs / "bus/pci/drivers/vfio-pci").mkdir(parents=True, exist_ok=True)
    (xml / f"{VM}.xml").write_text(domain_xml(VM))
    (xml / "other.xml").write_text(domain_xml("unrelated", ("0000:09:00.0",)))
    if holder:
        add_holder(proc, 4242, "cuda-job", "/dev/nvidia0")
    return {"sysfs": sysfs, "proc": proc, "xml": xml}


class FakeBackend:
    """Emulates sysfs bind/unbind on the fake tree and a scripted virsh.

    host_driver is the driver drivers_probe attaches: one name for every
    function, or a dict per address where None means no driver claims it
    (the kernel then leaves the function unbound and the write still succeeds).
    """

    def __init__(
        self,
        sysfs: Path,
        host_driver: str | dict = "nvidia",
        fail_bind: set | None = None,
        vm_state: str = "running",
        shutdown_works: bool = True,
    ):
        self.sysfs = sysfs
        self.host_driver = host_driver
        self.fail_bind = fail_bind or set()
        self.vm_state = vm_state
        self.shutdown_works = shutdown_works
        self.writes: list[tuple[str, str]] = []
        self.commands: list[list[str]] = []
        self.override: dict[str, str] = {}

    def _drv(self, addr):
        return self.sysfs / "bus/pci/devices" / addr / "driver"

    def _link(self, addr, drv):
        if self._drv(addr).is_symlink():
            self._drv(addr).unlink()
        (self.sysfs / "bus/pci/drivers" / drv).mkdir(parents=True, exist_ok=True)
        os.symlink(f"../../drivers/{drv}", self._drv(addr))

    def _probe_target(self, addr):
        if self.override.get(addr):
            return self.override[addr]
        if isinstance(self.host_driver, dict):
            return self.host_driver.get(addr)
        return self.host_driver

    def write(self, path: str, value: str) -> None:
        rel = Path(path).relative_to(self.sysfs).as_posix()
        self.writes.append((rel, value))
        val = value.strip()
        if rel.endswith("/driver/unbind"):
            if not self._drv(val).is_symlink():
                raise OSError(errno.ENODEV, "no driver bound")
            self._drv(val).unlink()
        elif rel.endswith("/driver_override"):
            self.override[rel.split("/")[3]] = val
        elif "/drivers/" in rel and rel.endswith("/bind"):
            drv = rel.split("/")[-2]
            if (val, drv) in self.fail_bind:
                raise OSError(errno.ENODEV, "bind failed")
            self._link(val, drv)
        elif rel.endswith("drivers_probe"):
            target = self._probe_target(val)
            if (val, target) in self.fail_bind:
                raise OSError(errno.ENODEV, "probe failed")
            if target:
                self._link(val, target)
        else:
            raise OSError(errno.EINVAL, f"unexpected write {rel}")

    def run(self, argv, timeout=60):
        self.commands.append(list(argv))
        if argv[1:2] == ["domstate"]:
            return 0, self.vm_state + "\n"
        if argv[1:2] == ["shutdown"]:
            if self.shutdown_works:
                self.vm_state = "shut off"
            return 0, ""
        if argv[1:2] == ["start"]:
            self.vm_state = "running"
            return 0, ""
        return 0, ""


@pytest.fixture
def tree(tmp_path):
    return build_tree(tmp_path)


def base_args(tree, tmp_path):
    return [
        "--sysfs-root",
        str(tree["sysfs"]),
        "--proc-root",
        str(tree["proc"]),
        "--xml-dir",
        str(tree["xml"]),
    ]


def mut_args(tree, tmp_path):
    return base_args(tree, tmp_path) + [
        "--lock-file",
        str(tmp_path / "lock"),
        "--log-file",
        str(tmp_path / "events.jsonl"),
    ]
