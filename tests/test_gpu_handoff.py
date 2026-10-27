import json
import os

import pytest

from gpu_handoff import probe
from gpu_handoff.cli import main
from tests.conftest import AUDIO, GPU, VM, FakeBackend, base_args, build_tree, mut_args


def test_precheck_clean_group(tmp_path, capsys):
    t = build_tree(tmp_path)
    rc = main(["precheck", GPU, "--vm", VM, *base_args(t, tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FAIL" not in out
    assert "iommu group clean" in out and "vfio-pci" in out and VM in out
    assert "flr bus" in out and AUDIO in out


def test_precheck_flags_dirty_group(tmp_path, capsys):
    t = build_tree(tmp_path, dirty=True)
    rc = main(["precheck", GPU, *base_args(t, tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    clean_line = next(
        line for line in out.splitlines() if line.startswith("iommu group clean")
    )
    assert "FAIL" in clean_line and "0000:02:00.0" in clean_line


def test_precheck_flags_holder_and_missing_reset(tmp_path, capsys):
    t = build_tree(tmp_path, holder=True, reset="")
    rc = main(["precheck", GPU, *base_args(t, tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pid 4242 (cuda-job)" in out
    assert "no reset method" in out


def test_precheck_reports_driver_and_address_normalisation(tmp_path, capsys):
    t = build_tree(tmp_path, driver="nvidia")
    assert probe.normalize_addr("01:00.0") == GPU
    with pytest.raises(ValueError):
        probe.normalize_addr("../etc")
    main(["precheck", "01:00.0", *base_args(t, tmp_path)])
    line = next(
        l
        for l in capsys.readouterr().out.splitlines()
        if l.startswith("current driver")
    )
    assert "PASS" in line and "nvidia" in line


def test_to_host_dry_run_order(tmp_path, capsys):
    t = build_tree(tmp_path)
    backend = FakeBackend(t["sysfs"])
    rc = main(
        ["to-host", GPU, "--vm", VM, "--dry-run", "--probe", *mut_args(t, tmp_path)],
        backend=backend,
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert backend.writes == [] and backend.commands == []
    assert not (tmp_path / "events.jsonl").exists()
    assert all(
        line.startswith("[dry-run] ") or line == "dry run complete" for line in out
    )
    kinds = []
    for line in out:
        body = line.removeprefix("[dry-run] ")
        if body.startswith("lock"):
            kinds.append("lock")
        elif body.startswith("run virsh shutdown"):
            kinds.append("shutdown")
        elif body.startswith("wait"):
            kinds.append("wait")
        elif "driver/unbind" in body:
            kinds.append("unbind:" + body.split("devices/")[1].split("/")[0])
        elif "driver_override" in body:
            kinds.append("override:" + body.split("devices/")[1].split("/")[0])
        elif "drivers_probe" in body:
            kinds.append("probe")
        elif body.startswith("verify"):
            kinds.append("verify")
        elif body.startswith("run nvidia-smi -L"):
            kinds.append("smi")
    assert kinds == [
        "lock",
        "shutdown",
        "wait",
        f"unbind:{GPU}",
        f"unbind:{AUDIO}",
        f"override:{GPU}",
        "probe",
        f"override:{AUDIO}",
        "probe",
        "verify",
        "verify",
        "smi",
    ]


def test_to_vm_refuses_while_holder_exists(tmp_path, capsys):
    t = build_tree(tmp_path, driver="nvidia", holder=True)
    backend = FakeBackend(t["sysfs"], vm_state="shut off")
    rc = main(["to-vm", GPU, "--vm", VM, *mut_args(t, tmp_path)], backend=backend)
    err = capsys.readouterr().err
    assert rc == 2
    assert "4242" in err and "cuda-job" in err and "/dev/nvidia0" in err
    assert backend.writes == []
    assert all(c[1] == "domstate" for c in backend.commands) or backend.commands == []


def test_to_vm_success_moves_all_functions(tmp_path):
    t = build_tree(tmp_path, driver="nvidia")
    backend = FakeBackend(t["sysfs"], vm_state="shut off")
    rc = main(["to-vm", GPU, "--vm", VM, *mut_args(t, tmp_path)], backend=backend)
    assert rc == 0
    assert probe.current_driver(t["sysfs"], GPU) == "vfio-pci"
    assert probe.current_driver(t["sysfs"], AUDIO) == "vfio-pci"
    assert ["virsh", "start", VM] in backend.commands
    events = [
        json.loads(l)["event"]
        for l in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events[0] == "to-vm_start" and events[-1] == "to-vm_done"


def test_to_host_rolls_back_on_failed_bind(tmp_path, capsys):
    t = build_tree(tmp_path)
    # the audio function cannot bind to the host driver
    backend = FakeBackend(t["sysfs"], fail_bind={(AUDIO, "nvidia")})
    rc = main(
        ["to-host", GPU, "--vm", VM, *mut_args(t, tmp_path)],
        backend=backend,
        sleep=lambda s: None,
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "rolled back" in err
    assert probe.current_driver(t["sysfs"], GPU) == "vfio-pci"
    assert probe.current_driver(t["sysfs"], AUDIO) == "vfio-pci"
    events = [
        json.loads(l)["event"]
        for l in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert "to-host_failed" in events and "rollback_done" in events


def test_to_host_shutdown_timeout_changes_nothing(tmp_path, capsys):
    t = build_tree(tmp_path)
    backend = FakeBackend(t["sysfs"], shutdown_works=False)
    now = [0.0]

    def tick(seconds):
        now[0] += seconds

    rc = main(
        ["to-host", GPU, "--vm", VM, "--timeout", "10", *mut_args(t, tmp_path)],
        backend=backend,
        sleep=tick,
        clock=lambda: now[0],
    )
    assert rc == 2
    assert "did not shut off" in capsys.readouterr().err
    assert backend.writes == []
    assert probe.current_driver(t["sysfs"], GPU) == "vfio-pci"


def test_lock_blocks_second_handoff(tmp_path, capsys):
    import fcntl

    t = build_tree(tmp_path)
    backend = FakeBackend(t["sysfs"])
    with open(tmp_path / "lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = main(["to-host", GPU, "--vm", VM, *mut_args(t, tmp_path)], backend=backend)
    assert rc == 2
    assert "another handoff" in capsys.readouterr().err
    assert backend.writes == [] and backend.commands == []


def test_to_host_refuses_dirty_group(tmp_path, capsys):
    t = build_tree(tmp_path, dirty=True)
    backend = FakeBackend(t["sysfs"])
    rc = main(["to-host", GPU, "--vm", VM, *mut_args(t, tmp_path)], backend=backend)
    assert rc == 2 and backend.writes == []
    assert "not clean" in capsys.readouterr().err
    assert os.path.exists(t["sysfs"])
