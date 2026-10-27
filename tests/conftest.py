"""Fake sysfs/proc/libvirt tree and a fake backend that emulates driver binding."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

GPU = "0000:01:00.0"
AUDIO = "0000:01:00.1"
BRIDGE = "0000:00:01.1"
OTHER = "0000:02:00.0"
VM = "vm-ann"

XML = """<domain type='kvm'>
  <name>{name}</name>
  <devices>
    <hostdev mode='subsystem' type='pci' managed='no'>
      <source><address domain='0x0000' bus='0x01' slot='0x00' function='0x0'/></source>
    </hostdev>
  </devices>
</domain>
"""


def _device(
    sysfs: Path,
    addr: str,
    cls: str,
    vendor: str,
    driver: str | None,
    group: int,
    reset: str = "flr bus",
):
    d = sysfs / "bus/pci/devices" / addr
    d.mkdir(parents=True)
    (d / "class").write_text(cls + "\n")
    (d / "vendor").write_text(vendor + "\n")
    (d / "reset_method").write_text(reset + "\n")
    (d / "driver_override").write_text("(null)\n")
    os.symlink(f"../../../../kernel/iommu_groups/{group}", d / "iommu_group")
    gdir = sysfs / "kernel/iommu_groups" / str(group) / "devices"
    gdir.mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../../../bus/pci/devices/{addr}", gdir / addr)
    if driver:
        link_driver(sysfs, addr, driver)


def link_driver(sysfs: Path, addr: str, driver: str) -> None:
    (sysfs / "bus/pci/drivers" / driver).mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../drivers/{driver}", sysfs / "bus/pci/devices" / addr / "driver")


def build_tree(
    root: Path,
    driver: str = "vfio-pci",
    dirty: bool = False,
    holder: bool = False,
    reset: str = "flr bus",
) -> dict[str, Path]:
    sysfs, proc, xml = root / "sys", root / "proc", root / "xml"
    sysfs.mkdir()
    xml.mkdir()
    proc.mkdir()
    _device(sysfs, GPU, "0x030000", "0x10de", driver, 7, reset)
    _device(sysfs, AUDIO, "0x040300", "0x10de", driver, 7)
    _device(sysfs, BRIDGE, "0x060400", "0x1022", "pcieport", 7)
    if dirty:
        _device(sysfs, OTHER, "0x010802", "0x144d", "nvme", 7)
    (sysfs / "bus/pci/drivers_probe").write_text("")
    (sysfs / "bus/pci/drivers/vfio-pci").mkdir(parents=True, exist_ok=True)
    (xml / f"{VM}.xml").write_text(XML.format(name=VM))
    (xml / "other.xml").write_text(
        XML.format(name="vm-unrelated").replace("0x01", "0x09")
    )
    if holder:
        fd = proc / "4242" / "fd"
        fd.mkdir(parents=True)
        os.symlink("/dev/nvidia0", fd / "3")
        (proc / "4242" / "comm").write_text("cuda-job\n")
    return {"sysfs": sysfs, "proc": proc, "xml": xml}


class FakeBackend:
    """Emulates sysfs bind/unbind on the fake tree and a scripted virsh."""

    def __init__(
        self,
        sysfs: Path,
        host_driver: str = "nvidia",
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
        link_driver_target = self.sysfs / "bus/pci/drivers" / drv
        link_driver_target.mkdir(parents=True, exist_ok=True)
        os.symlink(f"../../drivers/{drv}", self._drv(addr))

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
            target = self.override.get(val) or self.host_driver
            if (val, target) in self.fail_bind:
                raise OSError(errno.ENODEV, "probe failed")
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
