"""Read-only inspection of PCI devices through a sysfs/proc/libvirt-XML tree.

Every function takes the tree roots as arguments so tests can point them at a
fake tree. Nothing in this module writes anything.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ADDR_RE = re.compile(
    r"^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", re.IGNORECASE
)
KNOWN_DRIVERS = ("vfio-pci", "nvidia", "amdgpu", "nouveau", "none")
CLASS_AUDIO = 0x0403
BRIDGE_CLASSES = (0x0604, 0x0600)


def normalize_addr(text: str) -> str:
    """Return a full domain:bus:slot.fn address, or raise ValueError."""
    text = text.strip().lower()
    if not ADDR_RE.match(text):
        raise ValueError(f"not a PCI address: {text!r}")
    return text if text.count(":") == 2 else "0000:" + text


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def dev_path(sysfs: Path, addr: str) -> Path:
    return sysfs / "bus" / "pci" / "devices" / addr


def device_exists(sysfs: Path, addr: str) -> bool:
    return dev_path(sysfs, addr).is_dir()


def current_driver(sysfs: Path, addr: str) -> str:
    link = dev_path(sysfs, addr) / "driver"
    if not link.is_symlink():
        return "none"
    return os.path.basename(os.path.realpath(link))


def pci_class(sysfs: Path, addr: str) -> int:
    """Return the 16-bit base+sub class (0x0403 = audio), or 0 if unknown."""
    raw = read_text(dev_path(sysfs, addr) / "class")
    try:
        return int(raw, 16) >> 8 if raw else 0
    except ValueError:
        return 0


def pci_vendor(sysfs: Path, addr: str) -> str:
    return (read_text(dev_path(sysfs, addr) / "vendor") or "").lower()


def slot_of(addr: str) -> str:
    return addr.rsplit(".", 1)[0]


def group_members(sysfs: Path, addr: str) -> list[str] | None:
    """All devices in the IOMMU group of addr, or None when there is no group."""
    link = dev_path(sysfs, addr) / "iommu_group"
    if not link.exists():
        return None
    try:
        return sorted(os.listdir(Path(os.path.realpath(link)) / "devices"))
    except OSError:
        return None


def is_bridge(sysfs: Path, addr: str) -> bool:
    return pci_class(sysfs, addr) in BRIDGE_CLASSES


@dataclass
class GroupReport:
    members: list[str] = field(default_factory=list)
    foreign: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return bool(self.members) and not self.foreign


def group_report(sysfs: Path, addr: str) -> GroupReport:
    """A group is clean when every member is a bridge or a function of addr's slot."""
    members = group_members(sysfs, addr) or []
    foreign = [
        m for m in members if slot_of(m) != slot_of(addr) and not is_bridge(sysfs, m)
    ]
    return GroupReport(members, foreign)


def movable_functions(sysfs: Path, addr: str) -> list[str]:
    """Group members that get rebound: every non-bridge member."""
    members = group_members(sysfs, addr) or []
    return [m for m in members if not is_bridge(sysfs, m)]


def audio_siblings(sysfs: Path, addr: str) -> list[str]:
    root = sysfs / "bus" / "pci" / "devices"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [
        n
        for n in names
        if n != addr
        and slot_of(n) == slot_of(addr)
        and pci_class(sysfs, n) == CLASS_AUDIO
    ]


def reset_methods(sysfs: Path, addr: str) -> str | None:
    return read_text(dev_path(sysfs, addr) / "reset_method")


def unreadable_xml(xml_dir: Path) -> int:
    """Count domain XML files that exist but this user cannot read."""
    try:
        return sum(1 for f in Path(xml_dir).glob("*.xml") if not os.access(f, os.R_OK))
    except OSError:
        return 0


def vms_referencing(xml_dir: Path, addr: str) -> list[str]:
    """Names of libvirt domains whose XML has a PCI hostdev for addr."""
    domain, bus, rest = addr.split(":")
    slot, fn = rest.split(".")
    want = tuple(int(x, 16) for x in (domain, bus, slot)) + (int(fn, 16),)
    found = []
    try:
        files = sorted(Path(xml_dir).glob("*.xml"))
    except OSError:
        return []
    for xml_file in files:
        try:
            root = ET.parse(xml_file).getroot()
        except (OSError, ET.ParseError):
            continue
        for hostdev in root.iter("hostdev"):
            if hostdev.get("type") != "pci":
                continue
            src = hostdev.find("source/address")
            if src is None:
                continue
            try:
                got = tuple(
                    int(src.get(k, ""), 16)
                    for k in ("domain", "bus", "slot", "function")
                )
            except ValueError:
                continue
            if got == want:
                name = root.findtext("name") or xml_file.stem
                found.append(name)
                break
    return found


def device_holders(
    proc: Path, prefix: str = "/dev/nvidia"
) -> list[tuple[int, str, str]]:
    """Processes with an open fd on a /dev/nvidia* node: (pid, comm, node).

    The scan is system-wide: it cannot tell which of several GPUs a process
    uses, so any holder counts against the device.
    """
    out = []
    try:
        entries = sorted(os.listdir(proc), key=lambda s: (not s.isdigit(), s))
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        fd_dir = proc / entry / "fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        nodes = set()
        for fd in fds:
            try:
                target = os.readlink(fd_dir / fd)
            except OSError:
                continue
            if target.startswith(prefix):
                nodes.add(target.split(" ")[0])
        if nodes:
            comm = read_text(proc / entry / "comm") or "?"
            out.append((int(entry), comm, ",".join(sorted(nodes))))
    return out
