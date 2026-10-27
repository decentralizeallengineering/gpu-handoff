# gpu-handoff

Hand a PCIe GPU between a libvirt VM (vfio-pci passthrough) and the Linux host at runtime, without a reboot.

Typical use: the GPU serves a Windows or Linux guest most of the time, and you want it back on the host driver (CUDA, rendering) for a while, then back in the guest. gpu-handoff does the shutdown, unbind, bind, start and verification steps in a fixed order, refuses unsafe states, and rolls back when a step fails.

## Requirements

- Linux with the IOMMU enabled (`intel_iommu=on` or `amd_iommu=on`, plus `iommu=pt` if you like) and the GPU in an IOMMU group that holds only its own functions and bridges.
- The `vfio-pci` module loaded (`modprobe vfio-pci`). The tool does not load modules.
- libvirt with `virsh`, and the domain XML lists the GPU as a PCI `hostdev`. Use `managed='no'` so libvirt does not rebind the device itself.
- Root, because it writes to sysfs and calls `virsh`.
- Python 3.10 or newer, no third-party packages.
- Secure Boot: the NVIDIA proprietary driver, and any out-of-tree module, must be signed (enrolled MOK or distro-signed) or the host bind step will fail and rollback will leave the card on vfio-pci. vfio-pci itself ships in the kernel and needs no signing.

## Install

    pip install .

or run it in place with `python3 -m gpu_handoff`.

## Quick start

    gpu-handoff precheck 0000:01:00.0 --vm win-guest
    gpu-handoff to-host  0000:01:00.0 --vm win-guest --dry-run
    sudo gpu-handoff to-host 0000:01:00.0 --vm win-guest --probe
    sudo gpu-handoff to-vm   0000:01:00.0 --vm win-guest

Commands:

- `precheck ADDR` is read-only and prints a PASS/FAIL table: IOMMU group members and cleanliness, current driver (vfio-pci, nvidia, amdgpu, none, ...), which VM XML references the device, whether any process holds `/dev/nvidia*` (a `/proc/*/fd` scan), the `reset_method` list, and audio function siblings. Exit code 0 only if every row passes.
- `to-host ADDR --vm NAME` shuts the VM down gracefully (`--timeout`, default 120 s; it never force-kills), unbinds every non-bridge function of the group from vfio-pci, lets the kernel pick the host driver (or use `--host-driver`), and verifies the driver. `--probe` adds an `nvidia-smi -L` check. Any failure rebinds the functions to vfio-pci; the VM stays shut off.
- `to-vm ADDR --vm NAME` refuses, and prints the holders, if a host process has `/dev/nvidia*` open. Otherwise it unbinds the host drivers, binds vfio-pci, starts the VM and checks that it is running. A failure returns the functions to their host drivers.
- `--dry-run` on both mutating commands prints every sysfs write and command in order and changes nothing (read-only sysfs and `/proc` reads still happen, so the plan reflects the real state).

Exit codes: 0 success, 1 failed (rolled back where possible), 2 refused before any change.

A lock file (default `/run/lock/gpu-handoff.lock`, or `--lock-file`) stops two handoffs at once. Each real run appends JSON lines to an event log (default `$XDG_STATE_HOME/gpu-handoff/events.jsonl`, or `--log-file`).

## Configuration

Flags or environment variables: `GPU_HANDOFF_SYSFS_ROOT` (default `/sys`), `GPU_HANDOFF_PROC_ROOT` (`/proc`), `GPU_HANDOFF_XML_DIR` (`/etc/libvirt/qemu`), `GPU_HANDOFF_VIRSH` (`virsh`; for example `virsh -c qemu:///system`), `GPU_HANDOFF_LOCK`, `GPU_HANDOFF_LOG`. Pointing the roots at a fake tree is how the tests run; nothing in the test suite touches real hardware.

## Rule

Never run a handoff while the GPU is under load. Stop compute jobs and displays on the card first; an unbind under load can hang the driver or the host.

## Limits

- Vendor reset bugs: some cards (several AMD generations, some NVIDIA boards) do not reset cleanly between owners. The precheck only reports what `reset_method` lists; it cannot prove a reset works. A card that wedges after the guest shuts down may need a host reboot, and rollback cannot fix that.
- One-GPU hosts: if this GPU drives the host display, unbinding it kills the display session. Run headless, use the integrated GPU for the host console, or expect to lose the desktop.
- The holder scan looks only for `/dev/nvidia*` file descriptors, system-wide. It does not see DRM or `/dev/dri` users of other vendors, memory maps without an open descriptor, or which of several NVIDIA cards a process uses.
- No force-kill of the VM, no module loading, no persistence of the original driver across invocations: the host driver comes from the kernel probe or `--host-driver`.
- Groups with other endpoint devices (not bridges, not the GPU's slot) are refused rather than moved.
- The rollback is best effort; a failing rollback is logged and printed, and needs manual repair.

## License

Apache-2.0, see LICENSE.
