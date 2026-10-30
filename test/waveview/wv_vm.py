#!/usr/bin/env python3
"""Dev-only stand-in for `wv`: runs WaveView on the Parallels Windows VM. Not shipped.

Set SRP_WAVEVIEW to this file (it is executable) and waveview.py runs it as if it were
the WaveView binary: `wv_vm.py -ace_no_gui <script.tcl>`. The call is forwarded with
`prlctl exec`. The Mac home directory is visible in the VM only as \\\\Mac\\Home, so
every path (the script and the SRP_* variables) must lie under the home directory;
it is rewritten from /Users/<user>/... to \\\\Mac\\Home\\....

Environment: SRP_VM (default "Windows 11"), SRP_VM_WV (path of wv.exe in the VM).
"""
import os
import subprocess
import sys

VM = os.environ.get("SRP_VM", "Windows 11")
WV = os.environ.get("SRP_VM_WV", r"C:\synopsys\Custom WaveView O-2018.09-SP2\wv.exe")
HOME = os.path.realpath(os.path.expanduser("~"))
FORWARDED = ("SRP_IN", "SRP_OUT", "SRP_SIGNALS")     # paths
FLAGS = ("SRP_XCHECK",)                             # values, forwarded as they are


def vm_path(path: str) -> str:
    """/Users/<user>/a/b -> \\\\Mac\\Home\\a\\b; anything outside the home directory is refused."""
    full = os.path.realpath(path)
    if full != HOME and not full.startswith(HOME + os.sep):
        sys.exit(f"wv_vm.py: {path!r} is outside {HOME}; the VM cannot see it")
    rest = os.path.relpath(full, HOME)
    return "\\\\Mac\\Home\\" + rest.replace("/", "\\")


def main(argv) -> int:
    script = argv[-1]
    options = argv[:-1]
    sets = [f'set "{name}={vm_path(os.environ[name])}"' for name in FORWARDED if name in os.environ]
    sets += [f'set "{name}={os.environ[name]}"' for name in FLAGS if name in os.environ]
    command = " && ".join(sets + [f'"{WV}" {" ".join(options)} "{vm_path(script)}"'])
    proc = subprocess.run(["prlctl", "exec", VM, "cmd", "/c", command], stdin=subprocess.DEVNULL)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
