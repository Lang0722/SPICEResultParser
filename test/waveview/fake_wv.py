"""Stand-in for `wv` in the unit tests: serves real WaveView output from canned/. Not shipped.

Run as `python fake_wv.py -ace_no_gui <dir>/<script>.tcl` with SRP_IN, SRP_OUT and
optionally SRP_SIGNALS set, as waveview.run_script does. It writes
canned/<basename of SRP_IN>.<script>.txt to SRP_OUT, prints canned/<...>.console.txt
(or a plain console) to stderr, as the real WaveView on the VM does, and writes an empty
ace_tcl.log beside the script, as WaveView does. The canned files are real output of the
Tcl scripts, recorded on WaveView O-2018.09-SP2 on Windows 11 (CRLF line ends), so the
parsing is tested on the real format.

For extract.tcl and read_measures.tcl a set SRP_SIGNALS keeps only the records of the
listed signals, as the scripts do (no names in it: x only), and an unknown name gives
the error record of the scripts.

SRP_FAKE_WV selects a failure: "no_done" (the script stops on a Tcl error; exit code 1),
"silent" (no output file at all), "sleep" (never finishes), "launcher" (never finishes,
in a child process that holds the console pipe, as when wv is a launcher script; its
process ID goes to the file SRP_FAKE_PID), "lf" (LF line ends, as on Linux).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CANNED = os.path.join(HERE, "canned")
TCL_ERROR = (b"[Tcl Error] [line 12]:invalid command name \"sx_bogus\"\r\n"
             b"invalid command name \"sx_bogus\"\r\n"
             b"   *** [ERROR] ACE Tcl error at line 12 ***\r\n"
             b"  [ error ] ACE Tcl error at line 12\r\n"
             b"   *** Tcl script failed ... ***\r\n")
KEPT_RECORDS = (b"srp_extract", b"srp_measures", b"title", b"trace", b"x", b"error")


def sliced(lines, signals):
    """Keep the records of the listed signals (and the file-level records) with their values."""
    known = {line.split(b"\t", 1)[1].rstrip(b"\r\n") for line in lines
             if line.split(b"\t", 1)[0] in (b"real", b"complex", b"column")}
    for name in signals:
        if name not in known:
            return [b"error\tunknown signal " + name + b"\r\n", b"srp_done\r\n"]
    kept, keep = [], True
    for line in lines:
        if b"\t" in line:
            kind, name = line.rstrip(b"\r\n").split(b"\t", 1)
            keep = kind in KEPT_RECORDS or name in signals
        elif line.rstrip(b"\r\n") == b"srp_done":
            keep = True
        if keep:
            kept.append(line)
    return kept


def main(argv) -> int:
    script = os.path.splitext(os.path.basename(argv[-1]))[0]
    open(os.path.join(os.path.dirname(os.path.abspath(argv[-1])), "ace_tcl.log"), "wb").close()  # as WaveView
    mode = os.environ.get("SRP_FAKE_WV", "")
    source = os.environ["SRP_IN"]
    stem = os.path.join(CANNED, f"{os.path.basename(source)}.{script}")
    console = (f"   *** running Tcl script ... ***\r\n"
               f"   *** loading waveform file {source} ... ***\r\n").encode()
    if os.path.exists(stem + ".console.txt"):
        with open(stem + ".console.txt", "rb") as f:
            console = f.read()
    if mode == "sleep":
        time.sleep(60)
    if mode == "launcher":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open(os.environ["SRP_FAKE_PID"], "w") as f:
            f.write(str(child.pid))
        child.wait()
    if mode == "silent":
        sys.stderr.buffer.write(console[:console.find(b"\n") + 1])
        return 0
    with open(stem + ".txt", "rb") as f:
        lines = f.readlines()
    if "SRP_SIGNALS" in os.environ and script in ("extract", "read_measures"):
        with open(os.environ["SRP_SIGNALS"], "rb") as f:
            signals = [line.rstrip(b"\r\n") for line in f if line.strip()]
        lines = sliced(lines, signals)
    if mode == "no_done":
        lines = lines[:len(lines) // 2]
        console += TCL_ERROR
    data = b"".join(lines)
    if mode == "lf":
        data = data.replace(b"\r\n", b"\n")
    with open(os.environ["SRP_OUT"], "wb") as f:
        f.write(data)
    sys.stderr.buffer.write(console)
    return 1 if mode == "no_done" else 0       # as wv.exe after a Tcl error: the engine ignores the code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
