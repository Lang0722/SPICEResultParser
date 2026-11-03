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

For extract.tcl the fake also writes what the script writes and the canned output has
not: the npoints records (see with_counts). They come from the canned listing of the
same file, which holds WaveView's real -npoints values. It also applies SRP_XLO and
SRP_XHI (see windowed) and SRP_MAX_VALUES (see limited) as the script does.

The canned listings come from list_traces.tcl before it stopped asking for -npoints: the
fake serves their signal records without that fourth field, as the script writes them.

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
WINDOW_XNAMES = (b"TIME", b"time", b"HERTZ", b"frequency")    # WaveView gave the x unit s or Hz for these
# extract.tcl also gives no window to a file with a frequency that goes down in a trace. No
# canned file has one: test_waveview_e2e.TestWindow.test_frequency_out_of_order checks it.


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


def listed_counts(source):
    """(name, -npoints) of each signal of the canned listing of source, in its order."""
    path = os.path.join(CANNED, f"{os.path.basename(source)}.list_traces.txt")
    counts = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                fields = line.rstrip(b"\r\n").split(b"\t")
                if fields[0] == b"signal" and len(fields) > 3 and fields[3].isdigit():
                    counts.append((fields[1], fields[3]))
    return counts


def with_counts(lines, source, signals):
    """Add the npoints records before srp_done, as extract.tcl writes them: the x signal and
    each written signal (signals None: every signal; no names: x only, the first signal), or
    every signal with SRP_NPOINTS=all. An error output gets none."""
    if any(line.startswith(b"error\t") for line in lines):
        return lines
    counts = listed_counts(source)
    if os.environ.get("SRP_NPOINTS") == "all" or signals is None:
        names = {name for name, _ in counts}
    elif not signals:
        names = {name for name, _ in counts[:1]}
    else:
        names = set(signals)
    records = [b"npoints\t" + name + b"\t" + n + b"\r\n" for name, n in counts if name in names]
    end = len(lines) - 1 if lines and lines[-1].rstrip(b"\r\n") == b"srp_done" else len(lines)
    return lines[:end] + records + lines[end:]


def canned_xname(source):
    """The xname record of the canned listing of source (b"" when there is none)."""
    path = os.path.join(CANNED, f"{os.path.basename(source)}.list_traces.txt")
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                if line.startswith(b"xname\t"):
                    return line.rstrip(b"\r\n").split(b"\t", 1)[1]
    return b""


def windowed(lines, lo, hi):
    """Cut each trace to the window [lo, hi] (None: open) as extract.tcl does. A scan starts
    at point 0 when one of the first 8 points has x >= lo, else at the point before the first
    such point, and stops before the first x above hi. A span record follows each trace
    record. A signal whose values end inside the window gets the stopped record."""
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        i += 1
        if not out[-1].startswith(b"trace\t"):
            continue
        records = []                                    # (record line, value lines) of the trace
        while i < len(lines) and lines[i].split(b"\t", 1)[0] in (b"x", b"real", b"complex"):
            head, i = lines[i], i + 1
            values = []
            while i < len(lines) and b"\t" not in lines[i] and lines[i].rstrip(b"\r\n") != b"srp_done":
                values.append(lines[i])
                i += 1
            records.append((head, values))
        xs = next((v for head, v in records if head.startswith(b"x\t")), [])
        x = [float(v) for v in xs]
        k = len(x) if lo is None else next((j for j, v in enumerate(x) if v >= lo), len(x))
        start = 0 if lo is None or k < 8 else k - 1
        stop = len(x) if hi is None else next((j for j in range(start, len(x)) if x[j] > hi), len(x))
        ends = [v.rstrip(b"\r\n") for v in xs[:1] + xs[-1:]] if xs else [b"", b""]
        out.append(b"span\t" + ends[0] + b"\t" + ends[-1] + b"\r\n")
        for head, values in records:
            kept = values[start:stop]
            out += [head] + kept
            if kept and len(values) < stop and not head.startswith(b"x\t"):
                out.append(b"stopped\t" + head.rstrip(b"\r\n").split(b"\t", 1)[1] + b"\r\n")
    return out


def limited(lines, limit):
    """Stop before a row would go past the limit, as extract.tcl does: a point of the x
    column counts its whole row (x, 1 per real and 2 per complex signal of the trace)."""
    widths = {}
    trace = None
    for line in lines:
        kind, _, value = line.rstrip(b"\r\n").partition(b"\t")
        if kind == b"trace":
            trace = value
            widths[trace] = 1
        elif kind in (b"real", b"complex") and trace is not None:
            widths[trace] += 1 if kind == b"real" else 2
    kept, count, trace, in_x, last = [], 0, None, False, b""
    for line in lines:
        text = line.rstrip(b"\r\n")
        if b"\t" in text:
            kind, _, value = text.partition(b"\t")
            if kind == b"trace":
                trace, last = value, b""
            in_x = kind == b"x"
        elif in_x and text != b"srp_done":
            if count + widths[trace] > limit:
                return kept + [b"limit\t%d\t%s\t%s\r\n" % (count, trace, last), b"srp_done\r\n"]
            count += widths[trace]
            last = text
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
    if script == "list_traces":         # the count of the old script (see the docstring)
        lines = [b"\t".join(line.rstrip(b"\r\n").split(b"\t")[:3]) + b"\r\n" if line.startswith(b"signal\t")
                 else line for line in lines]
    signals = None
    if "SRP_SIGNALS" in os.environ and script in ("extract", "read_measures"):
        with open(os.environ["SRP_SIGNALS"], "rb") as f:
            signals = [line.rstrip(b"\r\n") for line in f if line.strip()]
        lines = sliced(lines, signals)
    if script == "extract":
        lines = with_counts(lines, source, signals)
        bounds = [float(os.environ[key]) if key in os.environ else None for key in ("SRP_XLO", "SRP_XHI")]
        if bounds != [None, None] and canned_xname(source) in WINDOW_XNAMES:
            lines = windowed(lines, *bounds)
        if "SRP_MAX_VALUES" in os.environ:
            lines = limited(lines, int(os.environ["SRP_MAX_VALUES"]))
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
