"""WaveView engine: read result files through Synopsys Custom WaveView in batch mode.

A second reader beside reader.py, nutmeg.py and measure.py, for a file the Python
parser cannot read (HSPICE post_version 2013, for example) or a result in doubt. It
runs one of the Tcl scripts in tcl/ with `wv -ace_no_gui`, reads the text file the
script writes, and builds the same Header, TraceSet and MeasureSet, so api.py and
mcp_server.py treat both engines alike.

Inputs reach Tcl in environment variables (SRP_IN, SRP_OUT, SRP_SIGNALS), never in
Tcl source. The exit code of WaveView does not show success (it is 0 after an error
that the script catches), so every script ends its output with `srp_done`; a missing
srp_done is a failure.

Differences from the Python engine: every point of the selected signals is read
into memory, and xrange is applied after the read; only plot 0 of a Nutmeg rawfile;
arrays are float64 (also for 9601 files); a measure file has no `alter#` column; a
truncated binary HSPICE file loses its partial sweep. Names follow the Python engine
(see raw_name). What WaveView does not report comes from the file header, through
reader.py and nutmeg.py when they can read it: the Nutmeg x name, Dimensions and point
count, the float32 x of a 9601 file, the analysis and sweep count of an HSPICE header,
and the names of named probes and expressions (vo=v(out): the file has vo, WaveView
lists v(vo)). Without that header (post_version 2013) a named probe keeps WaveView's
name. WaveView shows a truncated file on its console ("waveform file ... not complete").
"""
from __future__ import annotations

import atexit
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import measure, nutmeg, reader
from .reader import (Header, TraceSet, _analysis_from_extension, _uniquify, ac_names, ac_part, check_ac_format,
                     check_sweeps, parse_var_name, plot_index, resolve_columns, sweep_set, validate_range)

ENGINES = ("python", "waveview")
DEFAULT_TIMEOUT = 600.0         # seconds one WaveView run may take; SRP_WAVEVIEW_TIMEOUT overrides it
MAX_TIMEOUT = 1e6               # subprocess cannot wait much longer (2**31 ms on POSIX)
TCL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcl")
CONSOLE_TAIL = 10               # console lines quoted in an error when none names an error
NUTMEG_LIMIT = 1e30             # WaveView sets a larger Nutmeg value to +-1e30

_SCRIPT_VARIABLES = ("SRP_IN", "SRP_OUT", "SRP_SIGNALS", "SRP_XCHECK")   # set by _run only, never by the caller

_ERROR_LINE = re.compile(r"ERROR|error|Tcl Error")
_ERROR_RECORD = re.compile(r"^error\t(.*)$", re.M)
_INCOMPLETE = re.compile(r"waveform file '.*' not complete")       # WaveView's console line for a cut file
_RECORD = re.compile(r"^(\w+)\t(.*)$", re.M)                        # a record line; value lines have no tab
_WINDOWS_SPECIAL = re.compile(r"(-?)1\.#(QNAN|SNAN|IND|INF)\d*")      # NaN / infinity as Windows prints them
_HIERARCHY = re.compile(r"^((?:[^()]+\.)+)([A-Za-z][A-Za-z0-9_]*)\((.*)\)$")
_BARE = re.compile(r"^(?:par|deriv|integ|noise)\(([A-Za-z_][\w.$#\[\]<>:]*)\)$")   # WaveView's name of a bare name
_TITLE_LINE = re.compile(r"^[ \t]*\.TITLE(.*)$", re.M | re.I)

_running: set = set()           # process IDs of the WaveView runs in progress (see stop_all)


class WaveViewError(RuntimeError):
    """WaveView could not be run, or a script did not finish; the text quotes WaveView's console."""


def resolve_engine(engine: Optional[str] = None) -> str:
    """The engine to use: `engine` if given, else the SRP_ENGINE environment variable, else "python"."""
    if engine is None:
        env = os.environ.get("SRP_ENGINE", "").strip()
        if env and env not in ENGINES:
            raise ValueError(f"SRP_ENGINE must be one of {ENGINES}, not {env!r}")
        return env or "python"
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")
    return engine


def find_wv() -> str:
    """Path of the WaveView binary: SRP_WAVEVIEW if set, else `wv` on the PATH."""
    path = os.environ.get("SRP_WAVEVIEW", "").strip() or shutil.which("wv")
    if not path:
        raise WaveViewError("WaveView not found: set SRP_WAVEVIEW to the path of the wv (wv.exe) binary, "
                            "or put wv on the PATH")
    return path


def wv_command() -> List[str]:
    """The command that starts WaveView, before its arguments."""
    return [find_wv()]


def _timeout() -> float:
    text = os.environ.get("SRP_WAVEVIEW_TIMEOUT", "").strip()
    if not text:
        return DEFAULT_TIMEOUT
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and 0 < value <= MAX_TIMEOUT):
        raise ValueError(f"SRP_WAVEVIEW_TIMEOUT must be a positive number of seconds, at most {MAX_TIMEOUT:g}, "
                         f"not {text!r}")
    return value


def _console_lines(console: str) -> List[str]:
    """The console lines that name an error, else the last CONSOLE_TAIL lines."""
    lines = [line.rstrip() for line in console.splitlines() if line.strip()]
    return [line for line in lines if _ERROR_LINE.search(line)] or lines[-CONSOLE_TAIL:]


def _failure(script: str, message: str, console: str) -> WaveViewError:
    lines = _console_lines(console)
    text = f"WaveView ({script}): {message}"
    if lines:
        text += "\nWaveView console:\n" + "\n".join("  " + line.strip() for line in lines)
    return WaveViewError(text)


def run_script(name: str, env: Dict[str, str], directory: Optional[str] = None) -> str:
    """Run the Tcl script <directory>/<name> (default directory: TCL_DIR) in WaveView batch
    mode with `env` added to the environment; returns the console text (stdout and stderr).
    The only place that starts WaveView. SRP_IN, SRP_OUT and SRP_SIGNALS come from `env`
    only, never from the caller's environment: a stale SRP_SIGNALS there would select columns."""
    command = wv_command() + ["-ace_no_gui", os.path.join(directory or TCL_DIR, name)]
    timeout = _timeout()
    base = {k: v for k, v in os.environ.items() if k.upper() not in _SCRIPT_VARIABLES}
    try:
        # A new session on POSIX, so that _kill stops also what a wv launcher script starts.
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env={**base, **env}, start_new_session=os.name != "nt")
    except OSError as exc:
        raise WaveViewError(f"cannot start WaveView {command[0]!r}: {exc}") from None
    _running.add(proc.pid)
    try:
        with proc:
            try:
                output, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill(proc)
                try:
                    output, _ = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:   # a process outside the tree holds the pipe
                    output = b""
                    if os.name == "nt":             # a reader thread holds the pipe: closing it would wait
                        proc.stdout = None
                raise _failure(name, f"no result after {timeout:g} s; set SRP_WAVEVIEW_TIMEOUT to wait longer",
                               output.decode("utf-8", errors="replace")) from None
            except BaseException:                   # KeyboardInterrupt: the new session does not get it
                _kill(proc)
                raise
    finally:
        _running.discard(proc.pid)
    return output.decode("utf-8", errors="replace")


def _kill_tree(pid: int) -> None:
    """Stop the process `pid` and the processes it started (on POSIX: its session)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.killpg(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass


def _kill(proc: subprocess.Popen) -> None:
    """Stop WaveView and the processes it started. subprocess kills only its own child,
    which can be a launcher script; the real WaveView would then go on and hold the pipe."""
    _kill_tree(proc.pid)
    proc.kill()


def stop_all() -> None:
    """Stop every WaveView run in progress, with the processes it started. WaveView runs
    in its own session on POSIX, so a signal to the caller's process group does not reach
    it: a caller that a signal stops calls this first (mcp_server.main does, on SIGTERM
    and SIGHUP). It also runs at the exit of the interpreter."""
    for pid in list(_running):
        _kill_tree(pid)


atexit.register(stop_all)


def _run(name: str, path: str, signals: Optional[List[str]] = None, xcheck: bool = False) -> Tuple[str, int, str]:
    """Run a script on `path`; return its output, the offset of the final srp_done, where
    the records end, and the console. The text is not sliced: an extract output can be large.

    signals: exact WaveView names written to SRP_SIGNALS, one per line; None leaves it unset.
    xcheck: set SRP_XCHECK (extract.tcl checks that all signals have the same x points).
    Raises WaveViewError when the output has no srp_done or holds an error record.
    """
    # A WaveView that outlives a timeout can hold files in tmp (Windows): the clean-up must
    # not replace the timeout error, so it ignores errors.
    tmp = tempfile.mkdtemp(prefix="srp_wv_")
    try:
        # WaveView writes ace_tcl.log beside the script it runs. Run a copy, so that the log
        # goes to tmp and not to the package directory.
        shutil.copy(os.path.join(TCL_DIR, name), tmp)
        out = os.path.join(tmp, "out.txt")
        # On POSIX the file that the kernel opens: abspath removes "a/link/.." as text, and
        # the Python reads of the header open the path as it is given.
        full = os.path.abspath(path) if os.name == "nt" else os.path.realpath(path)
        env = {"SRP_IN": full, "SRP_OUT": out}
        if xcheck:
            env["SRP_XCHECK"] = "1"
        if signals is not None:
            env["SRP_SIGNALS"] = os.path.join(tmp, "signals.txt")
            with open(env["SRP_SIGNALS"], "w", encoding="utf-8", newline="\n") as f:
                f.write("".join(s + "\n" for s in signals))
        console = run_script(name, env, tmp)
        try:
            with open(out, encoding="utf-8", errors="replace") as f:    # universal newlines: CRLF on Windows
                text = f.read()
        except FileNotFoundError:
            text = ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    end = text.rfind("srp_done")
    if end < 0 or (end and text[end - 1] != "\n") or text[end + len("srp_done"):].strip("\n"):
        message = f"the script stopped before its end (no srp_done) on {path}"
        if not tmp.isascii():   # on Windows such characters can reach WaveView as "?"
            message += (f" (the temporary folder {tmp} has non-ASCII characters, which WaveView may not "
                        f"read: set TMP (Windows) or TMPDIR to a folder with an ASCII path)")
        raise _failure(name, message, console)
    error = _ERROR_RECORD.search(text, 0, end)
    if error:
        message = error.group(1)
        if not env["SRP_IN"].isascii() and "non-ASCII" not in message:
            # On Windows such characters can reach WaveView as "?", so the script cannot tell.
            message += " (the path has non-ASCII characters, which WaveView may not read)"
        raise _failure(name, message, console)
    return text, end, console


def _check_input(path: str, plot) -> None:
    """Raise before WaveView starts: a missing file, a measure file, or a plot other than 0."""
    with open(path, "rb") as f:                     # FileNotFoundError, IsADirectoryError, as the Python engine
        head = f.read(16)
    if not head:
        raise ValueError(f"{path}: empty file")
    if head.startswith(b"$DATA1"):
        raise ValueError(f"{path}: this is an HSPICE measure file (.mt/.ms/.ma); read it with read_measures")
    if plot_index(plot):
        raise ValueError(f"{path}: plot={plot}: the WaveView engine reads only plot 0 "
                         f"(WaveView opens only the first plot of a Nutmeg rawfile)")


def raw_name(wv_name: str, is_nutmeg: bool) -> str:
    """The name the file itself uses, from WaveView's name.

    WaveView writes a hierarchical signal as x1.x2.v(n1), and takes out the path that all
    nodes share: x1.v(a,b); the file has v(x1.x2.n1 and v(x1.a,x1.b. In an HSPICE file
    WaveView also writes p(r9) as power(r9), and adds a type to a bare name: the named
    expression vm=par('...') as par(vm), dv=deriv(...) as deriv(dv), ig=integ(...) as
    integ(ig), the .noise columns as noise(outnoise). HSPICE names have no closing paren
    (v(out), Nutmeg names keep it (v(out)), as in the Python engine. Without the file's
    header an unnamed par('x') cannot be told from x=par(...): both come out as x.
    """
    name = _unfold(wv_name)
    if is_nutmeg:
        return name
    bare = _BARE.match(name)
    if bare:
        return bare.group(1)
    if name.startswith("power("):
        name = "p(" + name[len("power("):]
    if "(" in name and name.endswith(")"):
        name = name[:-1]
    return name


def _unfold(wv_name: str) -> str:
    """WaveView's x1.x2.v(n1) and x1.v(a,b) -> v(x1.x2.n1) and v(x1.a,x1.b)."""
    match = _HIERARCHY.match(wv_name)
    if not match:
        return wv_name
    prefix, function, nodes = match.groups()
    return f"{function}(" + ",".join(prefix + node for node in nodes.split(",")) + ")"


def _is_own_name(unfolded: str, own: str) -> bool:
    """Whether the header name `own` names the WaveView signal `unfolded` (see _unfold).
    The file has a bare name for a named probe or expression, which WaveView lists with
    a type: vo as v(vo), x1.vo as x1.v(vo), pw as power(pw), vm as par(vm). An unnamed
    expression has its type in the file too: par(rv as par(rv)."""
    return unfolded == own + ")" or ("(" not in own and unfolded.endswith(f"({own})"))


def _shared_x(path: str) -> bool:
    """Whether the format gives one x to all signals: an HSPICE post=1 or post=2 file
    (binary: a block head; ASCII: the digits of the header). Not so in a post=3 file."""
    with open(path, "rb") as f:
        head = f.read(4)
    return head in (b"\x04\x00\x00\x00", b"\x00\x00\x00\x04") or head.isdigit()


@dataclass
class _Listing:
    """What list_traces.tcl and the file header report beyond the Header."""
    params: List[str]                           # sweep parameter names as WaveView writes them
    conditions: List[List[str]]                 # one per trace: each parameter's value (%g); [] when not swept
    columns: List[Tuple[Optional[str], Optional[int]]]   # per Header column: (WaveView name, part)
                                                # part: None real, 0/1 the first/second column of a complex one
    first: str = ""                             # WaveView name of the first signal (extract.tcl's x when no
                                                # signal is selected)
    npoints: Dict[str, int] = field(default_factory=dict)   # WaveView name -> -npoints (one more per sweep)
    incomplete: bool = False                    # WaveView's console says the file is not complete
    binary: bool = False                        # an HSPICE binary file: WaveView drops a partial sweep
    nutmeg: bool = False
    shared_x: bool = True                       # the format gives one x to all signals (see _shared_x)
    nsplit: int = 1                             # Nutmeg: sweeps of the Dimensions line in WaveView's one trace
    nut_points: int = 0                         # Nutmeg: points of plot 0 in its header (0: not known)
    nut_complete: Optional[int] = None          # Nutmeg, a file that is not complete: the complete points of
                                                # plot 0, as nutmeg.py counts them (None: not known)
    float32: bool = False                       # HSPICE 9601: the file stores x as float32
    declared: int = 0                           # sweeps the HSPICE header declares (0: not swept or not known)

    @property
    def ntraces(self) -> int:
        """Traces that WaveView writes: one per sweep, 0 when a swept file has no complete sweep."""
        return len(self.conditions) if self.params else 1

    @property
    def nsweeps(self) -> int:
        """Sweeps of the TraceSet."""
        return self.ntraces * self.nsplit


def _records(text: str, script: str, magic: str) -> List[List[str]]:
    records = [line.split("\t") for line in text.splitlines() if line]
    if not records or records[0][0] != magic:
        raise WaveViewError(f"WaveView ({script}): the output does not start with {magic!r}")
    return records[1:]


def _nutmeg_complete(path: str, p: nutmeg._Plot) -> int:
    """The complete points of plot `p` as nutmeg.py counts them. A cut file can end inside a
    point, and in an ASCII file a line without its newline counts as no value."""
    with open(path, "rb") as f:
        if p.is_binary:
            n = max(0, os.fstat(f.fileno()).st_size - p.data_start) // p.point_bytes
            return min(n, p.npoints) if p.count_known else n
        f.seek(p.data_start)
        return nutmeg._scan_ascii(f, path, p, p.npoints if p.count_known else None)


def _nutmeg_plot0(path: str) -> Optional[nutmeg._Plot]:
    """The header of plot 0 as nutmeg.py reads it; None when it cannot."""
    try:
        with open(path, "rb") as f:
            if f.read(len(nutmeg._BOM)) != nutmeg._BOM:
                f.seek(0)
            return nutmeg._parse_plot_header(f, path, 0)
    except Exception:           # any failure of the Python parser: this engine is its backup
        return None


def _hspice_header(path: str) -> Optional[Header]:
    """The header as reader.py reads it; None when it cannot (post_version 2013, for example).
    Its warnings pass: a warnings filter is not safe in threads."""
    try:
        return reader.read_header(path)
    except Exception:           # any failure of the Python parser: this engine is its backup
        return None


def _listing(path: str, plot, ac_format: str) -> Tuple[Header, _Listing]:
    path = os.fspath(path)
    check_ac_format(ac_format)
    _check_input(path, plot)
    title, xname, params, conditions, signals, npoints = "", "", [], [], [], {}
    text, end, console = _run("list_traces.tcl", path)
    for record in _records(text[:end], "list_traces.tcl", "srp_listing"):
        key, value = record[0], record[1] if len(record) > 1 else ""
        if key == "title":
            title = value
        elif key == "xname":
            xname = value
        elif key == "param":
            params.append(value)
        elif key == "sweep":
            conditions.append(record[1:])
        elif key == "signal":
            signals.append((value, record[2] if len(record) > 2 else "real"))
            if len(record) > 3 and record[3].isdigit():
                npoints[value] = int(record[3])
    seen = set()
    for wv_name, _ in signals:
        if wv_name in seen:     # extract.tcl addresses signals by name: both columns would get one signal
            raise WaveViewError(f"{path}: WaveView lists two signals named {wv_name}; the WaveView engine "
                                f"cannot tell them apart (read the file with engine='python')")
        seen.add(wv_name)
    if not signals:             # WaveView opens an empty file, and a file of zero bytes, without an error
        raise WaveViewError(f"{path}: WaveView finds no signals in the file (an empty file, or not a "
                            f"waveform file)")
    is_nut = nutmeg.is_nutmeg(path)
    incomplete = bool(_INCOMPLETE.search(console))
    nsplit, nut_points, nut_complete, float32, declared, own = 1, 0, None, False, 0, None
    if is_nut:
        plot0 = _nutmeg_plot0(path)
        if plot0 is not None:
            xname = plot0.raw_names[0]          # WaveView's xname is the type of variable 0 (voltage)
            nsplit = plot0.nsweeps
            nut_points = plot0.npoints if plot0.count_known else 0
            if incomplete:
                nut_complete = _nutmeg_complete(path, plot0)
    else:
        own = _hspice_header(path)
        if own is not None:
            float32 = own.dtype == np.float32
            declared = own.sweep_count_hint
        if not xname:                           # WaveView gives no x name for a post=3 file
            xname = {"tr": "TIME", "ac": "HERTZ"}.get(_analysis_from_extension(path), "x")
    sanitize = nutmeg._sanitize if is_nut else parse_var_name
    names, raw_names, col_raw = [sanitize(xname)], [xname], [xname]
    columns: List[Tuple[Optional[str], Optional[int]]] = [(None, None)]
    # A named probe (.probe vo=v(out)) has a bare name in the file, and WaveView lists v(vo).
    # Only such names come from the header: reader.py has bugs that WaveView does not have.
    own_names = own.raw_names[1:] if own is not None and len(own.raw_names) == len(signals) + 1 else None
    for i, (wv_name, datatype) in enumerate(signals):
        raw = raw_name(wv_name, is_nut)
        if own_names and _is_own_name(_unfold(wv_name), own_names[i]):
            raw = own_names[i]
        raw_names.append(raw)
        if datatype == "complex":
            names += ac_names(sanitize(raw), ac_format)
            col_raw += [raw, raw]
            columns += [(wv_name, 0), (wv_name, 1)]
        else:
            names.append(sanitize(raw))
            col_raw.append(raw)
            columns.append((wv_name, None))
    has_complex = any(datatype == "complex" for _, datatype in signals)
    if is_nut:
        analysis = nutmeg._analysis(title)
        if analysis == "other" and has_complex:
            analysis = "ac"
    else:
        analysis = ((own.analysis if own is not None else None) or _analysis_from_extension(path)
                    or ("ac" if has_complex or xname.upper() == "HERTZ" else None)
                    or ("tr" if xname.upper() == "TIME" else "sw"))
    header = Header(
        path=path, is_binary=False, version="waveview", analysis=analysis, ncols=len(names), nsweepparam=len(params),
        sweep_count_hint=(nsplit if nsplit > 1 else 0) if is_nut else max(len(conditions), declared), x_name=names[0],
        names=_uniquify(path, names), raw_names=raw_names, col_raw_names=col_raw, type_codes=[],
        sweep_params=[parse_var_name(p) for p in params], plot_names=[title] if is_nut else [], plot=0,
        ac_format=ac_format,
    )
    return header, _Listing(params=params, conditions=conditions, columns=columns, first=signals[0][0],
                            npoints=npoints, incomplete=incomplete,
                            binary=not is_nut and reader._is_binary(path), nutmeg=is_nut,
                            shared_x=is_nut or _shared_x(path), nsplit=nsplit, nut_points=nut_points,
                            nut_complete=nut_complete, float32=float32, declared=declared)


def read_header(path, plot: int = 0, ac_format: str = "magphase") -> Header:
    """The Header of a result file, as WaveView lists it (see reader.read_header)."""
    return _listing(path, plot, ac_format)[0]


def _special(match) -> str:
    return match.group(1) + ("inf" if match.group(2) == "INF" else "nan")


def _floats(text: str, path: str, per_line: int = 1) -> np.ndarray:
    """The numbers in text, per_line on each line, as float64; also NaN and infinity in
    the forms Windows prints. Some NumPy versions (2.1, for example) only warn about a
    value that np.fromstring cannot read, and stop there. A warnings filter is not safe
    in threads, so the count of the values is checked instead."""
    if not text or text.isspace():
        return np.empty(0)                          # np.fromstring reads white space alone as [-1.0]
    if "#" in text:
        text = _WINDOWS_SPECIAL.sub(_special, text)
    count = (text.count("\n") + 1 - text.startswith("\n") - text.endswith("\n")) * per_line
    try:
        values = np.fromstring(text, dtype=np.float64, sep=" ")    # no list of tokens: a column can be large
    except ValueError:
        values = None
    if values is not None and values.size == count:
        return values
    try:
        values = np.array(text.split(), dtype=np.float64)          # names the value
    except ValueError as exc:
        raise WaveViewError(f"{path}: WaveView wrote a value that is not a number ({exc})") from None
    if values.size != count:
        raise WaveViewError(f"{path}: WaveView wrote {values.size} values where {count} were expected "
                            f"({per_line} per line)")
    return values


def _parse_points(text: str, end: int, path: str) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """extract.tcl output up to `end` -> per trace: (x, {WaveView name: values}); complex values have shape (n, 2)."""
    if not text.startswith("srp_extract\t"):
        raise WaveViewError("WaveView (extract.tcl): the output does not start with 'srp_extract'")
    marks = list(_RECORD.finditer(text, 0, end))
    traces: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []
    for i, mark in enumerate(marks[1:], 1):
        kind, name = mark.group(1), mark.group(2)
        if kind == "trace":
            traces.append((np.empty(0), {}))
            continue
        if not traces:
            raise WaveViewError(f"WaveView (extract.tcl): record {kind!r} before the first trace")
        stop = marks[i + 1].start() if i + 1 < len(marks) else end
        values = _floats(text[mark.end():stop], path, 2 if kind == "complex" else 1)
        if kind == "x":
            traces[-1] = (values, traces[-1][1])
        elif kind == "complex":
            traces[-1][1][name] = values.reshape(-1, 2)
        else:
            traces[-1][1][name] = values
    return traces


def _split(trace: Tuple[np.ndarray, Dict[str, np.ndarray]], nsplit: int, npoints: int,
           path: str) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """WaveView's one trace of a Nutmeg plot -> the sweeps of its Dimensions line, of
    npoints // nsplit points each. A truncated file keeps the sweeps that it reaches, the
    last one partial, as nutmeg.py reads it."""
    x, ys = trace
    npoints = npoints or x.size
    if x.size > npoints or any(len(v) != x.size for v in ys.values()):
        raise WaveViewError(f"{path}: WaveView wrote {x.size} points, and the header of the plot has {npoints}")
    per = npoints // nsplit
    return [(x[k * per:(k + 1) * per], {name: v[k * per:(k + 1) * per] for name, v in ys.items()})
            for k in range(max(1, -(-x.size // per)))]


def _warn_truncated(path: str, listing: _Listing, found: str) -> None:
    lost = "WaveView drops its partial sweep" if listing.binary and listing.params else "its last sweep is partial"
    warnings.warn(f"{path}: {found}: the file is probably truncated, and {lost}", RuntimeWarning)


def _check_counts(path: str, listing: _Listing, traces, signals: List[str]) -> None:
    """The points of a signal must be WaveView's count of them, less one per sweep. WaveView
    ends a scan early after some values of about 1e34 or more. A file that is not complete
    can have more: a cut ASCII file has one more, and the file can grow between the runs."""
    extra = listing.ntraces if listing.params else 0
    counts = [(signals[0] if signals else listing.first, sum(x.size for x, _ in traces))]
    counts += [(name, sum(len(ys.get(name, ())) for _, ys in traces)) for name in signals]
    for name, count in counts:
        want = listing.npoints.get(name)
        if want is not None and (count < want - extra or (count > want - extra and not listing.incomplete)):
            raise WaveViewError(f"{path}: WaveView reads {count} of the {want - extra} points of {name}: WaveView "
                                f"stops early on some values of about 1e34 or more, so the engine cannot read this "
                                f"signal (read the file with engine='python', or leave out the signal)")


def _condition_values(values: List[str], params: List[str], path: str) -> List[float]:
    """One sweep record of list_traces.tcl (each parameter's value, %g) -> floats, in the order of params."""
    if len(values) != len(params):
        raise WaveViewError(f"{path}: WaveView wrote {len(values)} sweep values for the {len(params)} "
                            f"sweep parameters ({', '.join(params)})")
    floats = []
    for value in values:
        try:
            floats.append(float(value))
        except ValueError:
            raise WaveViewError(f"{path}: cannot read sweep value {value!r}") from None
    return floats


def read_traces(path, names=None, sweeps=None, plot: int = 0, xrange=None,
                ac_format: str = "magphase") -> TraceSet:
    """Read selected traces through WaveView (see reader.read_traces for the arguments).

    Two WaveView runs: list_traces.tcl for the names, then extract.tcl for every point of
    the selected signals. The sweeps filter and the closed xrange window are applied
    afterwards, so memory follows the selected signals times all their points.
    """
    path = os.fspath(path)
    check_ac_format(ac_format)
    xrange = validate_range("xrange", xrange)
    wanted = sweep_set(sweeps)
    header, listing = _listing(path, plot, ac_format)
    cols = resolve_columns(header, names)
    if listing.declared > listing.ntraces:
        found = f"the header declares {listing.declared} sweeps and WaveView reads {listing.ntraces}"
    elif not listing.ntraces:
        found = "WaveView reads no complete sweep"
    elif listing.incomplete:
        found = "WaveView reports that the file is not complete"
    else:
        found = ""
    truncated = bool(found)
    if truncated:
        _warn_truncated(path, listing, found)
    check_sweeps(path, wanted, listing.nsweeps, truncated)
    if not listing.ntraces:     # extract.tcl would read the file as one trace without points
        return TraceSet(header=header, selected=[header.names[c] for c in cols], sweep_values=[], sweep_indices=[],
                        data={header.names[c]: [] for c in cols}, truncated=True)
    signals: List[str] = []
    for c in cols[1:]:
        wv_name = listing.columns[c][0]
        if wv_name not in signals:
            signals.append(wv_name)
    text, end, _ = _run("extract.tcl", path, signals, xcheck=not listing.shared_x)
    traces = _parse_points(text, end, path)
    del text
    if len(traces) > listing.ntraces:   # the simulator ended a sweep between the two WaveView runs
        del traces[listing.ntraces:]
    if len(traces) != listing.ntraces:
        raise WaveViewError(f"{path}: WaveView wrote {len(traces)} traces; its listing has {listing.ntraces} "
                            f"(the file can have changed between the two WaveView runs)")
    _check_counts(path, listing, traces, signals)
    if listing.incomplete:
        # The last row can be partial: keep the points that every signal of the file has.
        # -npoints shows the signals that have a point of that row. WaveView also reads a
        # value that the cut makes short, so a Nutmeg plot keeps the count of nutmeg.py.
        low = min(listing.npoints.values(), default=0)
        x, ys = traces[-1]
        sizes = [x.size - listing.npoints.get(signals[0] if signals else listing.first, low) + low]
        sizes += [len(v) - listing.npoints.get(name, low) + low for name, v in ys.items()]
        if listing.nut_complete is not None:
            sizes.append(listing.nut_complete)
        n = max(0, min(sizes))
        traces[-1] = (x[:n], {name: v[:n] for name, v in ys.items()})
    if listing.nutmeg:
        if any(np.any(np.abs(a) == NUTMEG_LIMIT) for x, ys in traces for a in (x, *ys.values())):
            raise WaveViewError(f"{path}: WaveView changes a value of a Nutmeg file above {NUTMEG_LIMIT:g} to "
                                f"+-{NUTMEG_LIMIT:g}, and an infinite value changes the whole plot. This file holds "
                                f"such a value: read it with engine='python'")
        npoints = traces[0][0].size
        if 0 < npoints < listing.nut_points and not truncated:
            truncated = True
            _warn_truncated(path, listing, f"WaveView reads {npoints} of the {listing.nut_points} points of the plot")
    if listing.nsplit > 1:
        traces = _split(traces[0], listing.nsplit, listing.nut_points, path)
        if len(traces) < listing.nsplit:
            check_sweeps(path, wanted, len(traces), True)
    sweep_values = [_condition_values(c, listing.params, path) for c in listing.conditions] or [[] for _ in traces]
    if xrange is not None and listing.float32:
        with np.errstate(over="ignore"):    # as the Python engine compares: float32 x with float32 bounds
            xrange = (float(np.float32(xrange[0])), float(np.float32(xrange[1])))
    data: Dict[str, list] = {header.names[c]: [] for c in cols}
    kept: List[int] = []
    for i, (x, ys) in enumerate(traces):
        if wanted is not None and i not in wanted:
            continue
        kept.append(i)
        mask = None
        if xrange is not None:
            mask = (x >= xrange[0]) & (x <= xrange[1])
        for c in cols:
            wv_name, part = listing.columns[c]
            if c == 0:
                column = x
            else:
                values = ys.get(wv_name)
                if values is None or (values.ndim == 2) != (part is not None):
                    raise WaveViewError(f"{path}: WaveView wrote no {'complex' if part is not None else 'real'} "
                                        f"values for {wv_name} in trace {i}")
                if part is None:
                    column = values
                else:
                    re_, im = values[:, 0], values[:, 1]
                    column = ac_part(re_, im, ac_format, second=part == 1)
            if column.size != x.size:
                why = ("the file is probably truncated" if truncated else
                       "the signals have their own x points; extract one signal at a time")
                raise WaveViewError(f"{path}: WaveView wrote {column.size} points for {wv_name} and "
                                    f"{x.size} x values in trace {i}: {why}")
            data[header.names[c]].append(column if mask is None else column[mask])
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=[sweep_values[i] for i in kept],
        sweep_indices=kept,
        data=data,
        truncated=truncated,
    )


def read_measures(path, names=None) -> measure.MeasureSet:
    """Read an HSPICE measure file through WaveView (see measure.read_measures).

    The columns are those WaveView lists: `alter#` is not one. The file text, read with
    the rules of measure.read_measures, gives what WaveView does not: PARAM_COUNT, the
    whole title (WaveView cuts a long one), and the count of the values. That count
    shows a file that ends part-way through a row: WaveView reads that row too, also a
    value that the end of the file cuts short. A NaN value counts as failed.
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        first = f.readline(4096).decode("utf-8", errors="replace")
        if not first.startswith("$DATA1"):
            raise ValueError(f"{path}: not an HSPICE measure file (it does not start with $DATA1)")
        rest = f.read().decode("utf-8", errors="replace")
    match = measure._PARAM_COUNT.search(first)
    param_count = int(match.group(1)) if match else 0
    title_line = _TITLE_LINE.search(rest)
    tokens = rest[title_line.end() if title_line else 0:].split()
    text, end, _ = _run("read_measures.tcl", path)
    if not text.startswith("srp_measures\t"):
        raise WaveViewError("WaveView (read_measures.tcl): the output does not start with 'srp_measures'")
    marks = list(_RECORD.finditer(text, 0, end))
    title = ""
    wv_names: List[str] = []
    arrays: List[np.ndarray] = []
    for i, mark in enumerate(marks):
        if mark.group(1) == "title":
            title = mark.group(2)
        elif mark.group(1) == "column":
            stop = marks[i + 1].start() if i + 1 < len(marks) else end
            wv_names.append(mark.group(2))
            arrays.append(_floats(text[mark.end():stop], path))
    if title_line:
        title = title_line.group(1).strip().strip("'")
    nrows = min((a.size for a in arrays), default=0)
    ncols = tokens.index(measure._LAST_NAME) + 1 if measure._LAST_NAME in tokens else 0
    if ncols == len(arrays) + 1:                # the file's columns are WaveView's and alter#
        nvalues = len(tokens) - ncols
        nrows = nvalues // ncols
        for name, a in zip(wv_names, arrays):
            if a.size < nrows:
                raise WaveViewError(f"{path}: WaveView reads {a.size} of the {nrows} values of {name}: WaveView "
                                    f"stops early on some values of about 1e34 or more, so the engine cannot read "
                                    f"this file (read it with engine='python')")
        truncated = bool(nvalues % ncols)
    else:
        truncated = any(a.size != nrows for a in arrays)
    if truncated:
        warnings.warn(f"{path}: file ends with an incomplete row; keeping {nrows} complete rows", RuntimeWarning)
    all_names = measure._unique(path, wv_names)
    columns = {name: a[:nrows] for name, a in zip(all_names, arrays)}
    index, params = measure._leading_columns(all_names, param_count)
    kept = measure._select(all_names, index + params, names)
    values = {name: columns[name] for name in kept}
    failed = {name: int(np.isnan(v).sum()) for name, v in values.items() if np.isnan(v).any()}
    return measure.MeasureSet(path=path, title=title, params=params, names=kept, values=values, failed=failed,
                              nrows=nrows, truncated=truncated)
