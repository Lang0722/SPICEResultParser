"""MCP server exposing spice_result_parser to agents over stdio.

Install the extra first: pip install 'spice_result_parser[mcp]'. Then register the
command `srp-mcp` with your MCP client.

The module imports without the mcp package: `format_text`, the agent-readable
rendering of a trace selection, and the tool functions stay usable as plain
Python; only `main()` needs the server.
"""
import math
import os
import struct
from typing import Annotated, List, Literal, Optional, Union

import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        # mcp 2.x renamed FastMCP to MCPServer.
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        FastMCP = None

try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:
        ToolError = ValueError

try:
    from pydantic import Field
except ImportError:                          # no mcp, so no schema to describe
    def Field(description=None):
        return description

try:
    from mcp.types import ToolAnnotations
except ImportError:
    ToolAnnotations = None

from . import api, measure, reduce
from .reader import AC_FORMATS

server = FastMCP("spice-result-parser") if FastMCP is not None else None


def _tool(read_only: bool):
    """Register a function as a server tool, hinting whether it writes files; without the
    mcp package, leave it a plain function."""
    def register(fn):
        if server is None:
            return fn
        if ToolAnnotations is None:          # mcp before 1.7 has no tool annotations
            return server.tool()(fn)
        hints = ToolAnnotations(readOnlyHint=read_only, destructiveHint=not read_only, openWorldHint=False)
        return server.tool(annotations=hints)(fn)
    return register


_REPORTED = (ValueError, OSError, ImportError)   # errors whose text the client should see


def _error_text(exc: Exception) -> str:
    """The error as the agent should read it. A missing file named by a relative path
    says where the server looked: its working directory need not be the agent's."""
    text = str(exc)
    name = getattr(exc, "filename", None)
    if isinstance(exc, FileNotFoundError) and isinstance(name, str) and not os.path.isabs(name):
        text += f" (relative paths resolve against the server's working directory, {os.getcwd()})"
    return text


def _tool_errors(fn):
    """The mcp server forwards only ToolError text to the client; anything else becomes a bare
    'Error executing tool'. Re-raise the errors we mean the agent to read."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except _REPORTED as exc:
            raise ToolError(_error_text(exc)) from exc
        except Exception as exc:                 # unexpected: still better than a bare "Error executing tool"
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__wrapped__ = fn
    return wrapper


# -- agent-readable text -----------------------------------------------------------

TEXT_MAX_ROWS = 500     # rows one text result may hold, over all tables
TEXT_MAX_TABLES = 20    # tables one text result may hold: the first and the last TEXT_MAX_TABLES // 2
TEXT_MAX_STATS = 100    # min/max rows per (sweep, trace); above it, one row per trace over all
                        # sweeps, and at most this many traces
TEXT_MAX_TRACES = 16    # trace columns in the point tables; the min/max table lists more
TEXT_MAX_CELLS = 2000   # numbers the point tables may hold, x included: bounds wide tables
TEXT_ENVELOPE_SLICES = 100  # slices of an envelope table, when downsample does not set them
TEXT_MAX_CONSTANTS = 20 # traces named on the "constant:" line
MAX_RUNS = 32           # split a sweep on x restarts only up to this many runs
MCP_MAX_RANGES = 100    # traces whose [min, max] one JSON result lists; names= narrows to others

_ANALYSIS_WORDS = {"tr": "transient", "sw": "dc sweep", "ac": "ac", "op": "operating point",
                   "noise": "noise"}
TRUNCATED_LINE = "TRUNCATED: the file ended before its last point; the simulation did not finish"
MEASURE_TRUNCATED_LINE = "TRUNCATED: the file ends part-way through a row; only the complete rows are shown"
AC_LEGEND = {   # per ac_format
    "magphase": "complex traces are split into _Mag (linear magnitude) and _Phase (degrees)",
    "db": "complex traces are split into _dB (20*log10 of the magnitude) and _Phase (degrees)",
    "realimag": "complex traces are split into _Re (real part) and _Im (imaginary part)",
}


def _num(v) -> str:
    """6 significant digits; nan/inf spelled out; no negative zero."""
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v + 0.0:.6g}"


def _table(rows: List[List[str]]) -> List[str]:
    """Right-aligned columns separated by two spaces; rows[0] is the heading."""
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return ["  ".join(cell.rjust(w) for cell, w in zip(r, widths)) for r in rows]


def _runs(x) -> List[slice]:
    """Slices of x between restarts (x decreasing), as a nested .dc sweep that ngspice
    writes flat produces; one slice when x never restarts or restarts implausibly often."""
    starts = np.flatnonzero(np.diff(x) < 0) + 1
    if starts.size == 0 or starts.size + 1 > MAX_RUNS:
        return [slice(0, x.size)]
    bounds = [0, *starts.tolist(), x.size]
    return [slice(a, b) for a, b in zip(bounds, bounds[1:])]


def _min_max(arrays) -> List[str]:
    """min and max, NaN skipped, over the points of every array; "-" when there are none."""
    if not any(len(a) for a in arrays):
        return ["-", "-"]
    return [_num(v) for v in api.range_of(arrays)]


def _interval(bounds) -> str:
    """[lo, hi] as text, in the form xrange takes."""
    return f"[{_num(bounds[0])}, {_num(bounds[1])}]"


def _plot_extent(ts) -> str:
    """Where the whole plot's data lies, for an agent whose x window missed or clipped it."""
    if not ts.plot_points:
        return "the plot holds no points"
    return f"the whole plot has {ts.plot_points} points with {ts.header.x_name} in {_interval(ts.plot_x_range)}"


def _constant_value(arrays):
    """The value every sample of the arrays holds (NaN equal to NaN); None when they vary
    or hold no sample."""
    value = None
    for a in arrays:
        a = np.asarray(a)
        if not a.size:
            continue
        if value is None:
            value = a[0]
        same = np.isnan(a) if np.isnan(value) else a == value
        if not same.all():
            return None
    return value


def _percent(fraction: float) -> str:
    return f"{100 * fraction:.3g}%"



def format_text(ts, points: Optional[int] = None, tolerance: float = reduce.DEFAULT_TOLERANCE) -> str:
    """Agent-readable text for a TraceSet: a header, per-trace min/max, then a table per
    sweep with x once and one column per trace.

    The tables do not show every point. They show the fewest real samples such that
    straight lines between them stay within `tolerance` (a fraction of each trace's range
    over the selection) of every sample: reduce.breakpoints, shared by the traces of a
    table. On AC data the rows are chosen on a log-frequency axis, with _Mag compared in
    dB. The header gives the x range, the point count and, when rows were dropped, how
    many are shown and the error bound they meet. A table holds at most `points` rows
    (None: no limit but the budget); all of them hold at most TEXT_MAX_ROWS rows and
    TEXT_MAX_CELLS numbers. A table whose rows cannot meet the tolerance is too dense to
    follow (an oscillation, a switching node): it becomes an envelope instead, one row
    per slice of x with every trace's lowest and highest sample (reduce.peak_detect;
    TEXT_ENVELOPE_SLICES slices, or `points`), and the header says so. A trace that
    holds one value is named on a "constant:" line instead.

    Numbers are rounded to 6 significant digits; NaN and infinity print as nan/inf, and
    a change into or out of them is always shown. When an x window was applied, a line
    gives the whole plot's point count and x range. Each sweep gets its own table, and a
    sweep whose x restarts (a nested .dc written flat) one table per run. Of more than
    TEXT_MAX_TABLES tables only the first and the last TEXT_MAX_TABLES // 2 are shown, and
    the min/max table holds one row per trace over all sweeps when per-sweep rows would
    exceed TEXT_MAX_STATS. The point tables hold at most TEXT_MAX_TRACES traces and the
    min/max table at most TEXT_MAX_STATS; a line counts the traces left out.
    """
    h = ts.header
    x_name, traces = h.x_name, ts.selected[1:]
    nsweeps = len(ts.sweep_values)
    total = sum(int(ts.data[x_name][i].size) for i in range(nsweeps))
    constant = {}
    for name in traces:
        value = _constant_value(ts.data[name])
        if value is not None:
            constant[name] = value
    varying = [name for name in traces if name not in constant]
    columns = varying[:TEXT_MAX_TRACES]

    # (label, slice, sweep index) per table
    tables = []
    for i in range(nsweeps):
        label = ""
        if nsweeps > 1:
            params = ", ".join(f"{p}={_num(v)}" for p, v in zip(h.sweep_params, ts.sweep_values[i]))
            label = f"sweep {ts.sweep_indices[i]}" + (f": {params}" if params else "")
        runs = _runs(ts.data[x_name][i])
        for j, sl in enumerate(runs):
            run = f"run {j + 1} of {len(runs)} (x restarts)" if len(runs) > 1 else ""
            tables.append((", ".join(t for t in (label, run) if t), sl, i))
    hidden = 0
    if len(tables) > TEXT_MAX_TABLES:
        half = TEXT_MAX_TABLES // 2
        hidden = len(tables) - 2 * half
        tables = tables[:half] + [None] + tables[-half:]
    shown_tables = max(1, len(tables) - bool(hidden))
    budget = min(TEXT_MAX_ROWS, TEXT_MAX_CELLS // (1 + len(columns)))
    per_table = max(2, budget // shown_tables)
    if points is not None:
        per_table = max(2, min(per_table, points))
    # an envelope row holds a slice's two bounds and every trace's min and max
    slices = max(2, min(per_table, TEXT_ENVELOPE_SLICES if points is None else points,
                        TEXT_MAX_CELLS // ((2 + 2 * len(columns)) * shown_tables)))

    log = reduce.log_frequency(h)
    db = {name: reduce.in_db(h, name) for name in columns}
    spans = {name: reduce.finite_span(reduce.y_criterion(a, db[name]) for a in ts.data[name])
             for name in columns}
    floor = 4 * float(np.finfo(h.dtype).eps)
    body: List[str] = []
    shown, followed, enveloped = 0, 0, 0
    for table in tables if columns else []:
        if table is None:
            tip = "pass sweeps=[...] to see them" if nsweeps > 1 else "output='csv' writes every point"
            body.append(f"-- {hidden} tables not shown; {tip} --")
            continue
        label, sl, i = table
        x = ts.data[x_name][i][sl]
        values = [ts.data[name][i][sl] for name in columns]
        idx, error = reduce.breakpoints(
            reduce.x_criterion(x, log), [reduce.y_criterion(v, db[name]) for v, name in zip(values, columns)],
            [spans[name] for name in columns], tolerance, per_table, floor)
        if error <= max(tolerance, floor):
            followed += 1
            cols = [x[idx]] + [v[idx] for v in values]
            rows = [[x_name, *columns]] + [[_num(v) for v in row] for row in zip(*cols)]
        else:                                       # too dense to follow: show it as a peak-detect envelope
            enveloped += 1
            start, stop, mins, maxs = reduce.peak_detect(x, values, slices, log)
            cols = [start, stop] + [c for pair in zip(mins, maxs) for c in pair]
            heading = [f"{x_name}_from", f"{x_name}_to"] + [f"{k}({name})" for name in columns
                                                            for k in ("min", "max")]
            rows = [heading] + [[_num(v) for v in row] for row in zip(*cols)]
            label = (f"{label}; " if label else "") + "envelope: each row is a slice with every trace's min and max"
            if np.any(start[1:] != stop[:-1]):     # the simulator stepped over some slices
                label += "; slices without a sample are left out"
        shown += len(rows) - 1
        if label:
            body.append(f"-- {label} --")
        body.extend(_table(rows))

    head = f"{_ANALYSIS_WORDS.get(h.analysis, h.analysis)} | x: {x_name}"
    if total:
        head += f" in {_interval(api.range_of(ts.data[x_name]))}"
    head += f" | {total} points"
    if nsweeps > 1:
        head += f" in {nsweeps} sweeps"
    if body:
        each = " per table" if shown_tables > 1 else ""
        fidelity = (f"each trace within {_percent(tolerance)} of its range"
                    + ("; _Mag compared in dB" if any(db.values()) else ""))
        dense = f"too dense to follow every trace within {_percent(tolerance)} in {per_table} rows{each}"
        if not enveloped and shown < total:
            head += f" ({shown} shown, {fidelity})"
        elif enveloped and not followed:
            head += f" (an envelope of {slices} x slices{each}: {dense}; narrow xrange to see the waveform)"
        elif enveloped:
            head += f" ({shown} rows shown, {fidelity}; {enveloped} of {enveloped + followed} tables as envelopes: {dense})"
    if len(h.plot_names) > 1:
        head += f" | plot {h.plot}: {h.plot_names[h.plot]} (file has {len(h.plot_names)} plots)"
    lines = [TRUNCATED_LINE] if ts.truncated else []
    lines.append(head)
    if h.analysis == "ac":
        lines.append(AC_LEGEND[h.ac_format])
    if total == 0:
        if nsweeps == 0:
            lines.append("no sweeps in the selection")
        elif ts.xrange is not None:
            lines.append(f"no points in xrange {_interval(ts.xrange)}; {_plot_extent(ts)}")
        else:
            lines.append("no points in the selected sweeps")
        return "\n".join(lines)
    if ts.xrange is not None:
        lines.append(f"window: xrange {_interval(ts.xrange)}; {_plot_extent(ts)}")
    if constant:
        named = [f"{name} = {_num(v)}" for name, v in list(constant.items())[:TEXT_MAX_CONSTANTS]]
        more = len(constant) - len(named)
        lines.append("constant: " + ", ".join(named) + (f", and {more} more" if more else ""))
    if varying:
        listed = varying[:TEXT_MAX_STATS]
        if nsweeps <= 1 or nsweeps * len(varying) <= TEXT_MAX_STATS:
            stats = [["trace", "min", "max"]]
            for i in range(nsweeps):
                tag = f"[{ts.sweep_indices[i]}]" if nsweeps > 1 else ""
                for name in listed:
                    stats.append([name + tag, *_min_max(ts.data[name][i:i + 1])])
        else:
            stats = [[f"trace (all {nsweeps} sweeps)", "min", "max"]]
            stats += [[name, *_min_max(ts.data[name])] for name in listed]
        lines += ["", *_table(stats)]
        if len(listed) < len(varying):
            lines.append(f"-- {len(varying) - len(listed)} more traces not shown; select fewer names --")
    if len(columns) < len(varying):
        lines += ["", f"-- the tables below show the first {len(columns)} of {len(varying)} traces; "
                      f"select names to see the others --"]
    if body:
        lines += ["", *body]
    return "\n".join(lines)


def format_measures(ms, rows: int = 20) -> str:
    """Agent-readable text for a MeasureSet: a header, which measures failed, then one row
    per measurement with every kept column. Failed values print as `failed`.

    `rows` rows are shown (evenly spaced, first and last kept, at most TEXT_MAX_ROWS);
    when rows were dropped a min/max/mean table over every row comes first.
    """
    n = ms.rows
    idx = api.decimation_indices(n, max(2, min(rows, TEXT_MAX_ROWS)))
    shown = n if idx is None else int(idx.size)
    head = f"measures | {n} row{'' if n == 1 else 's'}"
    if shown < n:
        head += f" ({shown} shown)"
    if ms.params:
        head += " | params: " + ", ".join(ms.params)
    lines = [MEASURE_TRUNCATED_LINE] if ms.truncated else []
    lines.append(head)
    if ms.failed:
        lines.append("failed: " + ", ".join(f"{name} in {k} of {n} rows" for name, k in ms.failed.items()))
    if idx is not None:
        stats = [["measure", "min", "max", "mean"]]
        for name in ms.names:
            if name != "index" and name not in ms.params:
                stats.append([name, *(_num(v) for v in api.nan_stats(ms.values[name]))])
        lines += ["", *_table(stats)]
    cols = [ms.values[name] if idx is None else ms.values[name][idx] for name in ms.names]
    table = [list(ms.names)] + [["failed" if math.isnan(v) else _num(v) for v in row] for row in zip(*cols)]
    lines += ["", *_table(table)]
    return "\n".join(lines)


_NAMES_HELP = ("trace names (v_out), raw SPICE names (v(out)) or globs (v_*); case is ignored "
               "when nothing matches exactly. For AC, v(out) selects both columns of the pair. An "
               "unknown name fails with near names and the available traces.")
_PATH_HELP = "Prefer an absolute path: a relative one resolves against the server's working directory."
_AC_FORMAT_HELP = ("What each complex (AC) trace becomes: \"magphase\" (default; _Mag linear, "
                   "_Phase degrees), \"db\" (_dB = 20*log10 magnitude, _Phase) or \"realimag\" "
                   "(_Re, _Im). Files without complex data do not use it.")


def _json_result(result: dict, ts=None) -> dict:
    """A JSON result as the agent gets it: at most MCP_MAX_RANGES trace ranges, and a
    "note" when ranges were cut or the x window of ts kept no point."""
    notes = []
    if ts is not None and ts.xrange is not None and ts.sweep_values and not result["points"]:
        notes.append(f"no points in xrange {_interval(ts.xrange)}; {_plot_extent(ts)}")
    ranges = result.get("ranges")
    if ranges is not None and len(ranges) > MCP_MAX_RANGES:
        result["ranges"] = dict(list(ranges.items())[:MCP_MAX_RANGES])
        notes.append(f"ranges lists the first {MCP_MAX_RANGES} of {len(ranges)} traces; "
                     f"pass names=[...] for the others")
    if notes:
        result["note"] = "; ".join(notes)
    return result


@_tool(read_only=True)
@_tool_errors
def list_traces(
    path: Annotated[str, Field(description="Result file: HSPICE .tr#/.sw#/.ac# (binary 9601/2001 "
                                           "or ASCII) or a Nutmeg .raw (binary or ASCII). " + _PATH_HELP)],
    names: Annotated[Optional[Union[str, List[str]]], Field(
        description="Traces to list, as extract selects them: " + _NAMES_HELP + " None lists every "
                    "trace.")] = None,
    plot: Annotated[int, Field(description="Nutmeg only: which plot of the rawfile to describe, "
                                           "0-based (see \"plots\"). Must be 0 for HSPICE files.")] = 0,
    ac_format: Annotated[Literal[tuple(AC_FORMATS)], Field(
        description=_AC_FORMAT_HELP + " Names and ranges follow it, as extract returns them.")] = "magphase",
    header_only: Annotated[bool, Field(
        description="True reads only the header: instant on any file size, but without "
                    "x_range, points, sweep_values, truncated and ranges, and sweeps is the "
                    "count the header announces. Default False reads the data once.")] = False,
) -> dict:
    """List what a simulation result file holds, with the extents to set extract's
    parameters from. Call this first. It reads the whole file once and keeps none of it.

    Result fields:
    - path, format ("9601", "2001", "ascii" or "nutmeg"); analysis: "tr" transient, "sw"
      dc sweep, "ac"; Nutmeg adds "op", "noise", "other"
    - plot, plots: the plot described and the names of all plots (Nutmeg; HSPICE: 0, [])
    - x: the x variable (TIME, HERTZ, the swept source); x_range: its [min, max], for xrange
    - points: total points; sweeps: their count (extract's sweeps takes 0 to sweeps-1);
      sweep_params, sweep_values: each sweep's parameter values (HSPICE; Nutmeg: none)
    - truncated: the simulation had not finished writing the file
    - traces: every other trace name; ranges: each trace's [min, max], NaN skipped
      (null: no value, or infinite). At most 100 ranges: "note" then says to narrow names.

    AC traces are listed as <name>_Mag and <name>_Phase; v(out) selects the pair in
    extract. For a measure file (.mt#, .ms#, .ma#) use read_measures instead.
    """
    return _json_result(api.list_traces(path, plot, ac_format, names, header_only=header_only))


MCP_OUTPUTS = ("text", "summary", "csv", "npz", "png")


@_tool(read_only=False)
@_tool_errors
def extract(
    path: Annotated[str, Field(description="Result file, as for list_traces.")],
    names: Annotated[Optional[Union[str, List[str]]], Field(
        description="Traces to read, as list_traces lists them: " + _NAMES_HELP + " None reads "
                    "every trace. x is always included.")] = None,
    sweeps: Annotated[Optional[Union[int, List[int]]], Field(
        description="0-based sweep indices to keep (list_traces gives sweeps and each one's "
                    "sweep_values); None keeps all. An index the file does not hold is an "
                    "error. On files with many sweeps, pick a few here.")] = None,
    output: Annotated[Literal[MCP_OUTPUTS], Field(
        description="\"text\" (default): readable report, see the tool description. "
                    "\"summary\": JSON with exact numbers. \"csv\" / \"npz\": write every "
                    "point to a file. \"png\": write a plot image.")] = "text",
    downsample: Annotated[Optional[int], Field(
        description="csv/npz: keep this many evenly spaced points per sweep, first and last "
                    "included (default: every point). text: at most this many rows per table. "
                    "summary: list at most this many x/y points per trace and sweep (default: "
                    "none, statistics only). At least 2. png always draws every point's "
                    "envelope.")] = None,
    tolerance: Annotated[Optional[float], Field(
        description="text and summary: how far straight lines between the rows shown may "
                    "stray from any point, as a fraction of each trace's range in the "
                    "selection (default 0.01 = 1%; 0 drops only points on a straight line). "
                    "Rows are real samples, each trace's min and max among them. A signal too "
                    "dense for that in the rows (oscillation, switching) shows as an envelope "
                    "of x slices instead; narrow xrange to see its waveform. The range is the "
                    "selection's, so an xrange on a small-signal part (settling, ripple) shows "
                    "it in detail; lower tolerance for more rows anywhere. AC: chosen on log "
                    "frequency, _Mag compared in dB. An error with csv, npz and png.")] = None,
    dest: Annotated[Optional[str], Field(
        description="Output path for csv/npz/png (an absolute path is best); replaces an "
                    "existing file, but never the input. Default: <input>_<ext>_traces.<output> "
                    "beside the input. An error with text or summary.")] = None,
    xrange: Annotated[Optional[List[float]], Field(
        description="[min, max] on x (list_traces gives x_range): keep only rows inside the "
                    "closed interval, for every output. Applied while reading, so a narrow "
                    "window on a huge file is cheap: prefer it to reading everything.")] = None,
    yrange: Annotated[Optional[List[float]], Field(
        description="[min, max] vertical axis limits of the png plot (list_traces gives each "
                    "trace's range); other outputs do not use it, but a malformed value is "
                    "still an error.")] = None,
    plot: Annotated[int, Field(description="Nutmeg only: which plot to read, 0-based (see "
                                           "list_traces). Must be 0 for HSPICE files.")] = 0,
    ac_format: Annotated[Literal[tuple(AC_FORMATS)], Field(description=_AC_FORMAT_HELP)] = "magphase",
):
    """Read selected traces from an HSPICE result file or a Nutmeg rawfile. Call
    list_traces first: its names, x_range, sweeps and ranges are what names, xrange,
    sweeps and yrange take. Start with text, zoom with xrange; csv or npz hand every
    point to other tools.

    text: "TRUNCATED" first when the simulation had not finished writing; a header
    (analysis, x range, point count, rows shown and their error bound); with xrange, the
    whole plot's extent; constant traces on one line; a min/max table; then a table per
    sweep with x once and a column per trace (at most 16). Rows are real samples: the
    fewest whose straight lines stay within tolerance of every point (peaks, edges,
    glitches stay). Too dense for that (oscillation, switching): an envelope, a row
    per x slice with each trace's min and max. 6 significant digits.

    The JSON outputs hold list_traces' fields for the data read: x, x_range, points,
    sweeps, truncated, traces, ranges (at most 100; then a "note").
    summary: also sweep_values, sweep_indices and stats: per trace and sweep count,
    min, max, mean (NaN skipped), first, last; with downsample or tolerance, x/y points
    chosen the same way and their max_error, or an envelope. NaN and inf become null.
    csv, npz, png: the file is written; the result adds output, dest (absolute),
    sweep_indices when sweeps was given, and for png size [w, h]. For AC, plot the
    magnitude and the phase separately.
    """
    if output == "arrays":
        raise ValueError(f"output='arrays' is not available over MCP; use one of {MCP_OUTPUTS}")
    if output not in MCP_OUTPUTS:
        raise ValueError(f"output must be one of {MCP_OUTPUTS}, not {output!r}")
    downsample = api.check_downsample(downsample)
    tolerance = api.check_tolerance(tolerance)
    api.reject_tolerance(output, tolerance)
    xrange = api.validate_range("xrange", xrange)
    yrange = api.validate_range("yrange", yrange, finite=True)
    if output in ("text", "summary"):
        if dest is not None:
            raise ValueError(f"dest applies to csv, npz and png only, not to output={output!r}")
        ts = api.extract(path, names, sweeps, "arrays", xrange=xrange, plot=plot, ac_format=ac_format)
        if output == "text":
            return format_text(ts, downsample, reduce.DEFAULT_TOLERANCE if tolerance is None else tolerance)
        return _json_result(api.summarize(ts, downsample, tolerance), ts)
    api.resolve_dest(path, output, dest)             # fail before the read, not after it
    if output == "png":
        api._require_matplotlib()
    ts = api.extract(path, names, sweeps, "arrays", downsample, xrange=xrange, plot=plot,
                     ac_format=ac_format)
    written = os.path.abspath(api.write_file(ts, output, dest, yrange))
    facts = api.overview(ts)
    result = {"output": output, "dest": written}
    for key in ("x", "x_range", "points", "sweeps", "sweep_indices", "truncated", "traces", "ranges"):
        if key != "sweep_indices" or sweeps is not None:
            result[key] = facts[key]
    if output == "png":
        result["size"] = list(_png_size(written))
    return _json_result(result, ts)


@_tool(read_only=True)
@_tool_errors
def read_measures(
    path: Annotated[str, Field(description="HSPICE measure file: .mt# (transient), .ms# (dc) "
                                           "or .ma# (ac). " + _PATH_HELP)],
    names: Annotated[Optional[Union[str, List[str]]], Field(
        description="Measures to show, by name or glob (tp*); case is ignored when nothing "
                    "matches exactly. None shows every column. The index column and the swept "
                    "parameters are always kept. An unknown name fails with near names and the "
                    "available measures.")] = None,
    rows: Annotated[int, Field(
        description="Rows shown, evenly spaced with first and last kept; at least 2, at "
                    "most 500.")] = 20,
):
    """Read the results of the netlist's .measure statements: one row per simulation
    (per sweep point, Monte Carlo sample or temperature), one column per measure.

    Returns text: "TRUNCATED" first when the file ends part-way through a row, a header
    with the row count and swept parameters, a "failed:" line naming each measure that
    could not be evaluated and in how many rows, a min/max/mean table when rows were
    dropped, then the table itself. Failed values print as "failed"; numbers have 6
    significant digits.
    """
    if rows < 2:
        raise ValueError("rows must be at least 2")
    return format_measures(measure.read_measures(path, names), rows)


def _png_size(path: str):
    """(width, height) from the PNG IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def main() -> None:
    if server is None:
        raise ImportError("MCP support needs the optional dependency: pip install 'spice_result_parser[mcp]'")
    server.run()   # stdio transport


if __name__ == "__main__":
    main()
