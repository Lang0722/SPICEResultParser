"""MCP server exposing spice_result_parser to agents over stdio.

Install the extra first: pip install 'spice_result_parser[mcp]'. Then register the
command `srp-mcp` with your MCP client.

The module imports without the mcp package: `format_text`, the agent-readable
rendering of a trace selection, and the tool functions stay usable as plain
Python; only `main()` needs the server.
"""
import math
import struct
from typing import Annotated, List, Optional

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

from . import api, measure

server = FastMCP("spice-result-parser") if FastMCP is not None else None


def _tool(fn):
    """Register fn as a server tool; without the mcp package, leave it a plain function."""
    return server.tool()(fn) if server is not None else fn


_REPORTED = (ValueError, OSError, ImportError)   # errors whose text the client should see


def _tool_errors(fn):
    """The mcp server forwards only ToolError text to the client; anything else becomes a bare
    'Error executing tool'. Re-raise the errors we mean the agent to read."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except _REPORTED as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:                 # unexpected: still better than a bare "Error executing tool"
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__wrapped__ = fn
    return wrapper


# -- agent-readable text -----------------------------------------------------------

TEXT_MAX_ROWS = 500     # rows one text result may hold, over all tables
MAX_RUNS = 32           # split a sweep on x restarts only up to this many runs

_ANALYSIS_WORDS = {"tr": "transient", "sw": "dc sweep", "ac": "ac", "op": "operating point",
                   "noise": "noise"}
TRUNCATED_LINE = "TRUNCATED: the file ended before its last point; the simulation did not finish"
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


def format_text(ts, points: int = 20) -> str:
    """Agent-readable text for a TraceSet: a header, per-trace min/max, then one row
    per sample with x once and one column per trace.

    Numbers are rounded to 6 significant digits; NaN and infinity print as nan/inf.
    `points` rows are shown per table (evenly spaced, first and last kept), at most
    TEXT_MAX_ROWS in total; the header says how many points exist and, when rows
    were dropped, how many are shown. Each sweep gets its own table, and a sweep
    whose x restarts (a nested .dc written flat) one table per run.
    """
    h = ts.header
    x_name, traces = h.x_name, ts.selected[1:]
    nsweeps = len(ts.sweep_values)
    total = sum(int(ts.data[x_name][i].size) for i in range(nsweeps))

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
    per_table = max(2, min(points, TEXT_MAX_ROWS // max(1, len(tables))))

    body: List[str] = []
    shown = 0
    for label, sl, i in tables:
        x = ts.data[x_name][i][sl]
        idx = api.decimation_indices(x.size, per_table)
        cols = [x] + [ts.data[name][i][sl] for name in traces]
        if idx is not None:
            cols = [c[idx] for c in cols]
        shown += int(cols[0].size)
        rows = [[x_name, *traces]] + [[_num(v) for v in row] for row in zip(*cols)]
        if label:
            body.append(f"-- {label} --")
        body.extend(_table(rows))

    head = f"{_ANALYSIS_WORDS.get(h.analysis, h.analysis)} | x: {x_name} | {total} points"
    if nsweeps > 1:
        head += f" in {nsweeps} sweeps"
    if shown < total:
        head += f" ({shown} shown)"
    if len(h.plot_names) > 1:
        head += f" | plot {h.plot}: {h.plot_names[h.plot]} (file has {len(h.plot_names)} plots)"
    lines = [TRUNCATED_LINE] if ts.truncated else []
    lines.append(head)
    if h.analysis == "ac":
        lines.append(AC_LEGEND[h.ac_format])
    if total == 0:
        lines.append("no points in the selected x window")
        return "\n".join(lines)
    if traces:
        stats = [["trace", "min", "max"]]
        for i in range(nsweeps):
            tag = f"[{ts.sweep_indices[i]}]" if nsweeps > 1 else ""
            for name in traces:
                a = np.asarray(ts.data[name][i], dtype=np.float64)
                if a.size:
                    lo, hi, _ = api.nan_stats(a)
                    stats.append([name + tag, _num(lo), _num(hi)])
                else:
                    stats.append([name + tag, "-", "-"])
        lines += ["", *_table(stats)]
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
    lines = [head]
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


@_tool
@_tool_errors
def list_traces(
    path: Annotated[str, Field(description="Result file: HSPICE .tr#/.sw#/.ac# (binary 9601/2001 "
                                           "or ASCII) or a Nutmeg .raw (binary or ASCII).")],
    plot: Annotated[int, Field(description="Nutmeg only: which plot of the rawfile to describe, "
                                           "0-based (see \"plots\"). Must be 0 for HSPICE files.")] = 0,
) -> dict:
    """List the traces in a simulation result file. Call this first: it reads only the
    header, so it is cheap on any file size, and gives the names to pass to extract.

    Result fields:
    - path, format ("9601", "2001", "ascii" or "nutmeg")
    - analysis: "tr" transient, "sw" dc sweep, "ac"; Nutmeg adds "op", "noise", "other"
    - x: the x variable (TIME, HERTZ, the swept source), always returned by extract
    - traces: every other trace name
    - sweep_params: names of the swept parameters (HSPICE; [] for Nutmeg)
    - sweep_count_hint: number of sweeps the header announces; 0 means a single sweep
    - plot, plots: the plot described and the names of all plots (Nutmeg; HSPICE: 0, [])

    AC traces are listed as <name>_Mag and <name>_Phase. extract's ac_format renames
    them; select them there by raw name (v(out)) or glob (v_out*), which work in every
    format. For a measure file (.mt#, .ms#, .ma#) use read_measures instead.
    """
    return api.list_traces(path, plot)


@_tool
@_tool_errors
def extract(
    path: Annotated[str, Field(description="Result file, as for list_traces.")],
    names: Annotated[Optional[List[str]], Field(
        description="Traces to read: names from list_traces (v_out), raw names (v(out)) or "
                    "globs (v_*). None reads every trace. x is always included. For AC, "
                    "v(out) or v_out* selects both columns of the pair. An unknown name "
                    "fails with a list of the available traces.")] = None,
    sweeps: Annotated[Optional[List[int]], Field(
        description="0-based sweep indices to keep; None keeps all. Each sweep's parameter "
                    "values head its table in the text output.")] = None,
    output: Annotated[str, Field(
        description="\"text\" (default): readable report, see the tool description. "
                    "\"summary\": JSON with exact numbers. \"csv\" / \"npz\": write every "
                    "point to a file. \"png\": write a plot image.")] = "text",
    downsample: Annotated[Optional[int], Field(
        description="Evenly spaced points kept per sweep, first and last included; at least "
                    "2. Default: text 20 rows (500 rows in total at most), summary 200 points "
                    "(20000 in total at most), csv/npz every point. png draws at most 5000 "
                    "points per line.")] = None,
    dest: Annotated[Optional[str], Field(
        description="Output path for csv/npz/png; replaces an existing file. Default: "
                    "<input>_<ext>_traces.<output> beside the input.")] = None,
    xrange: Annotated[Optional[List[float]], Field(
        description="[min, max] on x: keep only rows inside the closed interval, for every "
                    "output. Applied while reading, so a narrow window on a huge file is "
                    "cheap: prefer it to reading everything.")] = None,
    yrange: Annotated[Optional[List[float]], Field(
        description="[min, max] vertical axis limits of the png plot; ignored otherwise.")] = None,
    plot: Annotated[int, Field(description="Nutmeg only: which plot to read, 0-based (see "
                                           "list_traces). Must be 0 for HSPICE files.")] = 0,
    ac_format: Annotated[str, Field(
        description="What each complex (AC) trace becomes: \"magphase\" (default; _Mag linear, "
                    "_Phase degrees), \"db\" (_dB = 20*log10 magnitude, _Phase) or "
                    "\"realimag\" (_Re, _Im). Ignored without complex data.")] = "magphase",
):
    """Read selected traces from an HSPICE result file or a Nutmeg rawfile. Call
    list_traces first for the names. Start with output="text" to look at the data, then
    narrow with xrange to zoom in; use csv or npz to hand every point to other tools.

    text: a header (analysis, x, point count, "(n shown)" when rows were dropped;
    "TRUNCATED" first when the simulation had not finished writing), a min/max table
    over every point, then one table per sweep with x once and one column per trace.
    Values have 6 significant digits; nan and inf are spelled out.

    summary: JSON with exact floats: per trace and sweep count, min, max, mean (NaN
    skipped), first, last, plus downsampled x/y points; NaN and inf become null.

    csv, npz, png: the file is written, and the result holds output, path, traces,
    sweeps, sweep_indices, points (per sweep), truncated, and for png size [w, h].
    """
    if output == "arrays":
        raise ValueError("output='arrays' is not available over MCP; use 'summary', 'csv', 'npz' or 'png'")
    if output not in ("text", "summary", "csv", "npz", "png"):
        raise ValueError(f"output must be one of ('text', 'summary', 'csv', 'npz', 'png'), not {output!r}")
    xrange = api.validate_range("xrange", xrange)
    yrange = api.validate_range("yrange", yrange)
    if output == "text":
        ts = api.extract(path, names, sweeps, "arrays", xrange=xrange, plot=plot, ac_format=ac_format)
        return format_text(ts, 20 if downsample is None else downsample)
    if output == "summary":
        return api.extract(path, names, sweeps, "summary", 200 if downsample is None else downsample,
                           xrange=xrange, plot=plot, ac_format=ac_format)
    ts = api.extract(path, names, sweeps, "arrays", downsample, xrange=xrange, plot=plot,
                     ac_format=ac_format)
    written = api.write_file(ts, output, dest, yrange)
    x = ts.header.x_name
    result = {
        "output": output, "path": written, "traces": ts.selected, "sweeps": len(ts.sweep_values),
        "sweep_indices": list(ts.sweep_indices),
        "points": [int(ts.data[x][i].size) for i in range(len(ts.sweep_values))],
        "truncated": ts.truncated,
    }
    if output == "png":
        result["size"] = list(_png_size(written))
    return result


@_tool
@_tool_errors
def read_measures(
    path: Annotated[str, Field(description="HSPICE measure file: .mt# (transient), .ms# (dc) "
                                           "or .ma# (ac).")],
    names: Annotated[Optional[List[str]], Field(
        description="Measures to show, by name or glob (tp*); None shows every column. The "
                    "index column and the swept parameters are always kept.")] = None,
    rows: Annotated[int, Field(
        description="Rows shown, evenly spaced with first and last kept; at most 500.")] = 20,
) -> str:
    """Read the results of the netlist's .measure statements: one row per simulation
    (per sweep point, Monte Carlo sample or temperature), one column per measure.

    Returns text: a header with the row count and swept parameters, a "failed:" line
    naming each measure that could not be evaluated and in how many rows, a min/max/mean
    table when rows were dropped, then the table itself. Failed values print as
    "failed"; numbers have 6 significant digits.
    """
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
