"""MCP server exposing spice_result_parser to agents over stdio.

Install the extra first: pip install 'spice_result_parser[mcp]'. Then register the
command `srp-mcp` with your MCP client.

The module imports without the mcp package: `format_text`, the agent-readable
rendering of a trace selection, and the tool functions stay usable as plain
Python; only `main()` needs the server.
"""
import math
import struct
from typing import List, Optional

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
def list_traces(path: str, plot: int = 0, ac_format: str = "magphase") -> dict:
    """List trace names in an HSPICE .tr*/.sw*/.ac* result file (binary 9601/2001 or ASCII)
    or a Nutmeg rawfile (ngspice / SPICE3 .raw, binary or ASCII).

    Reads only the header, so it is cheap on any file size. Returns the x variable
    name, the trace names to pass to `extract`, and the sweep parameter names.

    plot: which plot of a Nutmeg rawfile to describe (0-based). A rawfile can hold
          several plots; "plots" in the result names them all, and "plot" echoes the
          index described. HSPICE files hold a single plot and report "plots": [].
    ac_format: how complex (AC) variables are named: "magphase" (_Mag, _Phase),
          "db" (_dB, _Phase) or "realimag" (_Re, _Im). Pass the same value to extract.
    """
    return api.list_traces(path, plot, ac_format)


@_tool
@_tool_errors
def extract(path: str, names: Optional[List[str]] = None, sweeps: Optional[List[int]] = None,
            output: str = "text", downsample: Optional[int] = None, dest: Optional[str] = None,
            xrange: Optional[List[float]] = None, yrange: Optional[List[float]] = None,
            plot: int = 0, ac_format: str = "magphase"):
    """Extract selected traces from an HSPICE result file or a Nutmeg rawfile
    (ngspice / SPICE3 .raw, binary or ASCII) with memory bounded by the selection.

    names: trace names from list_traces (v_out), raw names (v(out)), or globs (v_*);
           None selects everything. The x variable is always included.
    sweeps: sweep indices to keep; None keeps all.
    output: "text" (default) returns a readable report: a header with the analysis, x
            variable and point count, per-trace min/max, then a table with x once and
            one column per trace, `downsample` rows per sweep (at most 500 in total),
            values rounded to 6 significant digits, nan/inf spelled out. "summary"
            returns the same as JSON with exact floats: per-trace count/min/max/mean/
            first/last plus `downsample` evenly spaced x/y points. "csv" or "npz" write
            a file and return its path; "png" writes a plot image (one line per trace
            per sweep) and returns its path and pixel size. "arrays" is not available
            over MCP.
    downsample: evenly spaced points kept per sweep, first and last always included.
            "text" defaults to 20 rows per sweep; "summary" to 200 points and never
            returns more than 20000 in total; None keeps every point for csv and npz;
            "png" draws at most 5000 points per line whatever downsample says.
    dest: output file path for csv/npz/png; default is beside the input file.
    xrange: [min, max] on the x variable (TIME, FREQ, sweep variable); only rows inside
            the closed interval are kept, for every output. The window is applied while
            the file streams, so a narrow window on a huge file is cheap in memory: use
            it rather than reading everything.
    yrange: [min, max] vertical axis limits for the png plot; ignored by other outputs.
    plot: which plot of a Nutmeg rawfile to read (0-based); see list_traces for the list.
          Ignored for HSPICE files, which hold a single plot.
    ac_format: what each complex (AC) variable becomes: "magphase" (default; _Mag linear
          magnitude, _Phase degrees), "db" (_dB = 20*log10 of the magnitude, _Phase) or
          "realimag" (_Re, _Im). Ignored for results without complex data.

    Measure files (.mt0, .ms0, .ma0) hold .measure results, not traces: use read_measures.
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
def read_measures(path: str, names: Optional[List[str]] = None, rows: int = 20) -> str:
    """Read an HSPICE measure file (.mt0 transient, .ms0 dc, .ma0 ac): the results of
    the netlist's .measure statements, one row per simulation (per sweep point, Monte
    Carlo sample or temperature).

    names: measure names or globs (tp*); None keeps every column. The index column and
           the swept parameters are always kept, so each row stays identifiable.
    rows: rows shown, evenly spaced with first and last kept (at most 500); when rows
          are dropped a min/max/mean table over all of them comes first.

    Returns a readable report: row count and swept parameters, which measures failed
    (a failed value prints as "failed"), then a table with one column per measure,
    values rounded to 6 significant digits.
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
