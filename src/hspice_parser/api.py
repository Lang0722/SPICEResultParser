"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

import csv
import math
import os
import zipfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .reader import TraceSet, read_header, read_traces

OUTPUTS = ("arrays", "csv", "npz", "summary", "png")


def list_traces(path, plot: int = 0) -> dict:
    """Trace names and file facts from the header only.

    plot selects one plot of a Nutmeg rawfile; "plots" lists every plot in it (HSPICE: []).
    """
    h = read_header(path, plot)
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "x": h.x_name,
        "traces": h.names[1:], "sweep_params": h.sweep_params, "sweep_count_hint": h.sweep_count_hint,
        "plot": h.plot, "plots": h.plot_names,
    }


def decimation_indices(n: int, downsample: Optional[int]):
    """Indices of `downsample` evenly spaced points out of n, first and last kept. None = keep all."""
    if downsample is None or downsample >= n:
        return None
    if downsample < 2:
        raise ValueError("downsample must be at least 2")
    return np.unique(np.linspace(0, n - 1, downsample).round().astype(np.intp))


def validate_range(name: str, value) -> Optional[Tuple[float, float]]:
    """Normalise an (lo, hi) pair to floats; None passes through."""
    if value is None:
        return None
    try:
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair of numbers (min, max), not {value!r}") from None
    if lo > hi:
        raise ValueError(f"{name} min must not exceed max: {value!r}")
    return lo, hi


def _decimate(ts: TraceSet, downsample: int) -> None:
    for i in range(len(ts.sweep_values)):
        idx = decimation_indices(ts.data[ts.header.x_name][i].size, downsample)
        if idx is None:
            continue
        for name in ts.selected:
            ts.data[name][i] = ts.data[name][i][idx]


SUMMARY_MAX_POINTS = 20000      # cap on x/y points serialised in one summary, over all traces and sweeps


def _json_float(v) -> Optional[float]:
    """float, or None for NaN and infinities (strict JSON has no token for them)."""
    v = float(v)
    return v if math.isfinite(v) else None


def _json_floats(arr) -> List[Optional[float]]:
    return [_json_float(v) for v in np.asarray(arr, dtype=np.float64).tolist()]


def summarize(ts: TraceSet, downsample: Optional[int] = None) -> dict:
    """JSON-serialisable statistics per trace and sweep; stats use the full trace, points are decimated.

    Points are emitted only when downsample is given, and never more than SUMMARY_MAX_POINTS
    in total: the budget is shared evenly by the (trace, sweep) series (a summary is for
    looking at, not for moving data). NaN and infinities are emitted as null.
    """
    h = ts.header
    x = ts.data[h.x_name]
    if downsample is not None:
        series = max(1, (len(ts.selected) - 1) * len(ts.sweep_values))
        downsample = min(downsample, max(2, SUMMARY_MAX_POINTS // series))
    traces = {}
    for name in ts.selected[1:]:
        per_sweep = []
        for i, arr in enumerate(ts.data[name]):
            a = np.asarray(arr, dtype=np.float64)
            entry = {"count": int(a.size)}
            if a.size:
                entry.update(min=_json_float(a.min()), max=_json_float(a.max()), mean=_json_float(a.mean()),
                             first=_json_float(a[0]), last=_json_float(a[-1]))
                if downsample is not None:
                    idx = decimation_indices(a.size, downsample)
                    xs, ys = (x[i], a) if idx is None else (x[i][idx], a[idx])
                    entry["x"] = _json_floats(xs)
                    entry["y"] = _json_floats(ys)
            per_sweep.append(entry)
        traces[name] = per_sweep
    return {
        "path": h.path, "format": h.version, "plot": h.plot, "plots": h.plot_names,
        "analysis": h.analysis, "x": h.x_name,
        "sweep_params": h.sweep_params, "sweep_values": [_json_floats(v) for v in ts.sweep_values],
        "sweeps": len(ts.sweep_values), "sweep_indices": list(ts.sweep_indices),
        "truncated": ts.truncated, "traces": traces,
    }


def default_dest(path, output: str) -> str:
    """<root>_<ext>_traces.<output> beside the input, e.g. run_tr0_traces.csv."""
    root, ext = os.path.splitext(os.fspath(path))
    return f"{root}_{ext.lstrip('.')}_traces.{output}"


CSV_ROWS_PER_SLICE = 65536      # rows stacked as float64 at a time; bounds writer memory


def _write_csv(ts: TraceSet, dest: str) -> None:
    h = ts.header
    with_lead = bool(h.nsweepparam) or len(ts.sweep_values) > 1
    lead_names = (["sweep"] + h.sweep_params) if with_lead else []
    with open(dest, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(lead_names + ts.selected)   # quotes names containing ","
        for i, params in enumerate(ts.sweep_values):
            n = ts.data[ts.selected[0]][i].size
            sweep_id = ts.sweep_indices[i]
            for start in range(0, n, CSV_ROWS_PER_SLICE):
                stop = min(start + CSV_ROWS_PER_SLICE, n)
                rows = stop - start
                lead = []
                if with_lead:
                    lead = ([np.full(rows, sweep_id, dtype=np.float64)]
                            + [np.full(rows, v, dtype=np.float64) for v in params])
                cols = [ts.data[name][i][start:stop].astype(np.float64) for name in ts.selected]
                np.savetxt(f, np.column_stack(lead + cols), delimiter=",", fmt="%.17g")


def _write_npz(ts: TraceSet, dest: str) -> None:
    single = len(ts.sweep_values) == 1
    arrays = {}
    for name in ts.selected:
        for i, arr in enumerate(ts.data[name]):
            arrays[name if single else f"{name}@{ts.sweep_indices[i]}"] = arr
    sweep_values = np.full((len(ts.sweep_values), ts.header.nsweepparam), np.nan)
    for i, values in enumerate(ts.sweep_values):
        sweep_values[i, :len(values)] = values
    arrays["__sweep_values__"] = sweep_values
    arrays["__sweep_params__"] = np.asarray(ts.header.sweep_params, dtype=str)
    # Written entry by entry rather than via np.savez(**arrays): trace names such as
    # "file" or "allow_pickle" would collide with savez's own parameters.
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, arr in arrays.items():
            with zf.open(key + ".npy", "w", force_zip64=True) as member:
                np.save(member, np.asarray(arr))


MAX_LEGEND_ENTRIES = 24
PLOT_MAX_POINTS = 5000          # points drawn per line; a 1200 px wide figure cannot show more


def _require_matplotlib():
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("output='png' needs the optional dependency: pip install 'hspice_parser[plot]'") from exc
    return Figure


def make_figure(ts: TraceSet, yrange: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
    """A matplotlib Figure with one line per trace per sweep; log x axis for AC results.

    Lines longer than PLOT_MAX_POINTS are decimated to evenly spaced points (first and
    last kept) before drawing, so a spike narrower than the stride can be missed. Plot a
    narrower `xrange` around it, or raise `api.PLOT_MAX_POINTS`, to see it.
    """
    Figure = _require_matplotlib()
    h = ts.header
    fig = Figure(figsize=(10, 6), dpi=120)
    ax = fig.add_subplot(111)
    multi = len(ts.sweep_values) > 1
    for name in ts.selected[1:]:
        for i, y in enumerate(ts.data[name]):
            label = name
            if multi:
                params = ", ".join(f"{p}={v:g}" for p, v in zip(h.sweep_params, ts.sweep_values[i]))
                label = f"{name} [{params or ts.sweep_indices[i]}]"
            x = ts.data[h.x_name][i]
            idx = decimation_indices(y.size, PLOT_MAX_POINTS)
            if idx is not None:
                x, y = x[idx], y[idx]
            ax.plot(x, y, linewidth=1, label=label)
    if h.analysis == "ac":
        ax.set_xscale("log")
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.set_xlabel(h.x_name)
    ax.grid(True, alpha=0.3)
    if 0 < len(ax.get_lines()) <= MAX_LEGEND_ENTRIES:
        ax.legend(fontsize="small")
    ax.set_title(title if title is not None else os.path.basename(h.path))
    return fig


def plot(ts: TraceSet, dest=None, yrange: Optional[Tuple[float, float]] = None,
         title: Optional[str] = None) -> str:
    """Write the traces as a PNG image; returns the path written. Needs matplotlib."""
    dest = os.fspath(dest) if dest is not None else default_dest(ts.header.path, "png")
    fig = make_figure(ts, validate_range("yrange", yrange), title)
    fig.savefig(dest, format="png")
    return dest


def write_file(ts: TraceSet, output: str, dest=None, yrange=None) -> str:
    """Write a TraceSet as csv, npz or png; returns the path written. yrange applies to png only."""
    if output not in ("csv", "npz", "png"):
        raise ValueError(f"write_file supports 'csv', 'npz' or 'png', not {output!r}")
    if output == "png":
        return plot(ts, dest, yrange)
    dest = os.fspath(dest) if dest is not None else default_dest(ts.header.path, output)
    if output == "csv":
        _write_csv(ts, dest)
    else:
        _write_npz(ts, dest)
    return dest


def extract(path, names=None, sweeps=None, output: str = "arrays",
            downsample: Optional[int] = None, dest=None,
            xrange: Optional[Sequence[float]] = None, yrange: Optional[Sequence[float]] = None,
            plot: int = 0):
    """Read selected traces and return them as arrays, a summary dict, or a written file path.

    xrange=(lo, hi) keeps only the rows whose x value (TIME, FREQ or the sweep
    variable) lies in the closed interval; the window is applied while the file streams,
    so peak memory follows the window rather than the column length, and it applies to
    every output. yrange=(lo, hi)
    sets the vertical axis limits of the png plot and is ignored by other outputs.
    plot selects one plot of a Nutmeg rawfile (0-based); see list_traces for the list.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    if downsample is not None and downsample < 2:
        raise ValueError("downsample must be at least 2")
    xrange = validate_range("xrange", xrange)
    yrange = validate_range("yrange", yrange)
    if output == "png":
        _require_matplotlib()
    ts = read_traces(path, names, sweeps, plot, xrange)
    if output == "summary":
        return summarize(ts, downsample)
    if downsample is not None:
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    return write_file(ts, output, dest, yrange)
