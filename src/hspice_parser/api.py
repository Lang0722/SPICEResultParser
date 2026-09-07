"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

from .reader import TraceSet, read_header, read_traces

OUTPUTS = ("arrays", "csv", "npz", "summary", "png")


def list_traces(path) -> dict:
    """Trace names and file facts from the header only."""
    h = read_header(path)
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "x": h.x_name,
        "traces": h.names[1:], "sweep_params": h.sweep_params, "sweep_count_hint": h.sweep_count_hint,
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


def _apply_xrange(ts: TraceSet, xrange: Tuple[float, float]) -> None:
    """Keep, in every sweep, only the rows whose x value lies inside the closed interval."""
    lo, hi = xrange
    x_name = ts.header.x_name
    for i in range(len(ts.sweep_values)):
        x = ts.data[x_name][i]
        mask = (x >= lo) & (x <= hi)
        if mask.all():
            continue
        for name in ts.selected:
            ts.data[name][i] = ts.data[name][i][mask]


def _decimate(ts: TraceSet, downsample: int) -> None:
    for i in range(len(ts.sweep_values)):
        idx = decimation_indices(ts.data[ts.header.x_name][i].size, downsample)
        if idx is None:
            continue
        for name in ts.selected:
            ts.data[name][i] = ts.data[name][i][idx]


def summarize(ts: TraceSet, downsample: Optional[int] = None) -> dict:
    """JSON-serialisable statistics per trace and sweep; stats use the full trace, points are decimated."""
    h = ts.header
    x = ts.data[h.x_name]
    traces = {}
    for name in ts.selected[1:]:
        per_sweep = []
        for i, arr in enumerate(ts.data[name]):
            a = arr.astype(np.float64)
            entry = {"count": int(a.size)}
            if a.size:
                entry.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                             first=float(a[0]), last=float(a[-1]))
                if downsample is not None:
                    idx = decimation_indices(a.size, downsample)
                    xs, ys = (x[i], a) if idx is None else (x[i][idx], a[idx])
                    entry["x"] = xs.astype(np.float64).tolist()
                    entry["y"] = ys.tolist()
            per_sweep.append(entry)
        traces[name] = per_sweep
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "x": h.x_name,
        "sweep_params": h.sweep_params, "sweep_values": ts.sweep_values,
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
        f.write(",".join(lead_names + ts.selected) + "\n")
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
    arrays["__sweep_values__"] = np.asarray(ts.sweep_values, dtype=np.float64).reshape(
        len(ts.sweep_values), ts.header.nsweepparam)
    arrays["__sweep_params__"] = np.asarray(ts.header.sweep_params, dtype=str)
    with open(dest, "wb") as f:            # file object: np.savez must not append ".npz"
        np.savez(f, **arrays)


MAX_LEGEND_ENTRIES = 24


def _require_matplotlib():
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("output='png' needs the optional dependency: pip install 'hspice_parser[plot]'") from exc
    return Figure


def make_figure(ts: TraceSet, yrange: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
    """A matplotlib Figure with one line per trace per sweep; log x axis for AC results."""
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
            ax.plot(ts.data[h.x_name][i], y, linewidth=1, label=label)
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
            xrange: Optional[Sequence[float]] = None, yrange: Optional[Sequence[float]] = None):
    """Read selected traces and return them as arrays, a summary dict, or a written file path.

    xrange=(lo, hi) keeps only the rows whose x value (TIME, FREQ or the sweep
    variable) lies in the closed interval; it applies to every output. yrange=(lo, hi)
    sets the vertical axis limits of the png plot and is ignored by other outputs.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    if downsample is not None and downsample < 2:
        raise ValueError("downsample must be at least 2")
    xrange = validate_range("xrange", xrange)
    yrange = validate_range("yrange", yrange)
    if output == "png":
        _require_matplotlib()
    ts = read_traces(path, names, sweeps)
    if xrange is not None:
        _apply_xrange(ts, xrange)
    if output == "summary":
        return summarize(ts, downsample)
    if downsample is not None:
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    return write_file(ts, output, dest, yrange)
