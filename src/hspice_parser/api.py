"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .reader import TraceSet, read_header, read_traces

OUTPUTS = ("arrays", "csv", "npz", "summary")


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
        "sweeps": len(ts.sweep_values), "truncated": ts.truncated, "traces": traces,
    }


def extract(path, names=None, sweeps=None, output: str = "arrays",
            downsample: Optional[int] = None, dest=None):
    """Read selected traces and return them as arrays, a summary dict, or a written file path."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    ts = read_traces(path, names, sweeps)
    if output == "summary":
        return summarize(ts, downsample)
    if downsample is not None:
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    raise NotImplementedError("file outputs arrive in the next task")
