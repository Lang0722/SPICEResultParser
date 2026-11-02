"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

import csv
import errno
import math
import operator
import os
import warnings
import zipfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import reduce
from .reader import TraceSet, nan_min_max, read_header, read_traces, resolve_columns, scan, validate_range
from .reduce import check_tolerance

OUTPUTS = ("arrays", "csv", "npz", "summary", "png")
TOLERANCE_OUTPUTS = ("text", "summary")         # the outputs whose points the reduction chooses


def list_traces(path, plot: int = 0, ac_format: str = "magphase", names=None,
                header_only: bool = False) -> dict:
    """What a result file holds, with the extents to choose extract's parameters from.

    Keys, in order: path, format, analysis, plot, plots, x, x_range, points, sweeps,
    sweep_params, sweep_values, truncated, traces, ranges. x_range, points and the
    per-trace ranges ([min, max], NaN skipped; null when there is no value) cover the
    whole plot, every sweep.

    They come from one streaming pass over the data (reader.scan): it holds no trace in
    memory but reads the whole file. header_only=True reads the header alone, which is
    instant on any file: it leaves out x_range, points, sweep_values, truncated and
    ranges, and sweeps is then the count the header announces.

    names narrows traces and ranges, as extract's names do (names, raw names, globs).
    plot selects one plot of a Nutmeg rawfile; plots lists every plot in it (HSPICE: []).
    ac_format names the columns of complex (AC) variables, and is what they are ranged in.
    """
    if header_only:
        h = read_header(path, plot, ac_format)
        traces = [h.names[c] for c in resolve_columns(h, names) if c]
        return {"path": h.path, "format": h.version, "analysis": h.analysis, "plot": h.plot,
                "plots": h.plot_names, "x": h.x_name, "sweeps": max(1, h.sweep_count_hint),
                "sweep_params": h.sweep_params, "traces": traces}
    s = scan(path, plot, ac_format, names)
    h = s.header
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "plot": h.plot,
        "plots": h.plot_names, "x": h.x_name, "x_range": _json_range(s.x_range), "points": s.points,
        "sweeps": len(s.sweep_points), "sweep_params": h.sweep_params,
        "sweep_values": _json_sweep_values(s.sweep_values, h.dtype), "truncated": s.truncated,
        "traces": list(s.ranges), "ranges": {name: _json_range(b) for name, b in s.ranges.items()},
    }


def decimation_indices(n: int, downsample: Optional[int]):
    """Indices of `downsample` evenly spaced points out of n, first and last kept. None = keep all."""
    if downsample is None or downsample >= n:
        return None
    if downsample < 2:
        raise ValueError("downsample must be at least 2")
    return np.unique(np.linspace(0, n - 1, downsample).round().astype(np.intp))


def check_downsample(downsample) -> Optional[int]:
    """downsample as an int of at least 2; None passes through."""
    if downsample is None:
        return None
    try:
        if isinstance(downsample, bool):
            raise TypeError
        downsample = operator.index(downsample)
    except TypeError:
        raise ValueError(f"downsample must be an integer, not {downsample!r}") from None
    if downsample < 2:
        raise ValueError("downsample must be at least 2")
    return downsample


def _decimate(ts: TraceSet, downsample: int) -> None:
    for i in range(len(ts.sweep_values)):
        idx = decimation_indices(ts.data[ts.header.x_name][i].size, downsample)
        if idx is None:
            continue
        for name in ts.selected:
            ts.data[name][i] = ts.data[name][i][idx]


SUMMARY_MAX_POINTS = 1000       # cap on x/y points serialised in one summary, over all traces and sweeps:
                                # about 12k tokens of JSON, so a summary fits an agent's context


def _json_float(v) -> Optional[float]:
    """v as a JSON number, or None for NaN and infinities (strict JSON has no token for
    them). A float32 (HSPICE 9601 data) becomes its shortest decimal, 0.1 and not
    0.10000000149011612: that reads back as exactly the same float32."""
    if isinstance(v, np.float32):
        return float(str(v)) if math.isfinite(v) else None
    v = float(v)
    return v if math.isfinite(v) else None


def _json_floats(arr) -> List[Optional[float]]:
    """_json_float of every value of arr, float32 kept short as there."""
    a = np.asarray(arr)
    values = (float(t) for t in a.astype(str).tolist()) if a.dtype == np.float32 \
        else a.astype(np.float64).tolist()
    return [v if math.isfinite(v) else None for v in values]


def _json_range(bounds) -> Optional[List[Optional[float]]]:
    """[min, max] for JSON; None when there was no value to bound (both NaN)."""
    lo, hi = bounds
    if math.isnan(lo) and math.isnan(hi):
        return None
    return [_json_float(lo), _json_float(hi)]


def _json_sweep_values(sweep_values, dtype) -> List[List[Optional[float]]]:
    """Sweep parameter values were read in the file's dtype; keep a float32 one short."""
    return [_json_floats(np.asarray(v, dtype=dtype)) for v in sweep_values]


def nan_stats(a: np.ndarray) -> Tuple[float, float, float]:
    """(min, max, mean) over the values that are not NaN; NaN for each when there are none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN slice / mean of empty slice
        return np.nanmin(a), np.nanmax(a), np.nanmean(a)


def range_of(arrays) -> Tuple:
    """(min, max) over every value of the arrays, NaN skipped, in their dtype; NaN when
    there is no value."""
    bounds = [nan_min_max(np.asarray(a)) for a in arrays if len(a)]
    if not bounds:
        return math.nan, math.nan
    return np.fmin.reduce(np.array([b[0] for b in bounds])), np.fmax.reduce(np.array([b[1] for b in bounds]))


def _nan_mean(a: np.ndarray) -> float:
    """Mean of the values that are not NaN, accumulated in float64; NaN when there are none."""
    mean = a.mean(dtype=np.float64)
    if math.isnan(mean):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # mean of empty slice
            mean = np.nanmean(a, dtype=np.float64)
    return mean


def overview(ts: TraceSet) -> dict:
    """The facts every JSON result shares, for the data in ts: x, x_range, points, sweeps,
    sweep_params, sweep_values, sweep_indices, truncated, traces and ranges (per trace,
    [min, max] over every kept sweep, NaN skipped). Bounds are null when there is no value;
    NaN and infinities are null."""
    h = ts.header
    x = ts.data[h.x_name]
    traces = ts.selected[1:]
    return {
        "x": h.x_name, "x_range": _json_range(range_of(x)), "points": sum(int(a.size) for a in x),
        "sweeps": len(ts.sweep_values), "sweep_params": h.sweep_params,
        "sweep_values": _json_sweep_values(ts.sweep_values, h.dtype),
        "sweep_indices": list(ts.sweep_indices), "truncated": ts.truncated,
        "traces": traces, "ranges": {name: _json_range(range_of(ts.data[name])) for name in traces},
    }


def summarize(ts: TraceSet, downsample: Optional[int] = None, tolerance: Optional[float] = None) -> dict:
    """JSON-serialisable description of ts: the file facts and overview(ts), then under
    "stats" per trace and per sweep count, min, max, mean, first and last, and points
    when downsample or tolerance is given.

    The statistics cover every point in ts. A TraceSet already decimated (extract with
    output="arrays" and downsample) gives statistics of the decimated points.
    min, max and mean skip NaN values; they are null only when a trace holds nothing else.

    Points are x and y lists of real samples, chosen per trace and sweep by
    reduce.breakpoints: straight lines between them stay within tolerance (default
    reduce.DEFAULT_TOLERANCE) of every sample, as a fraction of the trace's range over ts
    (for an AC _Mag trace, its range in dB); max_error gives the bound met. A series too
    dense to follow within its points (an oscillation, a switching node) gets an
    "envelope" instead: {"from", "to", "min", "max"} lists, one entry per slice of x with
    the slice's bounds and the trace's lowest and highest sample in it (reduce.peak_detect).
    A series holds at most downsample points (or half as many slices), and all of them at
    most SUMMARY_MAX_POINTS, shared evenly by the (trace, sweep) series (at least 2 each,
    so the cap holds only up to SUMMARY_MAX_POINTS / 2 series). The statistics always grow
    with the number of series: select a few names or sweeps on large files. NaN and
    infinities are emitted as null.
    """
    h = ts.header
    x = ts.data[h.x_name]
    with_points = downsample is not None or tolerance is not None
    if with_points:
        series = max(1, (len(ts.selected) - 1) * len(ts.sweep_values))
        cap = max(2, SUMMARY_MAX_POINTS // series)
        if downsample is not None:
            cap = min(cap, downsample)
        tolerance = reduce.DEFAULT_TOLERANCE if tolerance is None else tolerance
        floor = 4 * float(np.finfo(h.dtype).eps)
        log = reduce.log_frequency(h)
        x_criteria = [reduce.x_criterion(a, log) for a in x]
    stats = {}
    for name in ts.selected[1:]:
        if with_points:
            db = reduce.in_db(h, name)
            criteria = [reduce.y_criterion(a, db) for a in ts.data[name]]
            span = reduce.finite_span(criteria)
        per_sweep = []
        for i, arr in enumerate(ts.data[name]):
            a = np.asarray(arr)
            entry = {"count": int(a.size)}
            if a.size:
                lo, hi = nan_min_max(a)
                entry.update(min=_json_float(lo), max=_json_float(hi), mean=_json_float(_nan_mean(a)),
                             first=_json_float(a[0]), last=_json_float(a[-1]))
                if with_points:
                    idx, error = reduce.breakpoints(x_criteria[i], [criteria[i]], [span], tolerance, cap, floor)
                    if error <= max(tolerance, floor):
                        entry["x"] = _json_floats(x[i][idx])
                        entry["y"] = _json_floats(a[idx])
                        entry["max_error"] = float(f"{error:.3g}")
                    else:                           # too dense to follow: a peak-detect envelope instead
                        start, stop, mins, maxs = reduce.peak_detect(x[i], [a], max(2, cap // 2), log)
                        entry["envelope"] = {"from": [float(f"{v:.6g}") for v in start.tolist()],
                                             "to": [float(f"{v:.6g}") for v in stop.tolist()],
                                             "min": _json_floats(mins[0]), "max": _json_floats(maxs[0])}
            per_sweep.append(entry)
        stats[name] = per_sweep
    return {"path": h.path, "format": h.version, "analysis": h.analysis, "plot": h.plot,
            "plots": h.plot_names, **overview(ts), "stats": stats}


def default_dest(path, output: str) -> str:
    """<root>_<ext>_traces.<output> beside the input, e.g. run_tr0_traces.csv."""
    root, ext = os.path.splitext(os.fspath(path))
    return f"{root}_{ext.lstrip('.')}_traces.{output}"


def resolve_dest(source, output: str, dest=None) -> str:
    """The path a csv/npz/png of `source` goes to; raises when it cannot be written safely:
    a directory, a missing parent directory, or the input file itself."""
    if not os.path.exists(source):                  # name the missing input, not its default dest
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.fspath(source))
    dest = os.fspath(dest) if dest is not None else default_dest(source, output)
    if os.path.isdir(dest):
        raise ValueError(f"dest {dest!r} is a directory; give a file path")
    parent = os.path.dirname(os.path.abspath(dest))
    if not os.path.isdir(parent):
        raise ValueError(f"dest {dest!r}: directory {parent!r} does not exist")
    if os.path.exists(dest) and os.path.samefile(dest, source):
        raise ValueError(f"dest {dest!r} is the input file; writing it would destroy the simulation result")
    return dest


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
        raise ImportError("output='png' needs the optional dependency: pip install 'spice_result_parser[plot]'") from exc
    return Figure


def make_figure(ts: TraceSet, yrange: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
    """A matplotlib Figure with one line per trace per sweep; log x axis for AC results.

    A line longer than PLOT_MAX_POINTS is drawn from its envelope (reduce.envelope): the
    first, last, lowest and highest sample of equal runs of samples, so the drawing looks
    as with every point and a one-sample spike still shows.
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
            idx = reduce.envelope(y, PLOT_MAX_POINTS)
            if idx is not None:
                x, y = x[idx], y[idx]
            ax.plot(x, y, linewidth=1, label=label)
    if reduce.log_frequency(h):
        ax.set_xscale("log")
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.set_xlabel(h.x_name)
    ax.grid(True, alpha=0.3)
    if 0 < len(ax.get_lines()) <= MAX_LEGEND_ENTRIES:
        ax.legend(fontsize="small")
    ax.set_title(title if title is not None else os.path.basename(h.path))
    return fig


def _write_png(ts: TraceSet, dest: str, yrange: Optional[Tuple[float, float]] = None,
               title: Optional[str] = None) -> None:
    make_figure(ts, validate_range("yrange", yrange, finite=True), title).savefig(dest, format="png")


def write_file(ts: TraceSet, output: str, dest=None, yrange=None, title: Optional[str] = None) -> str:
    """Write a TraceSet as csv, npz or png; returns the path written.

    yrange and title apply to png only (png needs matplotlib). dest defaults to
    <input>_<ext>_traces.<output> beside the input and must not be the input file.
    """
    if output not in ("csv", "npz", "png"):
        raise ValueError(f"write_file supports 'csv', 'npz' or 'png', not {output!r}")
    dest = resolve_dest(ts.header.path, output, dest)
    if output == "png":
        _write_png(ts, dest, yrange, title)
    elif output == "csv":
        _write_csv(ts, dest)
    else:
        _write_npz(ts, dest)
    return dest


def reject_tolerance(output: str, tolerance) -> None:
    """tolerance chooses the points of text and summary only: refuse it elsewhere rather
    than let a caller believe a file was reduced."""
    if tolerance is not None and output not in TOLERANCE_OUTPUTS:
        raise ValueError(f"tolerance applies to text and summary only, not to output={output!r}: "
                         f"arrays, csv and npz keep every point (downsample thins them evenly), "
                         f"and png draws every point's envelope")


def extract(path, names=None, sweeps=None, output: str = "arrays",
            downsample: Optional[int] = None, dest=None,
            xrange: Optional[Sequence[float]] = None, yrange: Optional[Sequence[float]] = None,
            plot: int = 0, ac_format: str = "magphase", tolerance: Optional[float] = None):
    """Read selected traces and return them as arrays, a summary dict, or a written file path.

    xrange=(lo, hi) keeps only the rows whose x value (TIME, FREQ or the sweep
    variable) lies in the closed interval; the window is applied while the file streams,
    so peak memory follows the window rather than the column length, and it applies to
    every output. yrange=(lo, hi) sets the vertical axis limits of the png plot; other
    outputs do not use it, but a malformed yrange still raises.
    downsample=N keeps N evenly spaced points per sweep (first and last kept; at least 2).
    For "summary" it caps the x/y points listed per trace and sweep, which the reduction
    chooses (see summarize); with neither downsample nor tolerance, summary lists none.
    tolerance (summary only) is the error bound of those points, a fraction of each
    trace's range (default 0.01).
    dest names the csv/npz/png file (default <input>_<ext>_traces.<output> beside the
    input); it must not be the input file, and other outputs reject it.
    plot selects one plot of a Nutmeg rawfile (0-based); see list_traces for the list.
    ac_format is what each complex (AC) variable becomes: "magphase" (_Mag, _Phase in
    degrees), "db" (_dB = 20 log10 |z|, _Phase) or "realimag" (_Re, _Im).
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    downsample = check_downsample(downsample)
    tolerance = check_tolerance(tolerance)
    reject_tolerance(output, tolerance)
    xrange = validate_range("xrange", xrange)
    yrange = validate_range("yrange", yrange, finite=True)
    if output in ("csv", "npz", "png"):
        resolve_dest(path, output, dest)            # fail before the read, not after it
    elif dest is not None:
        raise ValueError(f"dest applies to csv, npz and png only, not to output={output!r}")
    if output == "png":
        _require_matplotlib()
    ts = read_traces(path, names, sweeps, plot, xrange, ac_format)
    if output == "summary":
        return summarize(ts, downsample, tolerance)
    if downsample is not None:
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    return write_file(ts, output, dest, yrange)
