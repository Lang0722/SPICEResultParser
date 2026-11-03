"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

import csv
import dataclasses
import errno
import math
import operator
import os
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import reader, reduce, waveview
from .reader import TraceSet, nan_min_max, read_header, read_traces, resolve_columns, scan, validate_range
from .reduce import check_tolerance

OUTPUTS = ("arrays", "csv", "npz", "summary", "png")
TOLERANCE_OUTPUTS = ("text", "summary")         # the outputs whose points the reduction chooses


def readers(engine: Optional[str] = None):
    """(read_header, read_traces, scan) of an engine: "python" (reader.py, nutmeg.py) or
    "waveview" (waveview.py). None uses the SRP_ENGINE environment variable, else "python"."""
    if waveview.resolve_engine(engine) == "waveview":
        return waveview.read_header, waveview.read_traces, waveview.scan
    return read_header, read_traces, scan


SCAN_MAX_BYTES = 1_000_000_000      # a larger file is large: list_traces reads only its header by default, and the
                                    # MCP extract reads it only with allow_large=True
WAVEVIEW_SCAN_MAX_VALUES = 2_000_000    # through WaveView, more values for the selected traces are large

# Bytes per second of a pass over the data, measured with the file in the page cache: (the
# scan of list_traces, read_traces of one trace). A disk can be slower: the slow estimate
# takes DISK_RATE. The text formats are slower than any disk: they parse every value.
PASS_RATES = {"9601": (600e6, 1.5e9), "2013": (330e6, 0.9e9), "2001": (1e9, 2e9), "ascii": (120e6, 120e6),
              "nutmeg": (1.5e9, 3e9), "nutmeg ascii": (75e6, 75e6)}
DISK_RATE = 200e6
WAVEVIEW_RATE = 250_000             # values that WaveView writes per second: 1 M in about 4 s on the test VM
_VALUE_BYTES = {"ascii": 13.2, "nutmeg ascii": 24.0}  # bytes of one value in a text file, about
_WAVEVIEW_VALUE_MEMORY = 33         # bytes per value through WaveView: its text, then a float64


@dataclasses.dataclass
class ReadCost:
    """What a read of the data costs, from the header (WaveView: its listing) alone. The
    Python engine reads the whole file for any names, sweeps and xrange; the WaveView engine
    writes every point of the selected signals, for any xrange."""
    engine: str
    bytes: int                      # size of the file
    points: int                     # points in the file, about (from its size); 0 for WaveView
    traces: int                     # selected traces
    values: int                     # WaveView: the values that it writes for them, x included; 0 for Python
    memory: int                     # bytes that x and the selected traces take without xrange, about
    seconds: Tuple[float, float]    # how long the read takes, about: the file in memory, the file on a disk
    large: bool                     # over SCAN_MAX_BYTES (WaveView: over WAVEVIEW_SCAN_MAX_VALUES)


def read_cost(path, plot: int = 0, ac_format: str = "magphase", names=None, engine: Optional[str] = None,
              scan: bool = False) -> ReadCost:
    """The cost of reading the traces that names selects, as read_traces reads them (scan=True:
    the pass of list_traces). It reads only the header; WaveView lists the file, and within
    waveview.listing_memo() that listing serves the read that follows."""
    read_head, _, _ = readers(engine)
    header = read_head(path, plot, ac_format)
    return _cost(path, header, resolve_columns(header, names), plot, ac_format, scan)


def _cost(path, header, cols: List[int], plot, ac_format: str, scan: bool) -> ReadCost:
    size = os.path.getsize(path)
    if header.version == "waveview":
        _, values, _ = waveview.scan_cost(path, plot, ac_format, [header.names[c] for c in cols[1:]])
        seconds = values / WAVEVIEW_RATE
        return ReadCost("waveview", size, 0, len(cols) - 1, values, values * _WAVEVIEW_VALUE_MEMORY,
                        (seconds, seconds), values > WAVEVIEW_SCAN_MAX_VALUES)
    kind = header.version + ("" if header.is_binary or header.version != "nutmeg" else " ascii")
    points = int(size / (header.stream_width * _VALUE_BYTES.get(kind, header.dtype.itemsize)))
    memory = points * sum(header.column_dtype(c).itemsize for c in cols)
    rate = PASS_RATES.get(kind, (DISK_RATE, DISK_RATE))[0 if scan else 1]
    return ReadCost("python", size, points, len(cols) - 1, 0, memory, (size / rate, size / min(rate, DISK_RATE)),
                    size > SCAN_MAX_BYTES)


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.2g} h"


def read_time(cost: ReadCost) -> str:
    """How long the read of cost takes, in words: 'about 1 to 8 min (...)'."""
    fast, slow = (_duration(s) for s in cost.seconds)
    if cost.engine == "waveview":
        return f"about {fast} (at the {WAVEVIEW_RATE:,} values per second of the test VM)"
    if fast == slow:
        return f"about {fast}"
    if fast.split()[1] == slow.split()[1]:
        fast = fast.split()[0]
    return f"about {fast} to {slow} (the file in memory, or on a disk that reads {_size(DISK_RATE)}/s)"


def _amount(n: int) -> str:
    """A count in few digits: 1.25e9, 3200."""
    return str(n) if n < 10_000 else f"{n:.3g}".replace("e+0", "e").replace("e+", "e")


def list_traces(path, plot: int = 0, ac_format: str = "magphase", names=None,
                header_only: Optional[bool] = None, engine: Optional[str] = None,
                max_traces: Optional[int] = None) -> dict:
    """What a result file holds, with the extents to choose extract's parameters from.

    Keys, in order: path, format, analysis, plot, plots, x, x_range, points, sweeps,
    sweep_params, sweep_values, truncated, trace_count, groups, traces, ranges, note.
    x_range, points and the per-trace ranges ([min, max], NaN skipped; null when there
    is no value) cover the whole plot, every sweep.

    They come from one streaming pass over the data (reader.scan): it holds no trace in
    memory but reads the whole file, whatever names selects. header_only=True reads the
    header alone, which is instant on any file: it leaves out x_range, points,
    sweep_values, truncated and ranges, and sweeps is then the count the header
    announces. header_only=None (the default) reads the data when that is fast: a file
    of at most SCAN_MAX_BYTES, and through WaveView at most WAVEVIEW_SCAN_MAX_VALUES
    values. Else it reads the header alone, and a note says why and how to read the data.

    names narrows traces and ranges, as extract's names do (names, raw names, globs).
    trace_count is the number of traces that it selects. max_traces lists at most that
    many of them in traces (and ranges): groups then counts all of them by name prefix
    (reader.name_groups, in at most max_traces and GROUPS_MAX keys), as requests for
    names, and a note says so.

    plot selects one plot of a Nutmeg rawfile; plots lists every plot in it (HSPICE: []).
    ac_format names the columns of complex (AC) variables, and is what they are ranged in.
    engine selects the reader, as in extract. The WaveView engine reports format
    "waveview"; it has no streaming pass, so it reads every point of the listed traces
    into memory to range them (header_only=True runs only WaveView's listing).
    """
    if max_traces is not None:
        max_traces = _count(max_traces, "max_traces")
    read_head, _, scan_plot = readers(engine)
    through_wv = waveview.resolve_engine(engine) == "waveview"
    with waveview.listing_memo():           # WaveView lists the file once for the header, the cost and the scan
        h = read_head(path, plot, ac_format)
        cols = resolve_columns(h, names)[1:]
        shown = cols if max_traces is None else cols[:max_traces]
        notes = []
        why = _own_x(path, h, shown, plot, ac_format) if through_wv and header_only is None else None
        cost = _cost(path, h, [0, *shown], plot, ac_format, scan=True)
        if why is None and cost.large:
            why = _large_note(cost, header_only is None)
        if why is not None:
            notes.append(why)
            header_only = True if header_only is None else header_only
        elif header_only is None:
            header_only = False
        if not header_only:
            s = scan_plot(path, plot, ac_format, names if len(shown) == len(cols) else [h.names[c] for c in shown])
    result = {"path": h.path, "format": h.version, "analysis": h.analysis, "plot": h.plot,
              "plots": h.plot_names, "x": h.x_name}
    if header_only:
        result.update(sweeps=max(1, h.sweep_count_hint), sweep_params=h.sweep_params)
    else:
        result.update(x_range=_json_range(s.x_range), points=s.points, sweeps=len(s.sweep_points),
                      sweep_params=h.sweep_params, sweep_values=_json_sweep_values(s.sweep_values, h.dtype),
                      truncated=s.truncated)
    result["trace_count"] = len(cols)
    if len(shown) < len(cols):
        result["groups"] = reader.name_groups([h.col_raw_names[c] for c in cols], min(reader.GROUPS_MAX, len(shown)))
        notes.append(f"traces lists the first {len(shown)} of {len(cols)} traces, and groups counts them all by "
                     f"name: pass a key of groups as names (to list_traces or extract) for its traces")
    result["traces"] = [h.names[c] for c in shown]
    if not header_only:
        result["ranges"] = {name: _json_range(b) for name, b in s.ranges.items()}
    if notes:
        result["note"] = "; ".join(notes)
    return result


def _count(value, name: str) -> int:
    """value as an int of at least 1, else a ValueError that names the parameter."""
    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, not {value!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n} bytes"


def _own_x(path, header, cols: List[int], plot, ac_format: str) -> Optional[str]:
    """Why list_traces does not range these traces through WaveView: a file whose signals have
    their own x points (extract.tcl stops at the second signal); None to range them."""
    signals, _, shared_x = waveview.scan_cost(path, plot, ac_format, [header.names[c] for c in cols])
    if signals > 1 and not shared_x:
        return ("list_traces read only the WaveView listing: in this file each signal has its own x points, "
                "and WaveView ranges one signal at a time (names with one trace)")
    return None


def _large_note(cost: ReadCost, skipped: bool) -> str:
    """list_traces' note on a large read (see ReadCost.large): what it costs, and with
    skipped (header_only=None) that list_traces did not make it and how to."""
    if cost.engine == "waveview":
        facts = (f"the ranges of these {cost.traces} traces take {cost.values:,} values from WaveView, more than "
                 f"the {WAVEVIEW_SCAN_MAX_VALUES:,} that list_traces reads by default, {read_time(cost)}")
        return (f"list_traces read only the WaveView listing: {facts}; header_only=False reads them, and names "
                f"with fewer traces takes fewer") if skipped else facts
    facts = (f"the file has {_size(cost.bytes)}, about {_amount(cost.points)} points, more than the "
             f"{_size(SCAN_MAX_BYTES)} that list_traces reads by default: a pass over the data goes through all "
             f"of it for any names, sweeps and xrange, {read_time(cost)}")
    return (f"list_traces read only the header: {facts}; header_only=False makes it, for x_range, points, "
            f"sweep_values, truncated and ranges") if skipped else facts


def value_eps(ts: TraceSet) -> float:
    """The relative rounding step of the values read: float32's for data stored in float32
    (HSPICE 9601), else float64's. Taken from the arrays, not the header, so that every
    engine gets the precision of its data; the reduction allows this much rounding."""
    steps = [float(np.finfo(a.dtype).eps) for name in ts.selected for a in ts.data[name]
             if np.issubdtype(a.dtype, np.floating)]
    return max(steps) if steps else float(np.finfo(ts.header.dtype).eps)


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
    """(min, max, mean) over the values that are not NaN; NaN for each when there are none.
    No warnings filter (it is not safe in threads): the NaN are dropped first."""
    kept = a[~np.isnan(a)]
    if not kept.size:
        return math.nan, math.nan, math.nan
    return kept.min(), kept.max(), kept.mean()


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
        kept = a[~np.isnan(a)]                  # not np.nanmean: it warns on an empty slice
        mean = kept.mean(dtype=np.float64) if kept.size else math.nan
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


DENSE_NAMES_MAX = 5             # traces a "too dense" note names; it counts the others


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _percent(fraction: float) -> str:
    return f"{100 * fraction:.3g}%"


def dense_reason(swung: List[int], errors: List[float], tolerance: float, unit: str, where: str = "") -> str:
    """Why a trace cannot be followed, for dense_phrase: it swings more often than its
    `unit` ("rows", "points") allow (swung: reduce.swings counts), else the best of them
    miss by errors (the fractions of its range that breakpoints reach). Each list holds one
    value per table or sweep where the trace is too dense; where says which ones."""
    if swung:       # a lower bound (reduce.swings): the smallest count holds in each table or sweep
        return f"it swings at least {min(swung)} times by more than {_percent(2 * tolerance)} of its range{where}"
    up_to = "up to " if len(errors) > 1 else ""
    finite = [e for e in errors if math.isfinite(e)]
    if finite:
        return f"its best {unit} miss by {up_to}{_percent(max(finite))} of its range{where}"
    return f"its changes between numbers, NaN and infinities alone need more {unit}{where}"


def _round_sig(value: float, digits: int, up: bool) -> float:
    """value (> 0) rounded up or down to `digits` significant digits."""
    scale = 10.0 ** (math.floor(math.log10(value)) - digits + 1)
    return float(f"{(math.ceil if up else math.floor)(value / scale) * scale:.12g}")


def zoom_advice(window: Optional[Tuple[float, float]], log: bool) -> Optional[Tuple[List[float], str]]:
    """A reduce.zoom_window (criterion values of x) as the caller passes it: ([lo, hi] in x,
    "xrange=[lo, hi], or another window up to <width> wide"). On a log-frequency axis the
    width is a factor, and the bounds are rounded inward to 3 digits. None for None."""
    if window is None:
        return None
    a, b = window
    if log:
        lo, hi = _round_sig(10.0 ** a, 3, up=True), _round_sig(10.0 ** b, 3, up=False)
        if not hi > lo:
            return None
        width = f"a factor of {hi / lo:.3g}"
    else:
        lo, hi = a, b
        width = f"{b - a:.3g}"
    return [lo, hi], f"xrange=[{lo:.6g}, {hi:.6g}], or another window up to {width} wide"


def dense_phrase(reasons: Dict[str, str], tolerance: float, budget: str, envelope: str,
                 zoom: Optional[str] = None) -> str:
    """What every output says of the traces that it cannot follow within the tolerance:

        v_osc is too dense to follow within 1% in 500 rows (it swings at least 1992 times by more
        than 2% of its range), so it is an envelope of 100 x slices; narrow xrange to see
        its waveform

    reasons maps each such trace to dense_reason; budget is "500 rows", "250 points" ...;
    envelope is what the caller gets instead ("an envelope of 100 x slices"). zoom is the
    window to look at next (zoom_advice): a caller that gets an envelope zooms in, so it
    learns how far at once. At most DENSE_NAMES_MAX traces are named; the others are
    counted."""
    names = list(reasons)
    one = len(names) == 1
    within = f"too dense to follow within {_percent(tolerance)} in {budget}"
    see = "its waveform" if one else "their waveforms"
    advice = f"to see {see}, zoom in with {zoom}" if zoom else f"narrow xrange to see {see}"
    if one:
        return f"{names[0]} is {within} ({reasons[names[0]]}), so it is {envelope}; {advice}"
    shown = names[:DENSE_NAMES_MAX]
    listed = [f"{name} ({reasons[name]})" for name in shown]
    if len(names) > len(shown):
        listed.append(f"{len(names) - len(shown)} more traces")
    envelopes = envelope.replace("an envelope", "envelopes", 1)
    return f"{_and(listed)} are {within}, so they are {envelopes}; {advice}"


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
    It also gets "too_dense": {"points": the points it had, and "swings": how often it
    swings by more than twice the tolerance (reduce.swings: more than those points can
    follow), or "max_error": what the best of the points reach, and "zoom": an xrange
    [lo, hi] that its points follow within the tolerance (reduce.zoom_window), the place
    to look next}. A top-level "note" names
    every such series, so a caller cannot take an envelope for the waveform.
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
        eps = value_eps(ts)
        log = reduce.log_frequency(h)
        x_criteria = [reduce.x_criterion(a, log) for a in x]
    stats = {}
    dense: Dict[str, list] = {}         # trace -> (sweep index, swings or None, error, i, entry) per enveloped series
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
                    swung = reduce.swings(x_criteria[i], criteria[i],
                                          reduce.swing_height(tolerance, span, criteria[i], eps))
                    if reduce.min_samples(swung) > cap:         # it swings more often than the points allow
                        idx, error = None, math.inf
                    else:
                        idx, error, swung = *reduce.breakpoints(x_criteria[i], [criteria[i]], [span], tolerance,
                                                                cap, eps), None
                    if error <= tolerance:
                        entry["x"] = _json_floats(x[i][idx])
                        entry["y"] = _json_floats(a[idx])
                        entry["max_error"] = float(f"{error:.3g}")
                    else:                           # too dense to follow: a peak-detect envelope instead
                        start, stop, mins, maxs = reduce.peak_detect(x[i], [a], max(1, cap // 2), log)
                        entry["too_dense"] = ({"points": cap, "swings": swung} if swung is not None else
                                              {"points": cap, "max_error": _json_float(float(f"{error:.3g}"))})
                        entry["envelope"] = {"from": _json_floats(start), "to": _json_floats(stop),
                                             "min": _json_floats(mins[0]), "max": _json_floats(maxs[0])}
                        dense.setdefault(name, []).append((ts.sweep_indices[i], swung, error, i, entry))
            per_sweep.append(entry)
        stats[name] = per_sweep
    result = {"path": h.path, "format": h.version, "analysis": h.analysis, "plot": h.plot,
              "plots": h.plot_names, **overview(ts), "stats": stats}
    if dense:
        nsweeps = len(ts.sweep_values)
        reasons, narrowest = {}, None
        for n, (name, hits) in enumerate(dense.items()):
            where = ""
            if nsweeps > 1:
                where = (" in every sweep" if len(hits) == nsweeps else
                         " in sweep" + ("s " if len(hits) > 1 else " ") + _and([str(h[0]) for h in hits]))
            reasons[name] = dense_reason([h[1] for h in hits if h[1] is not None],
                                         [h[2] for h in hits if h[1] is None], tolerance, "points", where)
            if n >= DENSE_NAMES_MAX:
                continue                                    # the note names these: zoom where it does
            # where to zoom: on the sweep where it swings most (or misses most), for every sweep
            i = max(hits, key=lambda h: (h[1] if h[1] is not None else -1, h[2]))[3]
            criterion = reduce.y_criterion(np.asarray(ts.data[name][i]), reduce.in_db(h, name))
            window = reduce.zoom_window(x_criteria[i], [criterion],
                                        [reduce.finite_span(reduce.y_criterion(np.asarray(a), reduce.in_db(h, name))
                                                            for a in ts.data[name])],
                                        tolerance, cap, eps)
            advice = zoom_advice(window, log)
            if advice is not None:
                for hit in hits:
                    hit[4]["too_dense"]["zoom"] = advice[0]
                if narrowest is None or window[1] - window[0] < narrowest[0]:
                    narrowest = (window[1] - window[0], advice[1])
        result["note"] = dense_phrase(reasons, tolerance, f"{cap} points",
                                      f"an envelope of {max(1, cap // 2)} x slices (from, to, min, max) "
                                      f"in place of x and y", narrowest[1] if narrowest else None)
    return result


def default_dest(path, output: str) -> str:
    """<root>_<ext>_traces.<output> beside the input, e.g. run_tr0_traces.csv."""
    root, ext = os.path.splitext(os.fspath(path))
    return f"{root}_{ext.lstrip('.')}_traces.{output}"


def resolve_dest(source, output: str, dest=None) -> str:
    """The path a csv/npz/png of `source` goes to; raises when it cannot be written safely:
    an empty path, a directory (or a path that ends like one), a parent that is not an
    existing directory, or the input file itself."""
    if not os.path.exists(source):                  # name the missing input, not its default dest
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.fspath(source))
    dest = os.fspath(dest) if dest is not None else default_dest(source, output)
    if not dest:
        raise ValueError("dest is empty; give a file path")
    if os.path.isdir(dest) or dest.endswith(("/", os.sep)) or os.path.basename(dest) in (".", ".."):
        raise ValueError(f"dest {dest!r} is a directory; give a file path")
    parent = os.path.dirname(dest) or os.curdir     # as written: abspath would drop a trailing "/."
    if not os.path.isdir(parent):
        raise ValueError(f"dest {dest!r}: directory {os.path.abspath(parent)!r} does not exist")
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
            plot: int = 0, ac_format: str = "magphase", tolerance: Optional[float] = None,
            engine: Optional[str] = None, max_values: Optional[int] = None):
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
    engine is the reader: "python" (this package's parser) or "waveview" (Synopsys
    WaveView in batch mode, see waveview.py: it reads every point of the selected
    traces, then applies xrange). None uses the SRP_ENGINE environment variable, else
    "python".
    max_values (WaveView only, see waveview.read_traces): the most values that WaveView
    writes. At the limit, extract raises waveview.ValueLimitError (a ValueError). The
    Python engine ignores it.
    """
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    read = readers(engine)[1]
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
    if read is waveview.read_traces:
        ts = read(path, names, sweeps, plot, xrange, ac_format, max_values=max_values)
    else:
        ts = read(path, names, sweeps, plot, xrange, ac_format)
    if output == "summary":
        return summarize(ts, downsample, tolerance)
    if downsample is not None and output != "png":  # png draws every point's envelope instead
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    return write_file(ts, output, dest, yrange)
