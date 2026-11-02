"""Shape-preserving reduction: which real samples to show, so that straight lines between
them stay close to every sample of a trace.

It serves the outputs that a person or an agent reads: the text report, the summary's
point lists and the png plot. csv, npz and arrays keep every point. Nothing here computes
new values: every index returned points at a sample of the file.

breakpoints() is a greedy Douglas-Peucker simplification with a vertical error bound, on
several traces that share their x values. When a signal is too dense for it within the
rows available (an oscillation, a switching node), peak_detect() gives the overview
instead: every trace's lowest and highest sample in each slice of x, as an
oscilloscope's peak-detect mode shows it. envelope() keeps the first, last, lowest and
highest sample of equal runs of samples, for drawing a line.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_TOLERANCE = 0.01    # fraction of each trace's range: about what a 100-pixel-high plot resolves


def check_tolerance(tolerance) -> Optional[float]:
    """tolerance as a float in [0, 1]; None passes through."""
    if tolerance is None:
        return None
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, (int, float, np.integer, np.floating)):
        raise ValueError(f"tolerance must be a number, a fraction of each trace's range such as 0.01 for 1%, "
                         f"not {tolerance!r}")
    value = float(tolerance)
    if not 0.0 <= value <= 1.0:                     # NaN fails too
        raise ValueError(f"tolerance must be between 0 and 1 (a fraction of each trace's range, such as "
                         f"0.01 for 1%), not {tolerance!r}")
    return value


def log_frequency(header) -> bool:
    """Whether x is a frequency that is read on a log axis: AC and noise analyses."""
    return header.analysis in ("ac", "noise")


def in_db(header, name: str) -> bool:
    """Whether trace name is a linear AC magnitude, which the reduction compares in dB, as on
    a Bode plot (else a roll-off to 1% of the peak would look flat)."""
    return header.ac_format == "magphase" and log_frequency(header) and name.endswith("_Mag")


def x_criterion(x: np.ndarray, log: bool) -> np.ndarray:
    """The x values the error is measured along: log10 of a frequency, when every one is positive."""
    x = np.asarray(x, dtype=np.float64)
    if log and x.size and bool(np.all(x > 0)):     # a NaN or a zero keeps the axis linear
        return np.log10(x)
    return x


def y_criterion(y: np.ndarray, db: bool) -> np.ndarray:
    """The values the error is measured in: 20 log10 |y| for a magnitude compared in dB
    (a zero becomes -inf, which is kept as a gap), else y itself."""
    if not db:
        return np.asarray(y)
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.abs(np.asarray(y, dtype=np.float64)))


def finite_span(arrays) -> float:
    """max - min over the finite values of the arrays; 0 when they hold no two distinct ones."""
    lo, hi = math.inf, -math.inf
    for a in arrays:
        a = np.asarray(a)
        if not a.size:
            continue
        finite = np.isfinite(a)
        if not finite.all():
            a = a[finite]
            if not a.size:
                continue
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    return hi - lo if hi > lo else 0.0


def _kinds(a: np.ndarray) -> np.ndarray:
    """0 finite, 1 NaN, 2 +inf, 3 -inf, per sample."""
    return np.where(np.isfinite(a), 0, np.where(np.isnan(a), 1, np.where(a > 0, 2, 3)))


def _boundaries(arrays) -> set:
    """Samples on both sides of every change between finite, NaN, +inf and -inf: they keep a
    gap or an overflow visible, and leave every segment between them of one kind per trace."""
    keep = set()
    for a in arrays:
        if np.isfinite(a).all():
            continue
        kinds = _kinds(a)
        change = np.flatnonzero(kinds[1:] != kinds[:-1])
        keep.update(change.tolist())
        keep.update((change + 1).tolist())
    return keep


def _extremes(channels) -> set:
    """The sample holding each trace's lowest and highest finite value (the first one on a tie)."""
    keep = set()
    for y, _ in channels:
        finite = np.isfinite(y)
        if not finite.any():
            continue
        keep.add(int(np.argmax(np.where(finite, y, -np.inf))))
        keep.add(int(np.argmin(np.where(finite, y, np.inf))))
    return keep


def breakpoints(x: np.ndarray, ys: Sequence[np.ndarray], spans: Sequence[float], tolerance: float,
                max_points: Optional[int] = None, floor: float = 0.0) -> Tuple[np.ndarray, float]:
    """The samples to keep of one sweep, and the error that keeping them leaves.

    x and ys are the criterion values (x_criterion, y_criterion) of traces that share x;
    spans are the traces' finite ranges over the whole selection, which the error is a
    fraction of. Straight lines between the kept samples stay within tolerance * span of
    every sample of every trace, measured vertically. A trace with span 0 (constant, or no
    finite value) needs no sample. The first and last samples are always kept, and so are
    both sides of every change between finite values, NaN and infinities; then, when
    max_points leaves room, each trace's lowest and highest sample, so a peak shows with
    its exact value and position.

    The segment with the largest relative error is split first, at its worst sample, so
    when max_points stops the split early the samples already kept are the ones that
    mattered most. floor raises a tolerance below the data's rounding (tolerance 0 then
    drops only the samples that lie on a straight line).

    Returns (sorted indices, error reached as a fraction of the span). The error is at most
    the tolerance unless max_points stopped the split; it is inf when the boundaries alone
    exceed max_points.
    """
    n = len(x)
    if n <= 2:
        return np.arange(n, dtype=np.intp), 0.0
    tol = max(tolerance, floor)
    x = np.asarray(x, dtype=np.float64)
    channels = [(np.asarray(y), float(s)) for y, s in zip(ys, spans) if s > 0]
    keep = {0, n - 1} | _boundaries([x] + [np.asarray(y) for y in ys])
    if max_points is not None and len(keep) > max_points:
        ordered = np.array(sorted(keep), dtype=np.intp)
        pick = np.unique(np.linspace(0, ordered.size - 1, max(2, max_points)).round().astype(np.intp))
        return ordered[pick], math.inf
    with_extremes = keep | _extremes(channels)
    if max_points is None or len(with_extremes) <= max_points:
        keep = with_extremes

    def worst(a: int, b: int) -> Tuple[float, int]:
        """Largest relative vertical error inside (a, b) over every trace, and where it is."""
        if b - a < 2 or not channels:
            return 0.0, -1
        seg = slice(a + 1, b)
        dx = x[b] - x[a]
        slope = (x[seg] - x[a]) / dx if dx != 0 and math.isfinite(dx) else None
        best, at = 0.0, -1
        for y, span in channels:
            ya, yb, inside = float(y[a]), float(y[b]), y[seg]
            with np.errstate(invalid="ignore", over="ignore"):
                if slope is not None:
                    err = np.abs(inside - (ya + (yb - ya) * slope))
                else:                               # a repeated x: the line is a vertical jump
                    err = np.maximum(min(ya, yb) - inside, inside - max(ya, yb))
                err = np.fmax(err / span, 0.0)      # NaN (a run of one non-finite kind) is no error
            k = int(np.argmax(err))
            if err[k] > best:
                best, at = float(err[k]), a + 1 + k
        return best, at

    heap = []
    ordered = sorted(keep)
    for a, b in zip(ordered, ordered[1:]):
        r, i = worst(a, b)
        if i >= 0:
            heap.append((-r, a, b, i))
    heapq.heapify(heap)
    while heap and -heap[0][0] > tol and (max_points is None or len(keep) < max_points):
        _, a, b, i = heapq.heappop(heap)
        keep.add(i)
        for s, e in ((a, i), (i, b)):
            r, j = worst(s, e)
            if j >= 0:
                heapq.heappush(heap, (-r, s, e, j))
    reached = -heap[0][0] if heap else 0.0
    return np.array(sorted(keep), dtype=np.intp), reached


def peak_detect(x: np.ndarray, ys: Sequence[np.ndarray], count: int, log: bool = False):
    """The envelope of traces that share x, as an oscilloscope's peak-detect mode shows it:
    the x range is cut into `count` slices of equal width (equal in log10 x on a log axis,
    when every x is positive), and each slice that holds samples gives its bounds and every
    trace's lowest and highest sample in it (NaN skipped; NaN when it holds nothing else).

    It is the overview for a signal too dense to follow with breakpoints (an oscillation, a
    switching node): its amplitude over x stays visible, and every value is a real sample.

    Returns (start, stop, mins, maxs): the bounds of the non-empty slices, in x units, and
    per trace an array of the lowest and of the highest sample of each.
    """
    x = np.asarray(x, dtype=np.float64)
    use_log = log and x.size > 0 and bool(np.all(x > 0))
    xc = np.log10(x) if use_log else x
    valid = np.isfinite(xc)
    if not valid.any():
        empty = np.empty(0)
        return empty, empty, [empty for _ in ys], [empty for _ in ys]
    all_valid = bool(valid.all())
    order = None if all_valid else np.flatnonzero(valid)    # None: every sample, in file order
    xv = xc if all_valid else xc[order]
    lo, hi = float(xv.min()), float(xv.max())
    count = max(1, int(count)) if hi > lo else 1
    width = (hi - lo) / count if hi > lo else 1.0
    slot = np.minimum(((xv - lo) / width).astype(np.intp), count - 1)
    if np.any(np.diff(slot) < 0):                   # x not monotonic: group the samples by slice
        sort = np.argsort(slot, kind="stable")
        order = sort if order is None else order[sort]
        slot = slot[sort]
    starts = np.flatnonzero(np.r_[True, slot[1:] != slot[:-1]])
    used = slot[starts]
    edges = lo + width * np.arange(count + 1, dtype=np.float64)
    edges[-1] = hi
    if use_log:
        edges = 10 ** edges
    mins, maxs = [], []
    for y in ys:
        values = np.asarray(y) if order is None else np.asarray(y)[order]
        with np.errstate(invalid="ignore"):
            mins.append(np.fmin.reduceat(values, starts))
            maxs.append(np.fmax.reduceat(values, starts))
    return edges[used], edges[used + 1], mins, maxs


def envelope(y: np.ndarray, max_points: int) -> Optional[np.ndarray]:
    """Indices that draw y as every sample would: the first, last, lowest and highest
    sample of each of max_points // 4 equal runs of samples, so a one-sample spike is never
    dropped. None when y has at most max_points samples."""
    n = len(y)
    if n <= max_points:
        return None
    runs = max(1, max_points // 4)
    bounds = np.linspace(0, n, runs + 1).round().astype(np.intp)
    keep: List[int] = []
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        part = np.asarray(y[a:b])
        keep += [a, b - 1]
        valid = ~np.isnan(part)
        if valid.any():
            keep += [a + int(np.nanargmin(part)), a + int(np.nanargmax(part))]
    return np.unique(np.array(keep, dtype=np.intp))
