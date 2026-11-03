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
import re
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
    a Bode plot (else a roll-off to 1% of the peak would look flat). A '#<column>' suffix
    (a sanitized name that repeats) does not change what the trace is."""
    base = re.sub(r"#\d+$", "", name)
    return header.ac_format == "magphase" and log_frequency(header) and base.endswith("_Mag")


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
                max_points: Optional[int] = None, eps: float = 0.0) -> Tuple[np.ndarray, float]:
    """The samples to keep of one sweep, and the error that keeping them leaves.

    x and ys are the criterion values (x_criterion, y_criterion) of traces that share x;
    spans are the traces' finite ranges over the whole selection, which the error is a
    fraction of. Straight lines between the kept samples stay within tolerance * span of
    every sample of every trace, measured vertically. A trace with span 0 (constant, or no
    finite value) needs no sample. The first and last samples are always kept, and so are
    both sides of every change between finite values, NaN and infinities; then, when
    max_points leaves room, each trace's lowest and highest sample, so a peak shows with
    its exact value and position.

    The line between two kept samples runs along x where x is monotonic between them. Where
    x stays put (a vertical jump: PWL corners, HSPICE breakpoints), a sample between the
    jump's ends is no error. Where x goes back and forth between them (a nested sweep
    written flat, x restarting), the line runs along the sample order instead, so no inner
    curve can hide behind two kept samples at the same x.

    The segment with the largest relative error is split first, at its worst sample, so
    when max_points stops the split early the samples already kept are the ones that
    mattered most. eps is the data's machine epsilon: a deviation within the rounding of
    the values themselves (4 eps of their size) is no error, so tolerance 0 drops exactly
    the samples that lie on a straight line, whatever their offset.

    Returns (sorted indices, error reached as a fraction of the span). The error is at most
    the tolerance unless max_points stopped the split; it is inf when the boundaries alone
    exceed max_points.
    """
    n = len(x)
    if n <= 2:
        return np.arange(n, dtype=np.intp), 0.0
    tol = tolerance
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

    rounding = 4 * eps

    def worst(a: int, b: int) -> Tuple[float, int]:
        """Largest relative vertical error inside (a, b) over every trace, and where it is."""
        if b - a < 2 or not channels:
            return 0.0, -1
        seg = slice(a + 1, b)
        dx = x[b] - x[a]
        steps = np.diff(x[a:b + 1])
        if dx != 0 and math.isfinite(dx) and bool(np.all(steps >= 0) if dx > 0 else np.all(steps <= 0)):
            slope = (x[seg] - x[a]) / dx            # x monotonic: the line runs along x
        elif not steps.any():
            slope = None                            # x stays put: a vertical jump
        else:                                       # x goes back and forth: along the sample order
            slope = np.arange(1, b - a, dtype=np.float64) / (b - a)
        best, at = 0.0, -1
        for y, span in channels:
            ya, yb, inside = float(y[a]), float(y[b]), y[seg]
            with np.errstate(invalid="ignore", over="ignore"):
                if slope is not None:
                    line = ya + (yb - ya) * slope
                    err = np.abs(inside - line)
                    if rounding:
                        err = err - rounding * np.maximum(np.abs(inside), np.abs(line))
                else:
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


SWING_SAMPLES = 16384           # samples a swing count looks at: the extremes of SWING_SAMPLES // 4 runs


def _run_extremes(y: np.ndarray, runs: int) -> np.ndarray:
    """Sorted indices of the first, lowest, highest and last sample of `runs` equal runs of
    y (finite values only), in one numpy pass."""
    n = y.size
    size = -(-n // runs)
    full = n // size
    parts = []
    if full:
        block = y[:full * size].reshape(full, size)
        base = np.arange(full) * size
        parts += [base, base + block.argmin(axis=1), base + block.argmax(axis=1), base + size - 1]
    if full * size < n:
        start, tail = full * size, y[full * size:]
        parts.append(np.array([start, start + int(tail.argmin()), start + int(tail.argmax()), n - 1]))
    return np.unique(np.concatenate(parts))


def swings(x: np.ndarray, y: np.ndarray, height: float, positions: bool = False):
    """How many times y swings by more than height: the legs of a zigzag over its finite
    samples, each from one extreme to the next and more than height long. A sine of 50
    cycles from 0 to 0 has 101 legs that a height below its swing counts. positions=True
    also returns the x and the y of the legs' ends (swings + 1 of them), in order.

    The swings + 1 ends of the legs are extremes, each more than height from the one before.
    Between two samples that breakpoints keeps, a straight line can follow at most two of
    them within height / 2 (one line cannot pass near a peak and both valleys around it).
    So following y within tolerance * span takes at least min_samples(swings(x, y,
    swing_height(...))) samples, and a trace that swings more often than its rows allow is
    too dense without a breakpoints run (which costs about points * rows on such a signal).
    The count is a lower bound: samples at one x (a vertical jump, which breakpoints draws
    upright) count by their first and last one only, and a long y is cut to the extremes of
    SWING_SAMPLES // 4 runs. Both can only lower it."""
    keep = np.isfinite(y)
    if x.size > 2:
        inner = np.zeros(x.size, bool)
        inner[1:-1] = (x[1:-1] == x[:-2]) & (x[1:-1] == x[2:])
        keep &= ~inner
    at = np.flatnonzero(keep)
    v = np.asarray(y[at], dtype=np.float64)
    if v.size < 3 or not height > 0:
        return (0, np.empty(0), np.empty(0)) if positions else 0
    if v.size > SWING_SAMPLES:
        cut = _run_extremes(v, SWING_SAMPLES // 4)
        at, v = at[cut], v[cut]
    count, up, hi, lo, hi_at, lo_at = 0, None, v[0], v[0], 0, 0
    ends: List[int] = []                             # indices (in v) of the extremes that end a leg
    for k, value in enumerate(v[1:].tolist(), 1):
        if up is None:
            if value - lo > height:
                count, up, hi, hi_at = 1, True, value, k
                ends.append(lo_at)
            elif hi - value > height:
                count, up, lo, lo_at = 1, False, value, k
                ends.append(hi_at)
            else:
                if value > hi:
                    hi, hi_at = value, k
                if value < lo:
                    lo, lo_at = value, k
        elif up:
            if value > hi:
                hi, hi_at = value, k
            elif hi - value > height:
                count, up, lo, lo_at = count + 1, False, value, k
                ends.append(hi_at)
        elif value < lo:
            lo, lo_at = value, k
        elif value - lo > height:
            count, up, hi, hi_at = count + 1, True, value, k
            ends.append(lo_at)
    if not positions:
        return count
    if count:
        ends.append(hi_at if up else lo_at)          # the end of the last leg
    ends_at = np.asarray(ends, dtype=np.intp)
    return count, np.asarray(x[at[ends_at]], dtype=np.float64), v[ends_at]


def _nice_floor(value: float) -> float:
    """The largest 1, 2 or 5 times a power of ten that is at most value (value > 0)."""
    power = 10.0 ** math.floor(math.log10(value))
    return next(step * power for step in (5.0, 2.0, 1.0) if step * power <= value * (1 + 1e-12))


def zoom_window(x: np.ndarray, ys: Sequence[np.ndarray], spans: Sequence[float], tolerance: float,
                budget: int, eps: float) -> Optional[Tuple[float, float]]:
    """A window of x, in the criterion values (x_criterion), that `budget` samples follow
    within tolerance (breakpoints, all of ys together): what a caller that got an envelope
    passes as xrange to see the waveform.

    Each swing (see swings) of length L takes about (pi / 4) sqrt(L / (tolerance * span))
    samples: what straight lines need to follow a sine within the tolerance (7.9 for a
    swing over the whole range at 1%). Summed over a sliding window, that finds where a
    width is hardest to follow, and the widest width that the estimate allows. breakpoints
    then checks it at that place, with each trace ranged over the window as the zoomed
    call ranges it (a window of a random walk spans far less than the whole), and narrows
    it until it fits (or widens it, where the estimate was too careful, as for a square
    wave). The width is then rounded down to a 1,
    2 or 5 times a power of ten, and its start, near the hardest place, to a tenth of it;
    that window is checked as the caller will pass it. So another window of that width fits
    too, unless the signal is harder there than the estimate says. None when x is not
    monotonic (an x window cannot pick out a part of it) or when every sample already fits."""
    n = len(x)
    if n <= budget or budget < 3 or not bool(np.all(np.diff(x) >= 0)):
        return None
    total = float(x[-1] - x[0])
    if not total > 0:
        return None
    # any window narrower than the closest `budget` samples holds fewer of them: it fits
    floor = float(np.min(x[budget - 1:] - x[:n - budget + 1]))
    if not floor > 0:
        floor = total / n
    where, cost = [np.empty(0)], [np.empty(0)]
    for y, span in zip(ys, spans):
        count, at, value = swings(x, y, swing_height(tolerance, span, y, eps), positions=True)
        if count and span > 0:
            where.append((at[1:] + at[:-1]) / 2)                       # each leg at its middle
            scale = tolerance * span if tolerance > 0 else span / budget ** 2
            cost.append(math.pi / 4 * np.sqrt(np.abs(np.diff(value)) / scale))
    order = np.argsort(np.concatenate(where), kind="stable")
    where, cost = np.concatenate(where)[order], np.concatenate(cost)[order]
    summed = np.concatenate([[0.0], np.cumsum(cost)])

    def hardest(w: float) -> Tuple[float, float]:
        """(start, estimated samples) of the window of width w that the estimate finds hardest."""
        if not where.size:
            return float(x[0]), 2.0
        held = summed[np.searchsorted(where, where + w, side="right")] - summed[:-1]
        k = int(np.argmax(held))
        return float(min(max(where[k], x[0]), x[-1] - w)), float(held[k]) + 2

    def fits_at(a: float, w: float) -> bool:
        """Whether the samples in [a, a + w] fit, ranged as the zoomed call ranges them: the
        tolerance is a fraction of each trace's range in the selection, so in the window."""
        i, j = int(np.searchsorted(x, a, "left")), int(np.searchsorted(x, a + w, "right"))
        if j - i <= budget:
            return True
        part = [y[i:j] for y in ys]
        _, error = breakpoints(x[i:j], part, [finite_span([p]) for p in part], tolerance, budget, eps)
        return error <= tolerance

    lo, hi = floor, total / 2                               # the widest width the estimate allows
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if hardest(mid)[1] <= budget:
            lo = mid
        else:
            hi = mid
        if hi <= lo * 1.01:
            break
    width = lo
    if fits_at(hardest(width)[0], width):
        for _ in range(2):                                  # wider, where the estimate was too careful
            if 2 * width > total / 2 or not fits_at(hardest(2 * width)[0], 2 * width):
                break
            width *= 2
    else:                                                   # narrower: floor fits, width does not
        lo, hi = floor, width
        for _ in range(16):
            if hi <= lo * 1.25:
                break
            mid = math.sqrt(lo * hi)
            if fits_at(hardest(mid)[0], mid):
                lo = mid
            else:
                hi = mid
        width = lo
    nice = _nice_floor(width)
    for _ in range(6):                                      # a round window, checked as the caller passes it
        step = nice / 10
        a = math.floor(hardest(nice)[0] / step) * step
        a, b = float(f"{a:.12g}"), float(f"{a + nice:.12g}")
        if fits_at(a, b - a):
            return a, b
        nice = _nice_floor(nice * 0.99)
    return None


def swing_height(tolerance: float, span: float, y: np.ndarray, eps: float) -> float:
    """The swing that a line within tolerance * span cannot cut: twice the error allowed,
    with breakpoints' allowance for rounding (4 eps of the values) and a margin on top."""
    finite = y[np.isfinite(y)]
    size = float(np.max(np.abs(finite))) if finite.size else 0.0
    return 2 * (tolerance * span + 4 * eps * size) * (1 + 1e-9)


def min_samples(swing_count: int) -> int:
    """The fewest samples that follow a trace with swing_count swings (see swings): its
    swing_count + 1 extremes, at most two per straight line, and one sample more."""
    return (swing_count + 2) // 2 + 1


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
