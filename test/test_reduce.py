"""Tests for the shape-preserving reduction (reduce.py) and the outputs that use it."""
import asyncio
import json
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402

from spice_result_parser import api, reduce  # noqa: E402
from spice_result_parser.reader import read_traces  # noqa: E402


def polyline_error(x, y, keep):
    """Largest vertical distance from any sample to the straight lines between the kept
    samples, computed independently of reduce.breakpoints (a repeated x is a vertical jump;
    a non-finite sample counts only where the lines through it are finite)."""
    worst = 0.0
    for a, b in zip(keep[:-1], keep[1:]):
        inside = np.arange(a + 1, b)
        if not inside.size:
            continue
        with np.errstate(invalid="ignore", over="ignore"):
            if x[b] != x[a]:
                line = y[a] + (y[b] - y[a]) * (x[inside] - x[a]) / (x[b] - x[a])
                err = np.abs(y[inside] - line)
            else:
                err = np.maximum(np.maximum(min(y[a], y[b]) - y[inside], y[inside] - max(y[a], y[b])), 0)
        err = err[np.isfinite(err)]
        if err.size:
            worst = max(worst, float(err.max()))
    return worst


def signals(n=4001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1e-6, n)
    step = np.where(t < 2e-7, 0.0, 1.2 * (1 - np.exp(-(t - 2e-7) / 3e-8) * np.cos(2 * np.pi * 2e7 * (t - 2e-7))))
    return t, {
        "sine": np.sin(2 * np.pi * 5e6 * t),
        "ringing step": step,
        "noise": rng.standard_normal(n),
        "glitch": np.where((t > 6e-7) & (t < 6.02e-7), 0.4, 0.0),
        "ramp": 3.0 * t / 1e-6,
    }


class TestBreakpoints(unittest.TestCase):
    def test_every_sample_is_within_the_tolerance(self):
        t, sig = signals()
        for name, y in sig.items():
            span = reduce.finite_span([y])
            for tol in (0.1, 0.01, 0.001):
                with self.subTest(signal=name, tolerance=tol):
                    keep, reached = reduce.breakpoints(t, [y], [span], tol)
                    self.assertEqual((keep[0], keep[-1]), (0, t.size - 1))
                    self.assertTrue(np.all(np.diff(keep) > 0))
                    self.assertLessEqual(reached, tol)
                    self.assertLessEqual(polyline_error(t, y, keep), tol * span * (1 + 1e-9))
                    self.assertAlmostEqual(polyline_error(t, y, keep) / span, reached, places=12)

    def test_smaller_tolerance_keeps_more(self):
        t, sig = signals()
        y = sig["ringing step"]
        counts = [reduce.breakpoints(t, [y], [reduce.finite_span([y])], tol)[0].size
                  for tol in (0.1, 0.01, 0.001)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])
        self.assertLess(counts[2], t.size / 4)

    def test_straight_lines_need_two_samples(self):
        t, sig = signals()
        keep, reached = reduce.breakpoints(t, [sig["ramp"]], [3.0], 0.0, eps=np.finfo(float).eps)
        self.assertEqual(keep.tolist(), [0, t.size - 1])
        self.assertLess(reached, 1e-12)                          # rounding only

    def test_tolerance_zero_keeps_every_corner(self):
        t = np.arange(7.0)
        y = np.array([0.0, 1, 2, 2, 2, 5, 8])                   # corners at 2, 4; 1, 3, 6 lie on lines
        keep, _ = reduce.breakpoints(t, [y], [8.0], 0.0, eps=np.finfo(float).eps)
        self.assertEqual(keep.tolist(), [0, 2, 4, 6])

    def test_budget_keeps_the_worst_first_and_reports_the_error_left(self):
        t, sig = signals()
        y, span = sig["ringing step"], reduce.finite_span([sig["ringing step"]])
        for budget in (2, 3, 5, 10, 20):
            keep, reached = reduce.breakpoints(t, [y], [span], 0.0001, budget)
            self.assertEqual(keep.size, budget)
            self.assertGreater(reached, 0.0001)                  # the budget binds before the tolerance
            self.assertAlmostEqual(polyline_error(t, y, keep) / span, reached, places=12)
        keep, _ = reduce.breakpoints(t, [y], [span], 0.0001, 3)
        self.assertEqual(int(keep[1]), int(np.argmax(y)))        # room for one more: the overshoot peak
        keep, _ = reduce.breakpoints(t, [y], [span], 0.0001, 2)
        self.assertEqual(keep.tolist(), [0, t.size - 1])         # no room: the ends alone

    def test_each_trace_keeps_its_exact_extremes(self):
        t, sig = signals()
        for name, y in sig.items():
            keep, _ = reduce.breakpoints(t, [y], [reduce.finite_span([y])], 0.05)
            self.assertIn(int(np.argmax(y)), keep.tolist(), name)
            self.assertIn(int(np.argmin(y)), keep.tolist(), name)

    def test_traces_share_their_samples(self):
        t = np.linspace(0, 1, 2001)
        early = np.where(t < 0.2, np.sin(40 * t), 0.0)           # busy early, flat late
        late = np.where(t > 0.8, np.sin(40 * t), 0.0)
        spans = [reduce.finite_span([early]), reduce.finite_span([late])]
        keep, reached = reduce.breakpoints(t, [early, late], spans, 0.01)
        self.assertLessEqual(reached, 0.01)
        for y, span in zip((early, late), spans):
            self.assertLessEqual(polyline_error(t, y, keep), 0.01 * span * (1 + 1e-9))
        self.assertTrue((t[keep] < 0.2).sum() > 5 and (t[keep] > 0.8).sum() > 5)
        self.assertLess(((t[keep] > 0.3) & (t[keep] < 0.7)).sum(), 2)   # nothing happens there

    def test_a_trace_without_range_needs_no_sample(self):
        t = np.linspace(0, 1, 101)
        keep, reached = reduce.breakpoints(t, [np.full(101, 2.5), np.full(101, np.nan)], [0.0, 0.0], 0.01)
        self.assertEqual((keep.tolist(), reached), ([0, 100], 0.0))

    def test_nan_and_inf_runs_stay_visible(self):
        t = np.arange(12.0)
        y = np.array([0, 1, 2, np.nan, np.nan, 5, 6, np.inf, 8, 9, -np.inf, 11.0])
        keep, _ = reduce.breakpoints(t, [y], [reduce.finite_span([y])], 0.5)
        for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11):               # both sides of every change of kind
            self.assertIn(i, keep.tolist())
        self.assertEqual(reduce.finite_span([y]), 11.0)

    def test_too_many_boundaries_for_the_budget(self):
        t = np.arange(100.0)
        y = np.where(np.arange(100) % 2, np.nan, 1.0)            # a gap at every other sample
        keep, reached = reduce.breakpoints(t, [y], [0.0], 0.01, 10)
        self.assertLessEqual(keep.size, 10)
        self.assertEqual((keep[0], keep[-1]), (0, 99))
        self.assertEqual(reached, np.inf)

    def test_repeated_x_is_a_vertical_jump(self):
        t = np.array([0.0, 1, 1, 2, 3])                          # a PWL step: two samples at t = 1
        y = np.array([0.0, 0, 1, 1, 1])
        keep, reached = reduce.breakpoints(t, [y], [1.0], 0.01)
        self.assertEqual(keep.tolist(), [0, 1, 2, 4])
        self.assertEqual(reached, 0.0)

    def test_a_glitch_survives_a_huge_trace(self):
        t = np.linspace(0, 1e-6, 1_000_001)
        y = np.zeros_like(t)
        y[777_777] = 1.0                                         # one sample
        keep, _ = reduce.breakpoints(t, [y], [1.0], 0.01, 50)
        self.assertIn(777_777, keep.tolist())

    def test_speed_on_a_million_points(self):
        t = np.linspace(0, 1e-6, 1_000_000)
        ys = [np.sin(2 * np.pi * k * t / 1e-6) + (t > 5e-7) for k in (3, 7)]
        start = time.perf_counter()
        keep, _ = reduce.breakpoints(t, ys, [reduce.finite_span([y]) for y in ys], 0.01, 500)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertLessEqual(keep.size, 500)


class TestCriteria(unittest.TestCase):
    def lowpass(self):
        f = np.logspace(0, 8, 161)                              # 20 points per decade
        h = 1 / (1 + 1j * f / 1e3) ** 2                         # -40 dB/decade above 1 kHz
        return f, np.abs(h)

    def test_frequency_is_reduced_on_a_log_axis(self):
        # on a log axis the -40 dB/decade asymptote is a straight line: no point needed on it,
        # while the knee around 1 kHz needs several; on a linear axis the same line is curved
        f, mag = self.lowpass()
        db = reduce.y_criterion(mag, True)
        span = reduce.finite_span([db])
        log, _ = reduce.breakpoints(reduce.x_criterion(f, True), [db], [span], 0.01)
        linear, _ = reduce.breakpoints(reduce.x_criterion(f, False), [db], [span], 0.01)
        self.assertGreaterEqual(int(((f[log] > 1e2) & (f[log] < 1e4)).sum()), 2)
        self.assertLessEqual(int((f[log] > 1e5).sum()), 1)       # the end only
        self.assertGreater(int((f[linear] > 1e5).sum()), 3)

    def test_magnitude_in_db_keeps_the_roll_off(self):
        f, mag = self.lowpass()
        x = reduce.x_criterion(f, True)
        in_db, _ = reduce.breakpoints(x, [reduce.y_criterion(mag, True)],
                                      [reduce.finite_span([reduce.y_criterion(mag, True)])], 0.01)
        linear, _ = reduce.breakpoints(x, [mag], [reduce.finite_span([mag])], 0.01)
        stop = lambda keep: int((f[keep] > 1e5).sum())           # noqa: E731
        self.assertLessEqual(stop(linear), 1)                    # linear: below 1% of the peak it is "flat"
        self.assertGreaterEqual(stop(in_db), 1)                  # dB: the straight -40 dB/decade line ends there
        self.assertTrue(np.all(reduce.y_criterion(np.array([1.0, 0.1, 0.0]), True)[:2] == [0.0, -20.0]))
        self.assertEqual(reduce.y_criterion(np.array([0.0]), True)[0], -np.inf)

    def test_which_traces_use_log_frequency_and_db(self):
        h = read_traces(HERE / "test_9601.ac0", ["v(vo)"]).header
        self.assertTrue(reduce.log_frequency(h))
        self.assertTrue(reduce.in_db(h, "v_vo_Mag"))
        self.assertFalse(reduce.in_db(h, "v_vo_Phase"))
        db = read_traces(HERE / "test_9601.ac0", ["v(vo)"], ac_format="db").header
        self.assertFalse(reduce.in_db(db, "v_vo_dB"))          # already in dB
        tr = read_traces(HERE / "test_9601.tr0", ["v(vo"]).header
        self.assertFalse(reduce.log_frequency(tr))
        self.assertTrue(np.array_equal(reduce.x_criterion(np.array([0.0, 1.0]), True), [0.0, 1.0]))


class TestEnvelope(unittest.TestCase):
    def test_short_lines_are_drawn_whole(self):
        self.assertIsNone(reduce.envelope(np.arange(10.0), 10))

    def test_extremes_and_ends_are_kept(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(100_000)
        y[31_415] = 50.0
        y[27_182] = -50.0
        keep = reduce.envelope(y, 400)
        self.assertLessEqual(keep.size, 400)
        self.assertEqual((keep[0], keep[-1]), (0, y.size - 1))
        self.assertTrue({31_415, 27_182} <= set(keep.tolist()))
        runs = np.array_split(np.arange(y.size), 100)            # every run's own min and max survive
        for run in runs[:5]:
            self.assertIn(run[0] + int(np.argmax(y[run])), keep.tolist())

    def test_nan_runs(self):
        y = np.full(1000, np.nan)
        y[500] = 1.0
        keep = reduce.envelope(y, 40)
        self.assertIn(500, keep.tolist())


class TestPeakDetect(unittest.TestCase):
    def brute_force(self, x, y, count, log=False):
        """The slices and their extremes, one slice at a time."""
        xc = np.log10(x) if log else x
        edges = np.linspace(xc.min(), xc.max(), count + 1)
        rows = []
        for k in range(count):
            inside = (xc >= edges[k]) & ((xc < edges[k + 1]) if k < count - 1 else (xc <= edges[k + 1]))
            if inside.any():
                bounds = (10 ** edges[k], 10 ** edges[k + 1]) if log else (edges[k], edges[k + 1])
                rows.append((*bounds, np.nanmin(y[inside]), np.nanmax(y[inside])))
        return np.array(rows)

    def test_matches_slice_by_slice(self):
        rng = np.random.default_rng(6)
        x = np.sort(rng.uniform(0, 1e-6, 5000))
        y = rng.standard_normal(5000)
        start, stop, mins, maxs = reduce.peak_detect(x, [y, -y], 37)
        ref = self.brute_force(x, y, 37)
        np.testing.assert_allclose(np.c_[start, stop], ref[:, :2], rtol=1e-12)
        np.testing.assert_array_equal(mins[0], ref[:, 2])
        np.testing.assert_array_equal(maxs[0], ref[:, 3])
        np.testing.assert_array_equal(mins[1], -maxs[0])           # every trace on the same slices
        self.assertTrue(set(mins[0]) <= set(y) and set(maxs[0]) <= set(y))   # real samples

    def test_empty_slices_are_left_out(self):
        x = np.r_[np.linspace(0, 0.9, 50), np.linspace(9.1, 10, 50)]   # nothing between 0.9 and 9.1
        start, stop, _, _ = reduce.peak_detect(x, [x], 10)
        self.assertEqual(start.size, 2)
        self.assertEqual((start[0], stop[0], start[1], stop[1]), (0.0, 1.0, 9.0, 10.0))
        start, _, _, _ = reduce.peak_detect(np.array([0.0, 1.0, 10.0]), [np.zeros(3)], 10)
        self.assertEqual(start.tolist(), [0.0, 1.0, 9.0])               # a sample on an edge opens the next slice

    def test_unordered_x_nan_x_and_nan_values(self):
        x = np.array([0.5, 0.1, np.nan, 0.9, 0.2, 0.6])
        y = np.array([5.0, 1.0, 99.0, 9.0, np.nan, 6.0])
        start, stop, mins, maxs = reduce.peak_detect(x, [y], 2)
        np.testing.assert_allclose(np.c_[start, stop], [[0.1, 0.5], [0.5, 0.9]])
        self.assertEqual((mins[0].tolist(), maxs[0].tolist()), ([1.0, 5.0], [1.0, 9.0]))   # 99 has no x

    def test_log_axis_and_single_x(self):
        f = np.logspace(0, 6, 61)
        y = np.log10(f)
        start, stop, mins, _ = reduce.peak_detect(f, [y], 6, log=True)
        np.testing.assert_allclose(start, [1, 10, 100, 1e3, 1e4, 1e5])   # a decade each
        np.testing.assert_allclose(mins[0], [0, 1, 2, 3, 4, 5])
        start, stop, mins, _ = reduce.peak_detect(np.full(5, 2.0), [np.arange(5.0)], 10)
        self.assertEqual((start.tolist(), stop.tolist(), mins[0].tolist()), ([2.0], [2.0], [0.0]))


class TestTolerance(unittest.TestCase):
    def test_validation(self):
        self.assertIsNone(reduce.check_tolerance(None))
        self.assertEqual(reduce.check_tolerance(0), 0.0)
        self.assertEqual(reduce.check_tolerance(1), 1.0)
        self.assertEqual(reduce.check_tolerance(np.float32(0.5)), 0.5)
        for bad in (True, "0.01", -0.1, 1.5, float("nan"), [0.01]):
            with self.assertRaisesRegex(ValueError, "tolerance"):
                reduce.check_tolerance(bad)

    def test_file_outputs_refuse_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            for output in ("arrays", "csv", "npz", "png"):
                with self.assertRaisesRegex(ValueError, "tolerance applies to text and summary only"):
                    api.extract(HERE / "test_9601.tr0", output=output, tolerance=0.01,
                                dest=None if output == "arrays" else str(Path(tmp) / f"o.{output}"))


class TestOutputs(unittest.TestCase):
    """The reduction as an agent sees it: text rows and summary points."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = np.linspace(0, 100e-9, 200_001)
        v = np.where(t < 10e-9, 0.0, 1.2 * (1 - np.exp(-(t - 10e-9) / 2e-9) * np.cos(2 * np.pi * 0.5e9 * (t - 10e-9))))
        v[(t > 70e-9) & (t < 71e-9)] += 0.35                   # a 1 ns glitch
        cls.path = Path(cls.tmp.name) / "step.tr0"
        fixtures.write_binary(cls.path, "2001", ["TIME", "v(out"], [1, 1], [([], np.c_[t, v])])
        cls.t, cls.v = t, v
        from spice_result_parser import mcp_server
        cls.m = mcp_server

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.split()[:1] == ["TIME"])
        return np.array([[float(c) for c in line.split()] for line in lines[start + 1:]])

    def test_text_shows_the_peak_and_the_glitch(self):
        text = self.m.format_text(read_traces(self.path))
        head = text.splitlines()[0]
        self.assertIn("200001 points (", head)
        self.assertIn(" shown, each trace within 1% of its range)", head)
        rows = self.rows(text)
        self.assertLess(len(rows), 60)
        peak = int(np.argmax(self.v[self.t < 60e-9]))
        self.assertIn(round(self.t[peak], 15), [round(x, 15) for x in rows[:, 0]])   # when the overshoot peaks
        glitch = rows[(rows[:, 0] >= 70e-9) & (rows[:, 0] <= 71e-9)]
        self.assertTrue(np.any(np.abs(glitch[:, 1] - 1.55) < 1e-3))                   # the glitch is there
        # the rows are real samples
        for x, y in rows:
            i = int(np.argmin(np.abs(self.t - x)))
            self.assertAlmostEqual(y, self.v[i], delta=1e-5 * 2)

    def test_a_tighter_tolerance_shows_more(self):
        ts = read_traces(self.path)
        coarse, fine = self.m.format_text(ts, tolerance=0.05), self.m.format_text(ts, tolerance=0.002)
        self.assertLess(len(self.rows(coarse)), len(self.rows(fine)))
        self.assertIn("within 0.2% of its range", fine.splitlines()[0])

    def test_a_window_on_a_small_signal_shows_its_detail(self):
        # the tolerance is relative to the selection's range: on the settling tail that range
        # is millivolts, so the window shows the ringing the whole view rounds away
        whole = self.rows(self.m.format_text(read_traces(self.path)))
        tail = self.rows(self.m.format_text(read_traces(self.path, xrange=(20e-9, 60e-9))))
        self.assertLess(((whole[:, 0] >= 20e-9) & (whole[:, 0] <= 60e-9)).sum(), 3)
        self.assertGreater(len(tail), 15)
        span = tail[:, 1].max() - tail[:, 1].min()
        self.assertLess(span, 0.02)
        # a window that still holds the whole swing keeps the same detail: the range did not change
        swing = self.rows(self.m.format_text(read_traces(self.path, xrange=(10e-9, 14e-9))))
        self.assertEqual(len(swing), ((whole[:, 0] >= 10e-9) & (whole[:, 0] <= 14e-9)).sum())

    def test_summary_points(self):
        s = api.extract(self.path, output="summary", tolerance=0.01)
        entry = s["stats"]["v_out"][0]
        self.assertLessEqual(entry["max_error"], 0.01)
        self.assertLess(len(entry["x"]), 60)
        keep = np.searchsorted(self.t, entry["x"])
        span = s["ranges"]["v_out"][1] - s["ranges"]["v_out"][0]
        self.assertLessEqual(polyline_error(self.t, self.v, keep), 0.01 * span * (1 + 1e-9))
        self.assertNotIn("x", api.extract(self.path, output="summary")["stats"]["v_out"][0])
        capped = api.extract(self.path, output="summary", downsample=5)["stats"]["v_out"][0]
        self.assertNotIn("x", capped)                            # 5 points cannot follow it: an envelope
        self.assertEqual(len(capped["envelope"]["from"]), 2)
        self.assertEqual(max(capped["envelope"]["max"]), entry["max"])
        json.dumps(s, allow_nan=False)

    def test_constant_traces_are_one_line(self):
        text = self.m.format_text(read_traces(HERE / "test_9601.ac0"))
        self.assertIn("constant: v_0_Mag = 0, v_0_Phase = 0, v_vs_Mag = 1, v_vs_Phase = 0", text)
        heading = [l.split() for l in text.splitlines() if l.lstrip().startswith("HERTZ")]
        self.assertEqual(heading, [["HERTZ", "v_vo_Mag", "v_vo_Phase", "i_vs_Mag", "i_vs_Phase"]])
        self.assertIn("_Mag compared in dB", text.splitlines()[0])
        self.assertNotIn("v_0_Mag", text.split("constant:")[1].split("\n", 1)[1])


class TestEnvelopeFallback(unittest.TestCase):
    """A signal too dense to follow within the rows: an oscillator start-up, 1000 cycles."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = np.linspace(0, 1e-6, 200_001)
        cls.t = t
        cls.v = 0.6 + 0.6 * (1 - np.exp(-t / 2e-7)) * np.sin(2 * np.pi * 1e9 * t)
        cls.path = Path(cls.tmp.name) / "osc.tr0"
        fixtures.write_binary(cls.path, "2001", ["TIME", "v(osc"], [1, 1], [([], np.c_[t, cls.v])])
        from spice_result_parser import mcp_server
        cls.m = mcp_server

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def table(self, text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.split()[:1] in (["TIME"], ["TIME_from"]))
        return lines[start].split(), np.array([[float(c) for c in line.split()] for line in lines[start + 1:]])

    def test_text_shows_the_envelope_over_the_whole_run(self):
        text = self.m.format_text(read_traces(self.path))
        self.assertIn("200001 points (an envelope of 100 x slices: too dense to follow every trace within 1% in "
                      "500 rows; narrow xrange to see the waveform)", text.splitlines()[0])
        self.assertIn("-- envelope: each row is a slice with every trace's min and max --", text)
        heading, rows = self.table(text)
        self.assertEqual(heading, ["TIME_from", "TIME_to", "min(v_osc)", "max(v_osc)"])
        self.assertEqual(len(rows), 100)
        self.assertEqual((rows[0, 0], rows[-1, 1]), (0.0, 1e-6))   # the whole run, not one corner of it
        swing = rows[:, 3] - rows[:, 2]
        self.assertLess(swing[0], 0.2)                             # the start-up grows ...
        self.assertGreater(swing[-1], 1.1)                         # ... to the full swing
        self.assertTrue(np.all(np.diff(swing) > -1e-3))
        self.assertLess(len(text), 6000)

    def test_zooming_in_shows_the_waveform(self):
        text = self.m.format_text(read_traces(self.path, xrange=(5e-7, 5.03e-7)))
        self.assertIn(" shown, each trace within 1% of its range)", text.splitlines()[0])
        heading, rows = self.table(text)
        self.assertEqual(heading, ["TIME", "v_osc"])
        self.assertGreater(len(rows), 3 * 8)                       # three cycles, each drawn

    def test_a_sweep_can_mix_followed_and_enveloped_tables(self):
        t = self.t[:20_001]
        step = np.where(t < 5e-8, 0.0, 1.0)
        sweeps = [([1.0], np.c_[t, step]), ([2.0], np.c_[t, self.v[:20_001]])]
        path = Path(self.tmp.name) / "mixed.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "case"], [1, 1], sweeps)
        text = self.m.format_text(read_traces(path))
        self.assertIn("1 of 2 tables as envelopes: too dense to follow every trace within 1% in 250 rows per "
                      "table", text.splitlines()[0])
        self.assertIn("-- sweep 0: case=1 --", text)
        self.assertIn("-- sweep 1: case=2; envelope: each row is a slice with every trace's min and max --", text)

    def test_summary_gives_the_envelope(self):
        s = api.extract(self.path, output="summary", tolerance=0.01, downsample=100)
        entry = s["stats"]["v_osc"][0]
        self.assertNotIn("x", entry)
        env = entry["envelope"]
        self.assertEqual(len(env["from"]), 50)                     # 100 points' worth: 50 slices
        self.assertEqual((env["from"][0], env["to"][-1]), (0.0, 1e-6))
        self.assertEqual((min(env["min"]), max(env["max"])), (entry["min"], entry["max"]))
        json.dumps(s, allow_nan=False)


class TestBugHuntRegressions(unittest.TestCase):
    """Edge cases found by a bug hunt over the reduction and the outputs built on it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        from spice_result_parser import mcp_server
        self.m = mcp_server

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_flat_nested_sweep_keeps_every_inner_curve(self):
        # x restarts 8 times; two kept samples at the same x must not hide the curves between them
        inner, outer = np.linspace(0, 1.8, 16), np.linspace(0.5, 3, 8)
        x = np.tile(inner, outer.size)
        y = np.concatenate([np.sin(inner * o * 3) for o in outer])
        keep, reached = reduce.breakpoints(x, [y], [reduce.finite_span([y])], 0.01)
        self.assertLessEqual(reached, 0.01)
        self.assertEqual(sorted(set((keep // inner.size).tolist())), list(range(outer.size)))
        for a, b in zip(keep[:-1], keep[1:]):                      # every gap: one run, or a straight index line
            inside = np.arange(a + 1, b)
            if inside.size and x[a] == x[b] and np.ptp(x[a:b + 1]) > 0:
                line = y[a] + (y[b] - y[a]) * (inside - a) / (b - a)
                self.assertLessEqual(np.abs(y[inside] - line).max(), 0.01 * np.ptp(y) + 1e-12)

    def test_tolerance_zero_ignores_rounding_whatever_the_offset(self):
        t = np.linspace(0, 1e-6, 401)
        for offset in (0.0, 1.0, 1e6):
            v = offset + 1e-3 * t / 1e-6
            keep, _ = reduce.breakpoints(t, [v], [reduce.finite_span([v])], 0.0, eps=np.finfo(np.float64).eps)
            self.assertEqual(keep.tolist(), [0, 400], offset)
        v32 = (1.0 + 1e-3 * t / 1e-6).astype(np.float32)
        keep, _ = reduce.breakpoints(t, [v32], [reduce.finite_span([v32])], 0.0, eps=np.finfo(np.float32).eps)
        self.assertLess(keep.size, 10)                              # float32 steps are not corners

    def test_a_repeated_name_is_still_a_magnitude(self):
        h = read_traces(HERE / "test_9601.ac0", ["v(vo)"]).header
        self.assertTrue(reduce.in_db(h, "v_a_b_Mag#3"))
        self.assertFalse(reduce.in_db(h, "v_a_b_Phase#4"))

    def test_summary_envelope_bounds_are_exact_and_within_budget(self):
        t = 1e-3 + np.arange(20_001) * 5e-12                     # a 5 GHz clock, 1 ms into the run
        v = 0.6 + 0.6 * np.sign(np.sin(2 * np.pi * 5e9 * t + 0.1))
        path = self.dir / "clk.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(clk"], [1, 1], [([], np.c_[t, v])])
        env = api.extract(path, output="summary", tolerance=0.01)["stats"]["v_clk"][0]["envelope"]
        self.assertEqual(len(set(env["from"])), len(env["from"]))  # no two slices print alike
        self.assertTrue(all(a < b for a, b in zip(env["from"], env["to"])))
        one = api.extract(path, output="summary", downsample=2)["stats"]["v_clk"][0]["envelope"]
        self.assertEqual(len(one["from"]), 1)                       # 2 points' worth: 1 slice

    def test_a_downward_sweep_is_one_run(self):
        x = np.linspace(1.8, 0.0, 25)
        path = self.dir / "desc.sw0"
        fixtures.write_binary(path, "2001", ["vin", "v(out"], [3, 1], [([], np.c_[x, 1.8 - x])])
        text = self.m.format_text(read_traces(path))
        self.assertNotIn("run ", text)
        self.assertNotIn("not shown", text)
        self.assertIn("25 points (2 shown", text.splitlines()[0])
        # a nested sweep whose inner variable runs downwards: three runs, not thirteen
        inner = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        x = np.tile(inner, 3)
        raw = self.dir / "nested_desc.raw"
        fixtures.write_nutmeg(raw, [("DC transfer characteristic", ["v-sweep", "v(out)"],
                                     np.c_[x, np.concatenate([inner * (1 + o) for o in (0, .5, 1)])], False)])
        heads = [l for l in self.m.format_text(read_traces(raw)).splitlines() if l.startswith("-- run")]
        self.assertEqual(heads, [f"-- run {k} of 3 (x restarts) --" for k in (1, 2, 3)])

    def test_close_x_values_print_apart(self):
        # the repo's 2001 sample has samples 25 ns apart at t = 100 us: 6 digits printed them alike
        text = self.m.format_text(read_traces(HERE / "test_2001.tr0", ["v(vs"]), tolerance=0)
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.split()[:1] in (["TIME"], ["TIME_from"]))
        xs = [l.split()[0] for l in lines[start + 1:]]
        self.assertEqual(len(xs), len(set(xs)), "two rows print the same x")
        t = 1e-3 + np.arange(2001) * 5e-11                         # a 100 ns window at 1 ms
        path = self.dir / "late.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a"], [1, 1], [([], np.c_[t, np.sin(t * 1e8)])])
        head = self.m.format_text(read_traces(path)).splitlines()[0]
        self.assertIn("x: TIME in [0.001, 0.0010001]", head)

    def test_a_small_ripple_on_a_large_value_shows(self):
        t = np.linspace(0, 1e-6, 2001)
        v = 1.2 + 1e-6 * np.sin(2 * np.pi * 3e6 * t)               # 1 uV ripple on a 1.2 V rail
        path = self.dir / "rail.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(vdd"], [1, 1], [([], np.c_[t, v])])
        text = self.m.format_text(read_traces(path))
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.split()[:1] == ["TIME"])
        ys = {l.split()[1] for l in lines[start + 1:]}
        self.assertGreater(len(ys), 5)                              # not "1.2" on every row
        stats = next(l.split() for l in lines if l.split()[:1] == ["v_vdd"])
        self.assertNotEqual(stats[1], stats[2])                     # min and max print apart

    def test_png_draws_every_point_whatever_downsample_says(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        t = np.linspace(0, 1e-6, 100_001)
        v = np.zeros_like(t)
        v[33_350] = 1.0
        path = self.dir / "glitch.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(out"], [1, 1], [([], np.c_[t, v])])
        out = self.m.extract(str(path), ["v(out"], output="png", downsample=1000, dest=str(self.dir / "g.png"))
        self.assertEqual((out["points"], out["ranges"]["v_out"]), (100_001, [0.0, 1.0]))
        self.assertTrue(api.extract(path, ["v(out"], output="png", downsample=1000, dest=str(self.dir / "h.png")))

    def test_dest_is_checked_before_the_read(self):
        from unittest import mock
        src = str(HERE / "test_9601.tr0")
        with mock.patch.object(api, "read_traces", side_effect=AssertionError("read first")):
            for dest, why in (("", "dest is empty"), (str(self.dir / "nodir") + "/", "is a directory"),
                              (src + "/.", "is a directory"), (str(self.dir / "no" / "o.csv"), "does not exist")):
                with self.assertRaisesRegex(ValueError, why):
                    api.extract(src, output="csv", dest=dest)

    def test_a_file_without_data_says_so(self):
        path = self.dir / "started.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a"], [1, 1], [([], np.empty((0, 2)))], final_sentinel=False)
        self.assertEqual(self.m.format_text(read_traces(path)).splitlines()[-1], "the file holds no data point")


class TestMcpTolerance(unittest.TestCase):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from spice_result_parser import mcp_server

        self.m = mcp_server

    def test_schema(self):
        tools = {t.name: t for t in asyncio.run(self.m.server.list_tools())}
        schema = getattr(tools["extract"], "inputSchema", None) or tools["extract"].input_schema
        self.assertIn("tolerance", schema["properties"])
        self.assertIn("fraction of each trace's range", schema["properties"]["tolerance"]["description"])
        self.assertLess(len(tools["extract"].description), 1400)

    def test_errors_reach_the_client(self):
        path = str(HERE / "test_9601.tr0")
        with self.assertRaisesRegex(self.m.ToolError, "tolerance applies to text and summary only"):
            self.m.extract(path, output="csv", tolerance=0.01)
        with self.assertRaisesRegex(self.m.ToolError, "tolerance must be between 0 and 1"):
            self.m.extract(path, tolerance=5)

    def test_text_and_summary(self):
        path = str(HERE / "test_9601.tr0")
        self.assertIn("within 5% of its range", self.m.extract(path, ["v(vo"], tolerance=0.05))
        out = self.m.extract(path, ["v(vo"], output="summary", tolerance=0.05)
        self.assertLessEqual(out["stats"]["v_vo"][0]["max_error"], 0.05)


if __name__ == "__main__":
    unittest.main()
