"""Tests for the Nutmeg rawfile reader (ngspice / SPICE3 .raw), binary and ASCII.

The real fixtures nutmeg_rc_bin.raw and nutmeg_rc_ascii.raw come from ngspice-47
running nutmeg_rc.sp; each holds three plots (tran, ac, dc) of four variables.
"""
import asyncio
import csv
import json
import re
import sys
import tempfile
import tracemalloc
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402

from spice_result_parser.reader import read_header, read_traces  # noqa: E402
from spice_result_parser import api  # noqa: E402
from spice_result_parser import nutmeg  # noqa: E402

BIN = HERE / "nutmeg_rc_bin.raw"
ASCII = HERE / "nutmeg_rc_ascii.raw"
REAL_FILES = (BIN, ASCII)
PLOT_NAMES = ["Transient Analysis", "AC Analysis", "DC transfer characteristic"]


def synthetic(directory, binary=True, name="synth.raw"):
    """A three-plot rawfile: tran (real), ac (complex), dc (real). Returns (path, datas)."""
    rng = np.random.default_rng(11)
    tran = rng.uniform(-1, 1, (37, 3))
    tran[:, 0] = np.arange(37) * 1e-9
    ac = rng.uniform(-2, 2, (21, 3)) + 1j * rng.uniform(-2, 2, (21, 3))
    ac[:, 0] = np.logspace(0, 6, 21)                    # frequency: real part only
    dc = rng.uniform(-1, 1, (9, 4))
    dc[:, 0] = np.linspace(0, 2, 9)
    path = Path(directory) / name
    written = fixtures.write_nutmeg(path, [
        ("Transient Analysis", ["time", "v(in)", "v(out)"], tran, False),
        ("AC Analysis", ["frequency", "v(in)", "v(out)"], ac, True),
        ("DC transfer characteristic", ["v(v-sweep)", "v(in)", "v(out)", "i(v1)"], dc, False),
    ], binary=binary)
    return path, written


class TestDetection(unittest.TestCase):
    def test_is_nutmeg(self):
        for path in REAL_FILES:
            self.assertTrue(nutmeg.is_nutmeg(path), path)
        for path in ("test_9601.tr0", "test_2001.tr0"):
            self.assertFalse(nutmeg.is_nutmeg(HERE / path), path)

    def test_leading_bom_and_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = synthetic(tmp)
            body = path.read_bytes()
            path.write_bytes(b"\xef\xbb\xbf" + body)
            self.assertTrue(nutmeg.is_nutmeg(path))
            self.assertEqual(read_header(path).plot_names, PLOT_NAMES)


class TestRealFileHeaders(unittest.TestCase):
    """Item 1: the header of each plot of the two ngspice files."""

    def test_transient(self):
        for path in REAL_FILES:
            h = read_header(path)
            self.assertEqual(h.version, "nutmeg")
            self.assertEqual(h.analysis, "tr")
            self.assertEqual(h.is_binary, path == BIN)
            self.assertEqual(h.ncols, 4)
            self.assertEqual(h.x_name, "time")
            self.assertEqual(h.names, ["time", "v_in", "v_out", "i_v1"])
            self.assertEqual(h.raw_names, ["time", "v(in)", "v(out)", "i(v1)"])
            self.assertEqual(h.col_raw_names, h.raw_names)
            self.assertEqual(h.plot_names, PLOT_NAMES)
            self.assertEqual(h.plot, 0)
            self.assertEqual(h.nsweepparam, 0)
            self.assertEqual(h.sweep_count_hint, 0)
            self.assertEqual(h.type_codes, [])
            self.assertEqual(h.sweep_params, [])
            self.assertEqual(h.dtype, np.dtype("<f8"))

    def test_ac(self):
        for path in REAL_FILES:
            h = read_header(path, plot=1)
            self.assertEqual(h.version, "nutmeg")
            self.assertEqual(h.analysis, "ac")
            self.assertEqual(h.ncols, 7)
            self.assertEqual(h.x_name, "frequency")
            self.assertEqual(h.names, ["frequency", "v_in_Mag", "v_in_Phase", "v_out_Mag",
                                       "v_out_Phase", "i_v1_Mag", "i_v1_Phase"])
            self.assertEqual(h.raw_names, ["frequency", "v(in)", "v(out)", "i(v1)"])
            self.assertEqual(h.col_raw_names, ["frequency", "v(in)", "v(in)", "v(out)",
                                               "v(out)", "i(v1)", "i(v1)"])
            self.assertEqual(h.plot_names, PLOT_NAMES)
            self.assertEqual(h.plot, 1)

    def test_dc(self):
        for path in REAL_FILES:
            h = read_header(path, plot=2)
            self.assertEqual(h.analysis, "sw")
            self.assertEqual(h.ncols, 4)
            self.assertEqual(h.x_name, "v_v-sweep")
            self.assertEqual(h.names, ["v_v-sweep", "v_in", "v_out", "i_v1"])
            self.assertEqual(h.raw_names, ["v(v-sweep)", "v(in)", "v(out)", "i(v1)"])
            self.assertEqual(h.plot_names, PLOT_NAMES)
            self.assertEqual(h.plot, 2)

    def test_header_reads_no_data(self):
        # Reading the last plot's header walks the file; the point counts prove the
        # headers, not the data sections, decide the layout.
        self.assertEqual([read_traces(BIN, plot=p).data[read_header(BIN, plot=p).x_name][0].size
                          for p in range(3)], [80, 46, 5])


class TestRealFileData(unittest.TestCase):
    def test_binary_and_ascii_agree(self):
        """Item 2: both flavours of the same run give the same arrays for every plot."""
        for p in range(3):
            b = read_traces(BIN, plot=p)
            a = read_traces(ASCII, plot=p)
            self.assertEqual(a.selected, b.selected)
            self.assertFalse(a.truncated or b.truncated)
            for name in b.selected:
                np.testing.assert_allclose(a.data[name][0], b.data[name][0], rtol=1e-12, atol=1e-18)

    def test_ascii_chunk_size_does_not_change_results(self):
        # The real file is far shorter than one chunk; a tiny chunk exercises the boundaries,
        # including plots 1 and 2, which are reached by skipping earlier data chunk by chunk.
        ref = {p: read_traces(ASCII, plot=p) for p in range(3)}
        original = nutmeg.ASCII_CHUNK_BYTES
        nutmeg.ASCII_CHUNK_BYTES = 64
        self.addCleanup(setattr, nutmeg, "ASCII_CHUNK_BYTES", original)
        self.assertEqual(read_header(ASCII, plot=2).plot_names, PLOT_NAMES)
        for p in range(3):
            ts = read_traces(ASCII, plot=p)
            for name in ts.selected:
                np.testing.assert_array_equal(ts.data[name][0], ref[p].data[name][0])

    def test_ac_column_zero_is_real_frequency(self):
        """Item 3 (real file): column 0 of a complex plot is its real part only."""
        for path in REAL_FILES:
            ts = read_traces(path, plot=1)
            freq = ts.data["frequency"][0]
            self.assertEqual(freq.dtype, np.dtype("float64"))
            self.assertEqual(freq.size, 46)
            self.assertAlmostEqual(freq[0], 1.0)
            np.testing.assert_allclose(freq, np.logspace(0, 9, 46), rtol=1e-12)
            # v(in) is the 1 V ac source: magnitude 1, phase 0.
            np.testing.assert_allclose(ts.data["v_in_Mag"][0], np.ones(46), rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_in_Phase"][0], np.zeros(46), atol=1e-12)

    def test_dc_sweep_column(self):
        """Item 4: the dc transfer plot sweeps V1 from 0 to 1 in steps of 0.25."""
        for path in REAL_FILES:
            ts = read_traces(path, plot=2)
            np.testing.assert_allclose(ts.data["v_v-sweep"][0], [0.0, 0.25, 0.5, 0.75, 1.0])
            np.testing.assert_allclose(ts.data["v_out"][0], [0.0, 0.25, 0.5, 0.75, 1.0])
            self.assertEqual(ts.sweep_values, [[]])
            self.assertEqual(ts.sweep_indices, [0])


class TestComplexExpansion(unittest.TestCase):
    """Item 3: Mag/Phase are abs and angle(deg) of the complex values."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mag_and_phase_match_the_synthetic_values(self):
        for binary in (True, False):
            path, written = synthetic(self.dir, binary, f"c{int(binary)}.raw")
            ac = written[1]
            ts = read_traces(path, plot=1)
            self.assertEqual(ts.selected, ["frequency", "v_in_Mag", "v_in_Phase",
                                           "v_out_Mag", "v_out_Phase"])
            np.testing.assert_allclose(ts.data["frequency"][0], ac[:, 0].real, rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_out_Mag"][0], np.abs(ac[:, 2]), rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_out_Phase"][0], np.angle(ac[:, 2], deg=True),
                                       rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_in_Mag"][0], np.abs(ac[:, 1]), rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_in_Phase"][0], np.angle(ac[:, 1], deg=True),
                                       rtol=1e-12)

    def test_ac_format_realimag_and_db(self):
        for binary in (True, False):
            path, written = synthetic(self.dir, binary, f"f{int(binary)}.raw")
            ac = written[1]
            ts = read_traces(path, ["v(out)"], plot=1, ac_format="realimag")
            self.assertEqual(ts.selected, ["frequency", "v_out_Re", "v_out_Im"])
            np.testing.assert_allclose(ts.data["v_out_Re"][0], ac[:, 2].real, rtol=1e-12)
            np.testing.assert_allclose(ts.data["v_out_Im"][0], ac[:, 2].imag, rtol=1e-12)
            ts = read_traces(path, ["v_out_dB"], plot=1, ac_format="db")
            self.assertEqual(ts.selected, ["frequency", "v_out_dB"])
            np.testing.assert_allclose(ts.data["v_out_dB"][0], 20 * np.log10(np.abs(ac[:, 2])), rtol=1e-12)
            self.assertEqual(read_header(path, plot=1, ac_format="db").names[1:3], ["v_in_dB", "v_in_Phase"])

    def test_unknown_ac_format_rejected(self):
        with self.assertRaisesRegex(ValueError, "ac_format must be one of"):
            read_traces(BIN, plot=1, ac_format="polar")


class TestSelection(unittest.TestCase):
    """Item 5."""

    def test_raw_name_with_paren(self):
        ts = read_traces(BIN, ["v(out)"])
        self.assertEqual(ts.selected, ["time", "v_out"])

    def test_sanitized_name(self):
        ts = read_traces(BIN, ["v_out"])
        self.assertEqual(ts.selected, ["time", "v_out"])
        np.testing.assert_array_equal(ts.data["v_out"][0], read_traces(BIN, ["v(out)"]).data["v_out"][0])

    def test_glob(self):
        ts = read_traces(ASCII, ["v_*"])
        self.assertEqual(ts.selected, ["time", "v_in", "v_out"])

    def test_raw_name_selects_mag_and_phase_on_an_ac_plot(self):
        ts = read_traces(BIN, ["v(out)"], plot=1)
        self.assertEqual(ts.selected, ["frequency", "v_out_Mag", "v_out_Phase"])

    def test_unknown_name_lists_available(self):
        with self.assertRaisesRegex(ValueError, r"unknown trace 'v\(nope\)'.*time, v_in, v_out, i_v1"):
            read_traces(BIN, ["v(nope)"])

    def test_hspice_raw_names_still_match_without_the_paren(self):
        ts = read_traces(HERE / "test_9601.tr0", ["v(vo"])
        self.assertEqual(ts.selected, ["TIME", "v_vo"])


class TestPlotIndex(unittest.TestCase):
    """Item 6."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_out_of_range(self):
        for path in REAL_FILES:
            with self.assertRaisesRegex(ValueError, "plot 3 out of range; the file has 3 plot"):
                read_header(path, plot=3)
            with self.assertRaisesRegex(ValueError, "plot 7 out of range; the file has 3 plot"):
                read_traces(path, plot=7)

    def test_third_plot_of_the_synthetic_file(self):
        for binary in (True, False):
            path, written = synthetic(self.dir, binary, f"p{int(binary)}.raw")
            h = read_header(path, plot=2)
            self.assertEqual(h.analysis, "sw")
            self.assertEqual(h.plot, 2)
            self.assertEqual(h.plot_names, PLOT_NAMES)
            self.assertEqual(h.names, ["v_v-sweep", "v_in", "v_out", "i_v1"])
            ts = read_traces(path, plot=2)
            for col, name in enumerate(ts.selected):
                np.testing.assert_allclose(ts.data[name][0], written[2][:, col].real, rtol=1e-12)

    def test_plot_on_an_hspice_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "applies to Nutmeg rawfiles only"):
            read_header(HERE / "test_9601.tr0", plot=1)
        with self.assertRaisesRegex(ValueError, "applies to Nutmeg rawfiles only"):
            read_traces(HERE / "test_9601.tr0", plot=1)


class TestDimensions(unittest.TestCase):
    """Item 7."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = np.arange(15 * 3, dtype=np.float64).reshape(15, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, binary, dimensions=(3, 5)):
        path = self.dir / f"dim{int(binary)}.raw"
        fixtures.write_nutmeg(path, [("DC transfer characteristic", ["v(sweep)", "v(in)", "v(out)"],
                                      self.data, False, list(dimensions))], binary=binary)
        return path

    def test_three_sweeps_of_five_points(self):
        for binary in (True, False):
            path = self.write(binary)
            self.assertEqual(read_header(path).sweep_count_hint, 3)
            ts = read_traces(path)
            self.assertEqual(ts.sweep_indices, [0, 1, 2])
            self.assertEqual(ts.sweep_values, [[], [], []])
            for i in range(3):
                np.testing.assert_array_equal(ts.data["v_out"][i], self.data[5 * i:5 * i + 5, 2])

    def test_sweeps_filter(self):
        for binary in (True, False):
            ts = read_traces(self.write(binary), ["v(out)"], sweeps=[1])
            self.assertEqual(ts.sweep_indices, [1])
            self.assertEqual(len(ts.data["v_out"]), 1)
            np.testing.assert_array_equal(ts.data["v_out"][0], self.data[5:10, 2])
            self.assertIsNone(ts.data["v_out"][0].base)   # the whole-file buffer is released

    def test_unusable_dimensions_are_ignored(self):
        path = self.write(True, dimensions=(4, 5))        # 4 * 5 != 15
        self.assertEqual(read_header(path).sweep_count_hint, 0)
        self.assertEqual(read_traces(path).sweep_indices, [0])


class TestTruncation(unittest.TestCase):
    """Items 8 and 9."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = np.arange(10 * 4, dtype=np.float64).reshape(10, 4)
        self.names = ["time", "v(a)", "v(b)", "i(c)"]

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, binary):
        path = self.dir / f"trunc{int(binary)}.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", self.names, self.data, False)],
                              binary=binary)
        return path

    def test_truncated_binary(self):
        path = self.write(True)
        raw = path.read_bytes()
        path.write_bytes(raw[:len(raw) - (4 * 8 + 3)])    # one point gone, three bytes of the next
        with self.assertWarnsRegex(RuntimeWarning, "file ends after 8 of 10 points"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.data["time"][0].size, 8)
        np.testing.assert_array_equal(ts.data["v_b"][0], self.data[:8, 2])

    def test_truncated_ascii(self):
        path = self.write(False)
        raw = path.read_bytes()
        cut = raw.index(b"\n 8\t") + 1                    # start of point 8
        path.write_bytes(raw[:cut] + raw[cut:].split(b"\n")[0] + b"\n")   # one value of point 8
        with self.assertWarnsRegex(RuntimeWarning, "file ends after 8 of 10 points"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.data["time"][0].size, 8)
        np.testing.assert_array_equal(ts.data["i_c"][0], self.data[:8, 3])

    def test_complete_file_is_not_truncated(self):
        for binary in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ts = read_traces(self.write(binary))
            self.assertFalse(ts.truncated)
            self.assertEqual(ts.data["time"][0].size, 10)

    def test_final_newline_is_optional_when_the_count_is_known(self):
        # A writer (or an editor) that leaves the last newline off still yields every point.
        for batch_layout in (False, True):
            path = self.dir / f"nonl{int(batch_layout)}.raw"
            fixtures.write_nutmeg(path, [("Transient Analysis", self.names, self.data, False)],
                                  binary=False, batch_layout=batch_layout)
            path.write_bytes(path.read_bytes().rstrip(b"\n"))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ts = read_traces(path)
            self.assertFalse(ts.truncated)
            self.assertEqual(ts.data["time"][0].size, 10)
            for col, name in enumerate(ts.selected):
                np.testing.assert_array_equal(ts.data[name][0], self.data[:, col])

    def test_file_cut_inside_a_header(self):
        for binary in (True, False):
            path = self.write(binary)
            raw = path.read_bytes()
            path.write_bytes(raw[:raw.index(b"No. Points") + 6])
            with self.assertRaisesRegex(ValueError, "file ends inside the header of plot 0"):
                read_header(path)
            with self.assertRaisesRegex(ValueError, "file ends inside the header of plot 0"):
                read_traces(path)

    def test_second_plot_cut_inside_its_header(self):
        path, _ = synthetic(self.dir, True, "cut.raw")
        raw = path.read_bytes()
        second = raw.index(b"Title:", 10)
        path.write_bytes(raw[:second + 40])
        with self.assertRaisesRegex(ValueError, "file ends inside the header of plot 1"):
            read_header(path)


class TestUnwrittenPointCount(unittest.TestCase):
    """A file the simulator is still writing: ngspice leaves 'No. Points: 0' in the header
    and patches it when the run ends."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = np.arange(12 * 4, dtype=np.float64).reshape(12, 4) * 0.5
        self.names = ["time", "v(a)", "v(b)", "i(c)"]

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, binary, **kwargs):
        path = self.dir / f"live{int(binary)}.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", self.names, self.data, False)],
                              binary=binary, unknown_points=True, **kwargs)
        return path

    def test_empty_plot_before_a_lowercase_title_is_complete(self):
        # 'No. Points: 0' followed by another plot means an empty plot, not a live run;
        # the next header's key is matched without regard to case, as in is_nutmeg.
        path = self.dir / "empty_then_more.raw"
        path.write_bytes(
            b"Title: t\nPlotname: Operating Point\nFlags: real\nNo. Variables: 1\nNo. Points: 0\n"
            b"Variables:\n\t0\tv(a)\tvoltage\nValues:\n"
            b"title: t\nPlotname: Transient Analysis\nFlags: real\nNo. Variables: 2\nNo. Points: 1\n"
            b"Variables:\n\t0\ttime\ttime\n\t1\tv(a)\tvoltage\nValues:\n 0\t0.0\n\t1.5\n")
        self.assertEqual(read_header(path).plot_names, ["Operating Point", "Transient Analysis"])
        ts = read_traces(path, plot=1)
        self.assertFalse(ts.truncated)
        np.testing.assert_array_equal(ts.data["v_a"][0], [1.5])

    def test_header_still_readable(self):
        for binary in (True, False):
            h = read_header(self.write(binary))
            self.assertEqual(h.names, ["time", "v_a", "v_b", "i_c"])
            self.assertEqual(h.plot_names, ["Transient Analysis"])
            self.assertEqual(h.sweep_count_hint, 0)
            self.assertEqual(api.list_traces(self.write(binary))["plots"], ["Transient Analysis"])

    def test_binary_whole_points(self):
        path = self.write(True)
        with self.assertWarnsRegex(RuntimeWarning, "'No. Points' has not been written yet"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.sweep_indices, [0])
        for col, name in enumerate(ts.selected):
            np.testing.assert_array_equal(ts.data[name][0], self.data[:, col])

    def test_binary_cut_mid_point(self):
        path = self.write(True)
        raw = path.read_bytes()
        path.write_bytes(raw[:len(raw) - 11])            # 11 bytes into the last point
        with self.assertWarnsRegex(RuntimeWarning, "Read 11 complete points"):
            ts = read_traces(path, ["v(b)"])
        self.assertTrue(ts.truncated)
        np.testing.assert_array_equal(ts.data["v_b"][0], self.data[:11, 2])

    def test_ascii_clean_cut(self):
        path = self.write(False)
        raw = path.read_bytes()
        keep = raw.index(b" 9\t")                        # everything before point 9
        path.write_bytes(raw[:keep])
        with self.assertWarnsRegex(RuntimeWarning, "Read 9 complete points"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.data["time"][0].size, 9)
        np.testing.assert_array_equal(ts.data["i_c"][0], self.data[:9, 3])

    def test_ascii_cut_mid_token(self):
        path = self.write(False)
        raw = path.read_bytes()
        cut = raw.index(b" 9\t")
        path.write_bytes(raw[:cut] + b" 9\t4.5000000\t1.8293")   # no trailing newline
        with self.assertWarnsRegex(RuntimeWarning, "Read 9 complete points"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.data["time"][0].size, 9)     # the half-written point is dropped
        np.testing.assert_array_equal(ts.data["time"][0], self.data[:9, 0])

    def test_ascii_buffers_are_released_after_the_estimate(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ts = read_traces(self.write(False), ["v(b)"])
        self.assertEqual(ts.data["v_b"][0].size, 12)
        self.assertIsNone(ts.data["v_b"][0].base)        # the estimated buffer is not pinned

    def test_ascii_estimate_grows_when_too_small(self):
        # A hand-written file with very short values beats the bytes-per-value estimate.
        path = self.dir / "dense.raw"
        rows = ["Title: dense", "Plotname: Transient Analysis", "Flags: real",
                "No. Variables: 2", "No. Points: 0", "Variables:",
                "\t0\ttime\ttime", "\t1\tv(a)\tvoltage", "Values:"]
        body = "".join(f"{i}\t{i}\t{i + 1}\n" for i in range(500))
        path.write_bytes(("\n".join(rows) + "\n" + body).encode("utf-8"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ts = read_traces(path)
        self.assertEqual(ts.data["time"][0].size, 500)
        np.testing.assert_array_equal(ts.data["v_a"][0], np.arange(1, 501, dtype=np.float64))

    def test_genuinely_empty_plot_keeps_the_following_plot(self):
        for binary in (True, False):
            path = self.dir / f"empty{int(binary)}.raw"
            fixtures.write_nutmeg(path, [
                ("Operating Point", ["v(in)", "v(out)"], np.empty((0, 2)), False),
                ("Transient Analysis", self.names, self.data, False),
            ], binary=binary)
            h = read_header(path)
            self.assertEqual(h.plot_names, ["Operating Point", "Transient Analysis"])
            self.assertEqual(h.analysis, "op")
            with warnings.catch_warnings():
                warnings.simplefilter("error")           # an empty plot is not "still being written"
                empty = read_traces(path)
            self.assertFalse(empty.truncated)
            self.assertEqual(empty.data["v_in"][0].size, 0)
            ts = read_traces(path, plot=1)
            self.assertFalse(ts.truncated)
            np.testing.assert_array_equal(ts.data["v_b"][0], self.data[:, 2])

    def test_trailing_empty_plot(self):
        path = self.dir / "trailing.raw"
        fixtures.write_nutmeg(path, [
            ("Transient Analysis", self.names, self.data, False),
            ("Operating Point", ["v(in)", "v(out)"], np.empty((0, 2)), False),
        ], binary=True)
        self.assertEqual(read_header(path).plot_names, ["Transient Analysis", "Operating Point"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ts = read_traces(path, plot=1)
        self.assertEqual(ts.data["v_in"][0].size, 0)


class TestBatchAsciiLayout(unittest.TestCase):
    """ngspice batch mode: '<index>\\t\\t<value>' and no blank line between points."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = np.arange(9 * 3, dtype=np.float64).reshape(9, 3) * 0.25
        self.names = ["time", "v(in)", "v(out)"]

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **kwargs):
        path = self.dir / f"b{len(kwargs)}{kwargs.get('batch_layout', 0)}.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", self.names, self.data, False)],
                              binary=False, **kwargs)
        return path

    def test_parses_like_the_write_layout(self):
        batch = read_traces(self.write(batch_layout=True))
        plain = read_traces(self.write())
        self.assertEqual(batch.selected, plain.selected)
        self.assertFalse(batch.truncated)
        for col, name in enumerate(batch.selected):
            np.testing.assert_array_equal(batch.data[name][0], self.data[:, col])
            np.testing.assert_array_equal(batch.data[name][0], plain.data[name][0])

    def test_batch_layout_with_an_unwritten_count(self):
        path = self.write(batch_layout=True, unknown_points=True)
        self.assertIn(b"0\t\t", path.read_bytes())
        self.assertNotIn(b"\n\n", path.read_bytes())
        with self.assertWarnsRegex(RuntimeWarning, "Read 9 complete points"):
            ts = read_traces(path, ["v(out)"])
        self.assertTrue(ts.truncated)
        np.testing.assert_array_equal(ts.data["v_out"][0], self.data[:, 2])

    def test_known_count_cut_mid_token(self):
        path = self.write(batch_layout=True)
        raw = path.read_bytes()
        cut = raw.index(b"\n7\t\t") + 1
        path.write_bytes(raw[:cut] + b"7\t\t1.7500\t2.00")     # no trailing newline
        with self.assertWarnsRegex(RuntimeWarning, "file ends after 7 of 9 points"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.data["time"][0].size, 7)
        np.testing.assert_array_equal(ts.data["v_out"][0], self.data[:7, 2])


class TestAsciiChunking(unittest.TestCase):
    """The data section is read in ASCII_CHUNK_BYTES chunks; plots after a big one are
    reached by counting tokens, so the chunk boundary must not shift anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(23)
        self.first = rng.uniform(-1, 1, (200, 3))           # well over a small chunk
        self.ac = rng.uniform(-2, 2, (40, 3)) + 1j * rng.uniform(-2, 2, (40, 3))
        self.ac[:, 0] = np.logspace(0, 6, 40)
        self.last = rng.uniform(-1, 1, (20, 4))
        self.original = nutmeg.ASCII_CHUNK_BYTES
        self.addCleanup(setattr, nutmeg, "ASCII_CHUNK_BYTES", self.original)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, batch_layout):
        path = self.dir / f"multi{int(batch_layout)}.raw"
        fixtures.write_nutmeg(path, [
            ("Transient Analysis", ["time", "v(in)", "v(out)"], self.first, False),
            ("AC Analysis", ["frequency", "v(in)", "v(out)"], self.ac, True),
            ("DC transfer characteristic", ["v(v-sweep)", "v(a)", "v(b)", "i(c)"], self.last, False),
        ], binary=False, batch_layout=batch_layout)
        return path

    def test_later_plots_are_reached_across_chunk_boundaries(self):
        for batch_layout in (False, True):
            path = self.write(batch_layout)
            self.assertGreater(path.stat().st_size, 10 * 300)
            for chunk in (300, 301, 4096, self.original):
                nutmeg.ASCII_CHUNK_BYTES = chunk
                self.assertEqual(read_header(path, plot=2).plot_names, PLOT_NAMES)
                for p, expected in enumerate((self.first, self.ac, self.last)):
                    ts = read_traces(path, plot=p)
                    self.assertFalse(ts.truncated, (batch_layout, chunk, p))
                    x = ts.data[ts.selected[0]][0]
                    np.testing.assert_allclose(x, expected[:, 0].real, rtol=1e-12)
                    self.assertEqual(x.size, expected.shape[0])
                if not batch_layout:
                    continue
                ts = read_traces(path, ["v(out)"], plot=1)      # complex, across boundaries
                np.testing.assert_allclose(ts.data["v_out_Mag"][0], np.abs(self.ac[:, 2]), rtol=1e-12)
                np.testing.assert_allclose(ts.data["v_out_Phase"][0],
                                           np.angle(self.ac[:, 2], deg=True), rtol=1e-12)

    def test_junk_in_the_data_names_the_file_and_plot(self):
        path = self.write(False)
        raw = path.read_bytes()
        start = raw.index(b"Values:") + len(b"Values:\n")
        cut = raw.index(b"\n", raw.index(b"\n", start) + 1)  # a value line inside plot 0
        path.write_bytes(raw[:cut] + b"\n\tnot-a-number" + raw[cut:])
        for chunk in (300, self.original):
            nutmeg.ASCII_CHUNK_BYTES = chunk
            with self.assertRaisesRegex(ValueError, r"plot 0: data section holds a non-numeric"):
                read_traces(path)
            with self.assertRaisesRegex(ValueError, re.escape(str(path))):
                read_traces(path)


class TestMemory(unittest.TestCase):
    """Item 10: one column of a wide file costs its own buffers plus one slab."""

    def test_one_column_of_a_wide_binary_file(self):
        rng = np.random.default_rng(3)
        npoints, nvars = 200_000, 20
        data = rng.random((npoints, nvars))                  # 32 MB
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.raw"
            fixtures.write_nutmeg(path, [("Transient Analysis",
                                          ["time"] + [f"v({i})" for i in range(nvars - 1)],
                                          data, False)], binary=True)
            expected = data[:, 8].copy()
            del data
            tracemalloc.start()
            tracemalloc.reset_peak()
            ts = read_traces(path, ["v(7)"])
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        column_bytes = npoints * 8                           # one float64 column
        self.assertLess(peak, 3 * column_bytes + nutmeg.SLAB_BYTES + 500_000, f"peak {peak} bytes")
        self.assertEqual(ts.selected, ["time", "v_7"])
        np.testing.assert_array_equal(ts.data["v_7"][0], expected)


class TestXrangeWhileReading(unittest.TestCase):
    """read_traces(xrange=...) on a rawfile must equal reading everything and cutting after."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_matches_post_hoc_cut(self, path, lo, hi, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full = read_traces(path, **kw)
            win = read_traces(path, xrange=(lo, hi), **kw)
        self.assertEqual(win.selected, full.selected)
        self.assertEqual(win.sweep_indices, full.sweep_indices)
        self.assertEqual(win.sweep_values, full.sweep_values)
        self.assertEqual(win.truncated, full.truncated)
        x = full.header.x_name
        for i in range(len(full.sweep_values)):
            mask = (full.data[x][i] >= lo) & (full.data[x][i] <= hi)
            for name in full.selected:
                np.testing.assert_array_equal(win.data[name][i], full.data[name][i][mask],
                                              err_msg=f"{path} {name} sweep {i}")
        return full, win

    def middle_window(self, path, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ts = read_traces(path, **kw)
        x = ts.data[ts.header.x_name][0]
        return float(np.percentile(x, 20)), float(np.percentile(x, 70))

    def test_real_files_every_plot(self):
        for path in REAL_FILES:
            for plot in range(3):
                lo, hi = self.middle_window(path, plot=plot)
                _, win = self.assert_matches_post_hoc_cut(path, lo, hi, plot=plot)
                self.assertGreater(win.data[win.selected[0]][0].size, 0)
                self.assertLess(win.data[win.selected[0]][0].size,
                                read_traces(path, plot=plot).data[win.selected[0]][0].size)

    def test_complex_plot_window_uses_the_real_x(self):
        for path in REAL_FILES:
            ts = read_traces(path, ["v(out)"], plot=1, xrange=(1e3, 1e6))
            f = ts.data["frequency"][0]
            self.assertTrue((f >= 1e3).all() and (f <= 1e6).all())
            full = read_traces(path, ["v(out)"], plot=1)
            mask = (full.data["frequency"][0] >= 1e3) & (full.data["frequency"][0] <= 1e6)
            np.testing.assert_array_equal(ts.data["v_out_Mag"][0], full.data["v_out_Mag"][0][mask])
            np.testing.assert_array_equal(ts.data["v_out_Phase"][0], full.data["v_out_Phase"][0][mask])

    def test_synthetic_binary_and_ascii(self):
        for binary in (True, False):
            path, _ = synthetic(self.dir, binary, f"w{int(binary)}.raw")
            for plot in range(3):
                lo, hi = self.middle_window(path, plot=plot)
                self.assert_matches_post_hoc_cut(path, lo, hi, plot=plot)

    def test_dimensions_sweeps(self):
        data = np.arange(15 * 3, dtype=np.float64).reshape(15, 3)
        data[:, 0] = np.tile(np.arange(5, dtype=np.float64), 3)      # x restarts every sweep
        for binary in (True, False):
            path = self.dir / f"dim{int(binary)}.raw"
            fixtures.write_nutmeg(path, [("DC transfer characteristic",
                                          ["v(sweep)", "v(in)", "v(out)"], data, False, [3, 5])],
                                  binary=binary)
            _, win = self.assert_matches_post_hoc_cut(path, 1.0, 3.0)
            self.assertEqual(win.sweep_indices, [0, 1, 2])
            for i in range(3):
                np.testing.assert_array_equal(win.data["v_sweep"][i], [1.0, 2.0, 3.0])
                np.testing.assert_array_equal(win.data["v_out"][i], data[5 * i + 1:5 * i + 4, 2])
            self.assert_matches_post_hoc_cut(path, 1.0, 3.0, sweeps=[1])
            self.assert_matches_post_hoc_cut(path, 1.0, 3.0, sweeps=[0, 2], names=["v(out)"])
            # a window that empties the middle sweep only
            ts = read_traces(path, xrange=(4.0, 4.0))
            self.assertEqual([ts.data["v_sweep"][i].size for i in range(3)], [1, 1, 1])

    def test_unknown_count(self):
        data = np.arange(12 * 4, dtype=np.float64).reshape(12, 4) * 0.5
        for binary in (True, False):
            path = self.dir / f"live{int(binary)}.raw"
            fixtures.write_nutmeg(path, [("Transient Analysis", ["time", "v(a)", "v(b)", "i(c)"],
                                          data, False)], binary=binary, unknown_points=True)
            _, win = self.assert_matches_post_hoc_cut(path, 4.0, 14.0)
            self.assertTrue(win.truncated)
            np.testing.assert_array_equal(win.data["time"][0], [4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

    def test_truncated_file(self):
        data = np.arange(12 * 4, dtype=np.float64).reshape(12, 4) * 0.5
        path = self.dir / "cut.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", ["time", "v(a)", "v(b)", "i(c)"],
                                      data, False)], binary=True)
        raw = path.read_bytes()
        path.write_bytes(raw[:len(raw) - 35])
        _, win = self.assert_matches_post_hoc_cut(path, 4.0, 40.0)
        self.assertTrue(win.truncated)

    def test_window_covering_everything_matches_no_window(self):
        plain = read_traces(BIN, ["v(out)"])
        wide = read_traces(BIN, ["v(out)"], xrange=(-1e30, 1e30))
        self.assertEqual(wide.sweep_indices, plain.sweep_indices)
        for name in plain.selected:
            np.testing.assert_array_equal(wide.data[name][0], plain.data[name][0])

    def test_empty_window(self):
        for path in REAL_FILES:
            ts = read_traces(path, ["v(out)"], xrange=(1e6, 2e6))    # time runs to 20 ns
            self.assertEqual(ts.sweep_indices, [0])
            self.assertEqual(ts.sweep_values, [[]])
            self.assertFalse(ts.truncated)
            for name in ts.selected:
                self.assertEqual(ts.data[name][0].size, 0)

    def test_window_boundaries_are_inclusive(self):
        for path in REAL_FILES:
            x = read_traces(path, plot=2).data["v_v-sweep"][0]
            ts = read_traces(path, ["v(out)"], plot=2, xrange=(float(x[1]), float(x[3])))
            np.testing.assert_array_equal(ts.data["v_v-sweep"][0], x[1:4])

    def test_memory_follows_the_window(self):
        rng = np.random.default_rng(19)
        npoints, nvars = 200_000, 20
        data = rng.random((npoints, nvars))
        data[:, 0] = np.arange(npoints) * 1e-12
        path = self.dir / "wide.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis",
                                      ["time"] + [f"v({i})" for i in range(nvars - 1)],
                                      data, False)], binary=True)
        lo, hi = 0.0, float(data[npoints // 20, 0])                  # 5% of the rows
        del data
        original = nutmeg.SLAB_BYTES
        nutmeg.SLAB_BYTES = 64 << 10                                 # isolate the buffers
        self.addCleanup(setattr, nutmeg, "SLAB_BYTES", original)
        peaks = []
        for xrange in (None, (lo, hi)):
            tracemalloc.start()
            tracemalloc.reset_peak()
            ts = read_traces(path, ["v(7)"], xrange=xrange)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
        self.assertEqual(ts.data["v_7"][0].size, npoints // 20 + 1)
        self.assertLess(peaks[1], 0.25 * peaks[0], f"windowed {peaks[1]} vs full {peaks[0]} bytes")


class TestApi(unittest.TestCase):
    """Items 11 and 13."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_traces_shows_plots(self):
        info = api.list_traces(BIN, plot=1)
        self.assertEqual(info, {
            "path": str(BIN), "format": "nutmeg", "analysis": "ac", "x": "frequency",
            "traces": ["v_in_Mag", "v_in_Phase", "v_out_Mag", "v_out_Phase",
                       "i_v1_Mag", "i_v1_Phase"],
            "sweep_params": [], "sweep_count_hint": 0, "plot": 1, "plots": PLOT_NAMES,
        })
        json.dumps(info)

    def test_hspice_files_report_no_plots(self):
        info = api.list_traces(HERE / "test_9601.tr0")
        self.assertEqual(info["plots"], [])
        self.assertEqual(info["plot"], 0)
        self.assertEqual(info["format"], "9601")

    def test_summary(self):
        s = api.extract(ASCII, ["v(out)"], output="summary", downsample=5, plot=2)
        self.assertEqual(s["format"], "nutmeg")
        self.assertEqual(s["plot"], 2)
        self.assertEqual(s["plots"], PLOT_NAMES)
        self.assertEqual(s["analysis"], "sw")
        self.assertEqual(s["x"], "v_v-sweep")
        self.assertEqual(s["sweeps"], 1)
        self.assertEqual(s["traces"]["v_out"][0]["count"], 5)
        self.assertAlmostEqual(s["traces"]["v_out"][0]["last"], 1.0)
        self.assertEqual(s["traces"]["v_out"][0]["x"], [0.0, 0.25, 0.5, 0.75, 1.0])
        json.dumps(s)

    def test_csv_round_trip(self):
        dest = self.dir / "ac.csv"
        out = api.extract(BIN, ["v(out)"], output="csv", dest=str(dest), plot=1)
        self.assertEqual(out, str(dest))
        with open(dest, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["frequency", "v_out_Mag", "v_out_Phase"])
        self.assertEqual(len(rows), 47)
        ref = api.extract(BIN, ["v(out)"], plot=1)
        np.testing.assert_allclose([float(r[1]) for r in rows[1:]], ref.data["v_out_Mag"][0])

    def test_npz_round_trip(self):
        dest = self.dir / "tran.npz"
        api.extract(ASCII, ["v_*"], output="npz", dest=str(dest))
        with np.load(dest) as z:
            self.assertEqual(sorted(k for k in z.files if not k.startswith("__")),
                             ["time", "v_in", "v_out"])
            np.testing.assert_allclose(z["v_out"], read_traces(BIN).data["v_out"][0],
                                       rtol=1e-12, atol=1e-18)

    def test_extract_forwards_plot_and_xrange(self):
        ts = api.extract(BIN, ["v(out)"], plot=1, xrange=(1e3, 1e6))
        x = ts.data["frequency"][0]
        self.assertTrue((x >= 1e3).all() and (x <= 1e6).all())
        self.assertEqual(x.size, ts.data["v_out_Mag"][0].size)

    def test_png_uses_a_log_axis_for_an_ac_plot(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        ts = api.extract(BIN, ["v(out)"], plot=1)
        fig = api.make_figure(ts)
        self.assertEqual(fig.axes[0].get_xscale(), "log")
        dest = self.dir / "ac.png"
        self.assertEqual(api.write_file(ts, "png", str(dest)), str(dest))
        self.assertTrue(dest.stat().st_size > 0)


class TestFormatText(unittest.TestCase):
    """mcp_server.format_text: the agent-readable rendering; runs without the mcp package."""

    def setUp(self):
        from spice_result_parser import mcp_server

        self.m = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def text(self, path, names=None, points=20, **kw):
        return self.m.format_text(api.extract(path, names, output="arrays", **kw), points)

    def test_importable_without_mcp(self):
        import importlib
        import subprocess

        code = ("import sys; sys.modules['mcp'] = None\n"
                "from spice_result_parser import mcp_server as m\n"
                "assert m.server is None and m.FastMCP is None\n"
                "assert m.format_text is not None and callable(m.extract)\n"
                "try:\n    m.main()\nexcept ImportError as e:\n    assert 'spice_result_parser[mcp]' in str(e)\n"
                "else:\n    raise SystemExit('main() should refuse to start')\n")
        src = Path(importlib.import_module("spice_result_parser").__file__).parent.parent
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(src))

    def test_dc_plot(self):
        t = self.text(BIN, ["v(out)"], plot=2)
        lines = t.splitlines()
        self.assertEqual(lines[0], "dc sweep | x: v_v-sweep | 5 points | plot 2: DC transfer characteristic "
                                   "(file has 3 plots)")
        self.assertNotIn("shown", lines[0])
        self.assertEqual(lines[2].split(), ["trace", "min", "max"])
        self.assertEqual(lines[3].split(), ["v_out", "0", "1"])
        self.assertEqual(lines[5].split(), ["v_v-sweep", "v_out"])
        self.assertEqual(lines[7].split(), ["0.25", "0.25"])
        self.assertEqual(len(lines), 11)
        self.assertEqual(t.count("v_v-sweep"), 2)   # header and column heading, never per trace

    def test_points_keeps_first_and_last(self):
        t = self.text(ASCII, ["v_out"], points=3)
        lines = t.splitlines()
        self.assertTrue(lines[0].startswith("transient | x: time | 80 points (3 shown)"))
        rows = [l.split() for l in lines[-3:]]
        self.assertEqual([r[0] for r in rows], ["0", "9.2e-09", "2e-08"])   # x[0], x[40], x[79]

    def test_ac_plot(self):
        t = self.text(BIN, ["v(out)"], plot=1, points=6)
        lines = t.splitlines()
        self.assertTrue(lines[0].startswith("ac | x: frequency | 46 points (6 shown)"))
        self.assertEqual(lines[1], self.m.AC_LEGEND["magphase"])
        self.assertEqual(lines[7].split(), ["frequency", "v_out_Mag", "v_out_Phase"])
        self.assertEqual(lines[8].split()[0], "1")
        self.assertEqual(lines[-1].split()[0], "1e+09")

    def test_one_plot_file_names_no_plot(self):
        t = self.text(HERE / "test_9601.tr0", ["v_vo"], points=2)
        self.assertNotIn("plot", t.splitlines()[0])
        self.assertTrue(t.startswith("transient | x: TIME | "))

    def test_sweeps_with_a_parameter(self):
        sweeps = []
        rng = np.random.default_rng(5)
        for n, r in ((150, 1000.0), (97, 2000.0), (231, 3000.0)):
            data = rng.random((n, 2))
            data[:, 0] = np.linspace(0, 1e-6, n)
            sweeps.append(([r], data))
        path = self.dir / "ms.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "r1"], [1, 1], sweeps)
        t = self.text(path, points=3)
        lines = t.splitlines()
        self.assertEqual(lines[0], "transient | x: TIME | 478 points in 3 sweeps (9 shown)")
        self.assertEqual(lines[3].split()[0], "v_a[0]")
        self.assertEqual(lines[5].split()[0], "v_a[2]")
        self.assertEqual([l for l in lines if l.startswith("--")],
                         ["-- sweep 0: r1=1000 --", "-- sweep 1: r1=2000 --", "-- sweep 2: r1=3000 --"])
        self.assertEqual(lines[-1].split()[0], "1e-06")
        # one sweep kept: plain table, no sweep labels
        one = self.text(path, points=3, sweeps=[2])
        self.assertTrue(one.startswith("transient | x: TIME | 231 points (3 shown)"))
        self.assertNotIn("--", one)
        self.assertNotIn("[", one)

    def test_flat_nested_sweep_is_split_on_x_restarts(self):
        x = np.tile(np.linspace(0, 0.8, 5), 3)
        y = np.concatenate([x[:5] * k for k in (1, 2, 3)])
        path = self.dir / "nested.raw"
        fixtures.write_nutmeg(path, [("DC transfer characteristic", ["v(v-sweep)", "i(v1)"],
                                      np.c_[x, y], False)])
        t = self.text(path, points=3)
        lines = t.splitlines()
        self.assertEqual(lines[0], "dc sweep | x: v_v-sweep | 15 points (9 shown)")
        self.assertEqual(lines[3].split(), ["i_v1", "0", "2.4"])
        heads = [i for i, l in enumerate(lines) if l.startswith("--")]
        self.assertEqual([lines[i] for i in heads],
                         [f"-- run {j} of 3 (x restarts) --" for j in (1, 2, 3)])
        for i, k in zip(heads, (1, 2, 3)):
            self.assertEqual(lines[i + 2].split(), ["0", "0"])
            self.assertEqual(lines[i + 4].split(), ["0.8", self.m._num(0.8 * k)])

    def test_random_x_is_not_split(self):
        from test_reader import make_multi

        path, _ = make_multi(self.dir, "2001")
        t = self.text(path, ["v_a"], points=3)
        self.assertNotIn("run ", t)
        self.assertEqual(t.count("-- sweep"), 3)

    def test_nan_inf_and_negative_zero(self):
        data = np.array([[0.0, -0.0], [1e-9, float("nan")], [2e-9, float("inf")]])
        path = self.dir / "nan.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", ["time", "v(out)"], data, False)])
        lines = self.text(path).splitlines()
        self.assertEqual(lines[3].split(), ["v_out", "0", "inf"])     # min/max skip the NaN
        self.assertEqual(lines[6].split(), ["0", "0"])
        self.assertEqual(lines[7].split(), ["1e-09", "nan"])
        self.assertEqual(lines[8].split(), ["2e-09", "inf"])
        self.assertEqual(self.m._num(float("-inf")), "-inf")
        self.assertEqual(self.m._num(0.30000000000000004), "0.3")
        self.assertEqual(self.m._num(-0.0009999999999999998), "-0.001")

    def test_truncated_file_says_so_first(self):
        data = np.arange(10 * 2, dtype=np.float64).reshape(10, 2)
        path = self.dir / "trunc.raw"
        fixtures.write_nutmeg(path, [("Transient Analysis", ["time", "v(a)"], data, False)])
        raw = path.read_bytes()
        path.write_bytes(raw[:len(raw) - (2 * 8 + 3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = self.text(path)
        lines = t.splitlines()
        self.assertEqual(lines[0], self.m.TRUNCATED_LINE)
        self.assertEqual(lines[1], "transient | x: time | 8 points")

    def test_empty_window(self):
        t = self.text(ASCII, ["v_out"], xrange=(5, 6))
        self.assertEqual(t, "transient | x: time | 0 points | plot 0: Transient Analysis (file has 3 plots)\n"
                            "no points in the selected x window")

    def test_row_cap(self):
        t = self.text(HERE / "test_9601.tr0", ["v_vo"], points=10 ** 6)
        self.assertIn(f"({self.m.TEXT_MAX_ROWS} shown)", t.splitlines()[0])
        self.assertEqual(len(t.splitlines()), 6 + self.m.TEXT_MAX_ROWS)

    def test_only_x(self):
        t = self.text(ASCII, [], points=2)
        lines = t.splitlines()
        self.assertTrue(lines[0].startswith("transient | x: time | 80 points (2 shown) | plot 0"))
        self.assertEqual(lines[2].split(), ["time"])
        self.assertEqual(len(lines), 5)


class TestMcp(unittest.TestCase):
    """Item 12."""

    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from spice_result_parser import mcp_server

        self.m = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_list_traces_tool_reports_plots(self):
        out = self.m.list_traces(str(ASCII), plot=2)
        self.assertEqual(out["plots"], PLOT_NAMES)
        self.assertEqual(out["traces"], ["v_in", "v_out", "i_v1"])
        json.dumps(out)

    def test_extract_tool_defaults_to_text(self):
        out = self.m.extract(str(BIN), ["v(out)"], plot=2)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("dc sweep | x: v_v-sweep | 5 points"))
        self.assertIn("(6 shown)", self.m.extract(str(BIN), ["v(out)"], plot=1, downsample=6))
        tools = {t.name: t for t in asyncio.run(self.m.server.list_tools())}
        schema = getattr(tools["extract"], "inputSchema", None) or tools["extract"].input_schema
        self.assertEqual(schema["properties"]["output"]["default"], "text")
        result = asyncio.run(self.m.server.call_tool("extract", {"path": str(BIN), "names": ["v(out)"], "plot": 2}))
        content = getattr(result, "content", None) or (result[0] if isinstance(result, tuple) else result)
        self.assertTrue(content[0].text.startswith("dc sweep | x: v_v-sweep"))

    def test_extract_tool_reaches_the_ac_plot(self):
        out = self.m.extract(str(BIN), ["v(out)"], output="summary", downsample=10, plot=1)
        self.assertEqual(out["analysis"], "ac")
        self.assertEqual(out["plot"], 1)
        self.assertEqual(sorted(out["traces"]), ["v_out_Mag", "v_out_Phase"])
        self.assertEqual(out["traces"]["v_out_Mag"][0]["count"], 46)
        self.assertEqual(len(out["traces"]["v_out_Mag"][0]["x"]), 10)
        json.dumps(out)

    def test_extract_tool_writes_a_csv_of_one_plot(self):
        dest = self.dir / "dc.csv"
        out = self.m.extract(str(ASCII), ["v(out)"], output="csv", dest=str(dest), plot=2)
        self.assertEqual(out["points"], [5])
        self.assertEqual(out["traces"], ["v_v-sweep", "v_out"])
        self.assertFalse(out["truncated"])
        self.assertTrue(dest.exists())
        json.dumps(out)

    def test_plot_out_of_range_reaches_the_client(self):
        with self.assertRaisesRegex(self.m.ToolError, "out of range"):
            self.m.list_traces(str(BIN), plot=9)

    def test_tool_schema_carries_plot(self):
        tools = {t.name: t for t in asyncio.run(self.m.server.list_tools())}
        for name in ("list_traces", "extract"):
            tool = tools[name]
            schema = getattr(tool, "inputSchema", None) or tool.input_schema   # mcp 1.x / 2.x
            self.assertIn("plot", schema["properties"])


class TestDocs(unittest.TestCase):
    def test_nutmeg_format_is_documented(self):
        text = (HERE.parent / "nutmeg_output.md").read_text()
        for token in ("Plotname", "Binary:", "Values:", "Dimensions"):
            self.assertIn(token, text)
        for doc in ("Usage.md", "README.md"):
            text = (HERE.parent / doc).read_text()
            self.assertIn("Nutmeg", text)
            self.assertIn("`text`", text)


if __name__ == "__main__":
    unittest.main()
