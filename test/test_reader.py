"""Tests for the streaming reader, API and MCP server. Fixtures are generated in temp dirs."""
import asyncio
import json
import pickle
import shutil
import struct
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402

from hspice_parser.reader import read_header, read_traces  # noqa: E402
from hspice_parser import reader  # noqa: E402
from hspice_parser import api  # noqa: E402


class TestHeader(unittest.TestCase):
    def test_tr_9601(self):
        h = read_header(HERE / "test_9601.tr0")
        self.assertTrue(h.is_binary)
        self.assertEqual(h.version, "9601")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.ncols, 5)
        self.assertEqual(h.nsweepparam, 0)
        self.assertEqual(h.sweep_count_hint, 0)
        self.assertEqual(h.x_name, "TIME")
        self.assertEqual(h.names, ["TIME", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.raw_names, ["TIME", "v(0", "v(vo", "v(vs", "i(vs"])
        self.assertEqual(h.col_raw_names, h.raw_names)
        self.assertEqual(h.type_codes, [1, 1, 1, 1, 8])
        self.assertEqual(h.sweep_params, [])
        self.assertEqual(h.dtype, np.dtype("<f4"))
        self.assertEqual(h.sentinel, np.float32(1e30))
        self.assertEqual(h.path, str(HERE / "test_9601.tr0"))

    def test_tr_2001(self):
        h = read_header(HERE / "test_2001.tr0")
        self.assertEqual(h.version, "2001")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.names, ["TIME", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.dtype, np.dtype("<f8"))
        self.assertEqual(h.sentinel, 1e30)

    def test_sw_9601(self):
        h = read_header(HERE / "test_9601.sw0")
        self.assertEqual(h.analysis, "sw")
        self.assertEqual(h.x_name, "r1")
        self.assertEqual(h.names, ["r1", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.type_codes, [3, 1, 1, 1, 8])

    def test_ac_9601(self):
        h = read_header(HERE / "test_9601.ac0")
        self.assertEqual(h.analysis, "ac")
        self.assertEqual(h.ncols, 9)
        self.assertEqual(h.x_name, "HERTZ")
        self.assertEqual(
            h.names,
            ["HERTZ", "v_0_Mag", "v_0_Phase", "v_vo_Mag", "v_vo_Phase",
             "v_vs_Mag", "v_vs_Phase", "i_vs_Mag", "i_vs_Phase"],
        )
        self.assertEqual(h.raw_names, ["HERTZ", "v(0", "v(vo", "v(vs", "i(vs"])
        self.assertEqual(
            h.col_raw_names,
            ["HERTZ", "v(0", "v(0", "v(vo", "v(vo", "v(vs", "v(vs", "i(vs", "i(vs"],
        )

    def test_extension_disagrees_with_type_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "renamed.sw0"
            shutil.copy(HERE / "test_9601.tr0", bad)
            with self.assertRaisesRegex(ValueError, "extension says 'sw'"):
                read_header(bad)

    def test_unsupported_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "future.tr0"
            data = (HERE / "test_9601.tr0").read_bytes().replace(b"9601    *", b"2013    *", 1)
            bad.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "unsupported post_version '2013'"):
                read_header(bad)

    def test_unknown_extension_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "anything.bin"
            shutil.copy(HERE / "test_9601.tr0", other)
            self.assertEqual(read_header(other).analysis, "tr")


class TestFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fortran_fields(self):
        self.assertEqual(fixtures.fortran(1e30), "0.1000000E+31")
        self.assertEqual(fixtures.fortran(0.0), "0.0000000E+00")
        self.assertEqual(fixtures.fortran(1000.0), "0.1000000E+04")
        self.assertEqual(fixtures.fortran(-2.5e-11), "-.2500000E-10")
        for v in [3.14159, -0.00042, 7.5e12, 9.9999999e-3, 1.0, -1.0]:
            self.assertEqual(len(fixtures.fortran(v)), 13)
            self.assertAlmostEqual(float(fixtures.fortran(v)), v, delta=abs(v) * 1e-6)

    def test_binary_fixture_readable_by_old_parser(self):
        from hspice_parser.hspiceParser import parse_header, read_binary_signal_file

        rng = np.random.default_rng(1)
        sweeps = [([1000.0], rng.random((11, 4))), ([2000.0], rng.random((6, 4)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b", "i(c", "r1"], [1, 1, 1, 8], sweeps)
        header, blocks = read_binary_signal_file(str(path))
        self.assertEqual(header[20:24], "2001")
        self.assertEqual(header[0:12], "000400000001")
        self.assertEqual(parse_header(header)[1], ["TIME", "v_a", "v_b", "i_c", "r1"])
        flat = np.array(sum(blocks, []))
        np.testing.assert_array_equal(flat, stream)
        self.assertEqual(len(stream), (1 + 44 + 1) + (1 + 24 + 1))

    def test_binary_fixture_9601_blocks(self):
        rng = np.random.default_rng(4)
        sweeps = [([], rng.random((3000, 5)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "9601", ["TIME", "v(a", "v(b", "v(c", "i(d"], [1, 1, 1, 1, 8], sweeps)
        self.assertEqual(stream.dtype, np.dtype("<f4"))
        raw = path.read_bytes()
        n0 = struct.unpack("<i", raw[12:16])[0]
        self.assertEqual(raw[16:36], b"00050000000000009601")          # nauto=5, nsweepparam=0, version at 16:20
        first_data = 20 + n0
        self.assertEqual(struct.unpack("<i", raw[first_data + 12:first_data + 16])[0], 8192)

    def test_binary_fixture_truncation_options(self):
        rng = np.random.default_rng(5)
        sweeps = [([1.0], rng.random((10, 3))), ([2.0], rng.random((10, 3)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b", "p"], [1, 1, 1], sweeps,
                                       final_sentinel=False, drop_tail=2)
        self.assertEqual(len(stream), (1 + 30 + 1) + (1 + 30) - 2)
        self.assertNotEqual(stream[-1], 1e30)

    def test_ascii_fixture_readable_by_old_parser(self):
        from hspice_parser.hspiceParser import signal_file_ascii_read

        rng = np.random.default_rng(2)
        sweeps = [([1000.0], rng.uniform(-1, 1, (7, 4))), ([2000.0], rng.uniform(-1, 1, (5, 4)))]
        path = self.dir / "fx.tr0"
        fixtures.write_ascii(path, ["TIME", "v(a", "v(b", "i(c", "r1"], [1, 1, 1, 8], sweeps)
        old = signal_file_ascii_read(str(path))
        self.assertEqual(sorted(old), sorted(["TIME", "v_a", "v_b", "i_c", "param_r1"]))
        self.assertEqual(old["param_r1"], [[1000.0], [2000.0]])
        np.testing.assert_allclose(old["v_a"][0], sweeps[0][1][:, 1], rtol=1e-6)
        self.assertEqual(len(old["i_c"][1]), 5)


MULTI_NAMES = ["TIME", "v(a", "v(b", "v(c", "v(d", "i(e", "i(f", "r1"]   # 7 data columns + 1 sweep param
MULTI_CODES = [1, 1, 1, 1, 1, 8, 8]


def make_multi(directory, version, **kw):
    """Three-sweep transient fixture with 7 columns so points straddle 8192-byte blocks."""
    rng = np.random.default_rng(3)
    sweeps = [([1000.0], rng.random((1500, 7))),
              ([2000.0], rng.random((977, 7))),
              ([3000.0], rng.random((2310, 7)))]
    path = Path(directory) / f"multi_{version}.tr0"
    fixtures.write_binary(path, version, MULTI_NAMES, MULTI_CODES, sweeps, **kw)
    return path, sweeps


class TestBinaryRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_matches_old(self, ts, old):
        self.assertEqual(sorted(ts.data), sorted(old))
        self.assertEqual(ts.selected, list(old))
        for key in old:
            self.assertEqual(len(ts.data[key]), len(old[key]))
            for new_arr, old_lst in zip(ts.data[key], old[key]):
                np.testing.assert_array_equal(new_arr.astype(np.float64), np.asarray(old_lst, dtype=np.float64))

    def test_tr_9601_matches_old(self):
        with open(HERE / "data_dict_9601.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_9601.tr0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.sweep_values, [[]])
        self.assertFalse(ts.truncated)
        self.assertEqual(ts.data["TIME"][0].dtype, np.dtype("<f4"))

    def test_tr_2001_matches_old(self):
        with open(HERE / "data_dict_tr_2001.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_2001.tr0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.data["TIME"][0].dtype, np.dtype("<f8"))

    def test_sw_9601_matches_old(self):
        with open(HERE / "data_dict_sw_9601.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_9601.sw0")
        self.assert_matches_old(ts, old)

    def test_ac_9601_matches_old_ac_path(self):
        # data_dict_ac_9601.pickle was produced through the "tr" path and is wrong; use the live ac path.
        from hspice_parser.hspiceParser import read_binary_signal_file, write_to_dict

        header_str, blocks = read_binary_signal_file(str(HERE / "test_9601.ac0"))
        old, _, _, _ = write_to_dict(blocks, header_str, "ac")
        ts = read_traces(HERE / "test_9601.ac0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.selected[:3], ["HERTZ", "v_0_Mag", "v_0_Phase"])
        self.assertEqual(ts.data["HERTZ"][0].size, 41)

    def _check_multi(self, version):
        path, sweeps = make_multi(self.dir, version)
        dtype = np.dtype("<f4") if version == "9601" else np.dtype("<f8")
        ts = read_traces(path)
        self.assertEqual(ts.header.nsweepparam, 1)
        self.assertEqual(ts.header.sweep_params, ["r1"])
        self.assertEqual(ts.header.sweep_count_hint, 3)
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0], [3000.0]])
        self.assertEqual(ts.selected, ["TIME", "v_a", "v_b", "v_c", "v_d", "i_e", "i_f"])
        self.assertFalse(ts.truncated)
        for col, name in enumerate(ts.selected):
            self.assertEqual(len(ts.data[name]), 3)
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(ts.data[name][i], data[:, col].astype(dtype))

    def test_multi_sweep_9601(self):
        self._check_multi("9601")

    def test_multi_sweep_2001(self):
        self._check_multi("2001")

    def test_truncated_file_keeps_complete_points(self):
        path, sweeps = make_multi(self.dir, "2001", final_sentinel=False, drop_tail=3)
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 3)
        self.assertEqual(ts.sweep_values[2], [3000.0])
        last = sweeps[2][1]
        for col, name in enumerate(ts.selected):
            self.assertEqual(ts.data[name][2].size, 2309)
            np.testing.assert_array_equal(ts.data[name][2], last[:2309, col])
        np.testing.assert_array_equal(ts.data["v_a"][1], sweeps[1][1][:, 1])

    def test_corrupt_tail_raises(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off1 = 20 + n0
        size1 = struct.unpack("<i", b[off1 + 12:off1 + 16])[0]
        tail = off1 + 16 + size1
        b[tail:tail + 4] = struct.pack("<i", size1 + 4)
        path.write_bytes(bytes(b))
        with self.assertRaisesRegex(ValueError, "block 1: tail length"):
            read_traces(path)

    def test_short_final_block_truncates_gracefully(self):
        path, sweeps = make_multi(self.dir, "9601")
        b = path.read_bytes()
        path.write_bytes(b[:-10])                       # kills the last tail and part of its payload
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 3)
        for col, name in enumerate(ts.selected):
            for i in (0, 1):
                np.testing.assert_array_equal(ts.data[name][i], sweeps[i][1][:, col].astype(np.dtype("<f4")))

    def test_truncated_mid_block_keeps_prefix(self):
        path, sweeps = make_multi(self.dir, "9601")
        b = path.read_bytes()
        cut = 12000
        n0 = struct.unpack("<i", b[12:16])[0]
        payload2 = 20 + n0 + (16 + 8192 + 4) + 16       # first byte of block 2's payload
        self.assertLess(payload2, cut)
        self.assertLess(cut, payload2 + 8192)           # the cut lands inside block 2
        path.write_bytes(b[:cut])
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 1)
        values = (8192 + (cut - payload2)) // 4         # whole float32 values kept
        npoints = (values - 1) // 7                     # one sweep-parameter value precedes the points
        for col, name in enumerate(ts.selected):
            self.assertEqual(ts.data[name][0].size, npoints)
            np.testing.assert_array_equal(ts.data[name][0],
                                          sweeps[0][1][:npoints, col].astype(np.dtype("<f4")))

    def test_negative_block_size_raises(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off1 = 20 + n0
        b[off1 + 12:off1 + 16] = struct.pack("<i", -8)
        path.write_bytes(bytes(b))
        with self.assertRaisesRegex(ValueError, "block 1: negative block size"):
            read_traces(path)

    def test_collector_grows_when_capacity_is_too_small(self):
        path, sweeps = make_multi(self.dir, "2001")
        header = read_header(path)
        collector = reader._Collector(header, list(range(header.ncols)), None, capacity=1)
        with open(path, "rb") as f:
            reader._read_binary_header(f, path)
            for block in reader._iter_binary_blocks(f, header):
                collector.feed(block)
        self.assertFalse(collector.finish())
        self.assertGreaterEqual(collector.capacity, 1500 + 977 + 2310)
        self.assertEqual(collector.sweep_indices, [0, 1, 2])
        for col in range(7):
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(collector.data[col][i], data[:, col])

    def test_kept_sweeps_are_views_of_one_buffer(self):
        path, _ = make_multi(self.dir, "2001")
        ts = read_traces(path, ["v_a"])
        for arr in ts.data["v_a"]:
            self.assertIsNotNone(arr.base)
        bases = {id(arr.base) for arr in ts.data["v_a"]}
        self.assertEqual(len(bases), 1)
        self.assertEqual(sum(a.size for a in ts.data["v_a"]), 1500 + 977 + 2310)

    def test_header_only_file_has_no_sweeps(self):
        path = self.dir / "empty.tr0"
        text = fixtures.header_text("2001", ["TIME", "v(a"], [1, 1], 0, 0)
        path.write_bytes(fixtures._block(text.encode("utf-8")))
        ts = read_traces(path)
        self.assertEqual(ts.sweep_values, [])
        self.assertEqual(ts.sweep_indices, [])
        self.assertEqual(ts.data["v_a"], [])
        self.assertFalse(ts.truncated)

    def test_capacity_estimates(self):
        # 16424 bytes of frames after a header: two 8192-byte-payload frames would be 16424 bytes
        self.assertEqual(reader._binary_capacity(16424 + 100, 100, 8192, 20, 8), 16384 // 8 // 20 + 1)
        self.assertEqual(reader._binary_capacity(100, 100, 8192, 20, 8), 1)
        self.assertEqual(reader._ascii_capacity(13 * 40 + 200, 4), (13 * 40 + 200) // 13 // 4 + 1)

    def test_bulk_path_with_tiny_chunks(self):
        path, sweeps = make_multi(self.dir, "2001")
        original = reader.FRAMES_PER_CHUNK
        reader.FRAMES_PER_CHUNK = 3
        self.addCleanup(setattr, reader, "FRAMES_PER_CHUNK", original)
        ts = read_traces(path, ["v_c", "i_f"], sweeps=[0, 2])
        self.assertEqual(ts.sweep_indices, [0, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [3000.0]])
        self.assertFalse(ts.truncated)
        for name, col in (("TIME", 0), ("v_c", 3), ("i_f", 6)):
            np.testing.assert_array_equal(ts.data[name][0], sweeps[0][1][:, col])
            np.testing.assert_array_equal(ts.data[name][1], sweeps[2][1][:, col])

    def test_bulk_buffer_is_clamped_for_huge_declared_blocks(self):
        # A 1 MB block size makes the fixture a single ~268 KB block; unclamped, the bulk
        # reader would allocate FRAMES_PER_CHUNK frames of that size (~69 MB). The clamp on
        # remaining file bytes must bring the read buffer down to one frame.
        path, sweeps = make_multi(self.dir, "2001", block_bytes=1 << 20)
        tracemalloc.start()
        try:
            ts = read_traces(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, reader._MAX_CHUNK_BYTES * 3 + 1_000_000, f"peak {peak} bytes")
        for col, name in enumerate(ts.selected):
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(ts.data[name][i], data[:, col])

    def test_mixed_block_sizes_fall_back_to_per_block(self):
        path, sweeps = make_multi(self.dir, "2001", block_bytes=[8192, 8192, 8192, 4096, 8192, 2048])
        ts = read_traces(path)
        self.assertEqual(ts.sweep_indices, [0, 1, 2])
        for col, name in enumerate(ts.selected):
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(ts.data[name][i], data[:, col])

    def test_corruption_deep_in_a_chunk_names_the_block(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off = 20 + n0 + 4 * (8192 + 20)          # head of data block 5
        size = struct.unpack("<i", b[off + 12:off + 16])[0]
        self.assertEqual(size, 8192)
        tail = off + 16 + size
        b[tail:tail + 4] = struct.pack("<i", size + 4)
        path.write_bytes(bytes(b))
        with self.assertRaisesRegex(ValueError, "block 5: tail length"):
            read_traces(path)

    def test_bulk_and_sample_files_agree_on_block_boundaries(self):
        # test_9601.tr0 has six 8192-byte blocks and one short block: bulk path then per-block hand-off.
        ts = read_traces(HERE / "test_9601.tr0")
        with open(HERE / "data_dict_9601.pickle", "rb") as f:
            old = pickle.load(f)
        np.testing.assert_array_equal(ts.data["i_vs"][0].astype(np.float64), np.asarray(old["i_vs"][0]))

    def test_selective_sweeps_do_not_pin_the_whole_buffer(self):
        path, sweeps = make_multi(self.dir, "2001")
        ts = read_traces(path, ["v_a"], sweeps=[1])
        self.assertEqual(ts.sweep_indices, [1])
        for name in ("TIME", "v_a"):
            arr = ts.data[name][0]
            self.assertIsNone(arr.base)                 # owns its memory: the file-sized buffer is released
            self.assertEqual(arr.size, 977)
        np.testing.assert_array_equal(ts.data["v_a"][0], sweeps[1][1][:, 1])


class TestDuplicateNames(unittest.TestCase):
    """`.` and `:` both sanitize to `_`, so distinct HSPICE names can collide."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(11)
        self.data = rng.random((50, 4))
        self.path = self.dir / "dup.tr0"
        fixtures.write_binary(self.path, "2001", ["TIME", "v(a.b", "v(a:b", "v(c"],
                              [1, 1, 1, 1], [([], self.data)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_stay_one_to_one_with_columns(self):
        with self.assertWarns(RuntimeWarning):
            info = api.list_traces(self.path)
        self.assertEqual(info["traces"], ["v_a_b", "v_a_b#2", "v_c"])
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(self.path)
        self.assertEqual(ts.selected, ["TIME", "v_a_b", "v_a_b#2", "v_c"])
        self.assertEqual(len(ts.data), 4)
        for col, name in enumerate(ts.selected):
            np.testing.assert_array_equal(ts.data[name][0], self.data[:, col])

    def test_csv_header_and_downsample(self):
        dest = self.dir / "dup.csv"
        with self.assertWarns(RuntimeWarning):
            api.extract(self.path, output="csv", dest=dest)
        self.assertEqual(dest.read_text().splitlines()[0], "TIME,v_a_b,v_a_b#2,v_c")
        with self.assertWarns(RuntimeWarning):
            ts = api.extract(self.path, downsample=5)        # used to raise IndexError
        for name in ts.selected:
            self.assertEqual(ts.data[name][0].size, 5)


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_names(self):
        ts = read_traces(self.path, ["v_b", "i_f"])
        self.assertEqual(ts.selected, ["TIME", "v_b", "i_f"])
        self.assertEqual(sorted(ts.data), ["TIME", "i_f", "v_b"])
        np.testing.assert_array_equal(ts.data["v_b"][0], self.sweeps[0][1][:, 2])
        np.testing.assert_array_equal(ts.data["i_f"][2], self.sweeps[2][1][:, 6])

    def test_raw_names_with_or_without_paren(self):
        ts = read_traces(self.path, ["v(b)", "i(f"])
        self.assertEqual(ts.selected, ["TIME", "v_b", "i_f"])

    def test_glob(self):
        ts = read_traces(self.path, ["i_*"])
        self.assertEqual(ts.selected, ["TIME", "i_e", "i_f"])

    def test_single_string(self):
        ts = read_traces(self.path, "v_a")
        self.assertEqual(ts.selected, ["TIME", "v_a"])

    def test_x_always_first_even_if_requested_last(self):
        ts = read_traces(self.path, ["v_d", "TIME"])
        self.assertEqual(ts.selected, ["TIME", "v_d"])

    def test_unknown_lists_available(self):
        with self.assertRaisesRegex(ValueError, r"unknown trace 'nope'; available traces: TIME, v_a, v_b"):
            read_traces(self.path, ["nope"])

    def test_sweeps_filter(self):
        ts = read_traces(self.path, ["v_a"], sweeps=[1])
        self.assertEqual(ts.sweep_values, [[2000.0]])
        self.assertEqual(ts.sweep_indices, [1])
        self.assertEqual(len(ts.data["v_a"]), 1)
        np.testing.assert_array_equal(ts.data["v_a"][0], self.sweeps[1][1][:, 1])
        ts = read_traces(self.path, ["v_a"], sweeps=[0, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [3000.0]])
        self.assertEqual(ts.sweep_indices, [0, 2])
        np.testing.assert_array_equal(ts.data["TIME"][1], self.sweeps[2][1][:, 0])

    def test_ac_raw_name_selects_mag_and_phase(self):
        ts = read_traces(HERE / "test_9601.ac0", ["v(vo)"])
        self.assertEqual(ts.selected, ["HERTZ", "v_vo_Mag", "v_vo_Phase"])

    def test_memory_stays_near_selected_size(self):
        rng = np.random.default_rng(6)
        npoints, ncols = 500_000, 20
        data = rng.random((npoints, ncols), dtype=np.float32)          # 40 MB
        path = self.dir / "big.tr0"
        fixtures.write_binary(path, "9601", ["TIME"] + [f"v({i}" for i in range(ncols - 1)],
                              [1] * ncols, [([], data)])
        expected = data[:, 8].copy()
        del data
        tracemalloc.start()
        tracemalloc.reset_peak()
        ts = read_traces(path, ["v(7"])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        selected_bytes = 2 * npoints * 4                                 # TIME + v_7 as float32
        chunk_bytes = reader.FRAMES_PER_CHUNK * (8192 + 20)              # one bulk read buffer
        self.assertLess(peak, selected_bytes + 3 * chunk_bytes + 1_000_000, f"peak {peak} bytes")
        self.assertEqual(ts.selected, ["TIME", "v_7"])
        np.testing.assert_array_equal(ts.data["v_7"][0], expected)

ASCII_NAMES = ["TIME", "v(a", "v(b", "i(c", "r1"]
ASCII_CODES = [1, 1, 1, 8]


def make_ascii(directory):
    rng = np.random.default_rng(2)
    sweeps = [([1000.0], rng.uniform(-1, 1, (7, 4))), ([2000.0], rng.uniform(-1, 1, (5, 4)))]
    path = Path(directory) / "ascii.tr0"
    fixtures.write_ascii(path, ASCII_NAMES, ASCII_CODES, sweeps)
    return path, sweeps


class TestAsciiRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_ascii(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header(self):
        h = read_header(self.path)
        self.assertFalse(h.is_binary)
        self.assertEqual(h.version, "ascii")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.ncols, 4)
        self.assertEqual(h.nsweepparam, 1)
        self.assertEqual(h.sweep_count_hint, 2)
        self.assertEqual(h.names, ["TIME", "v_a", "v_b", "i_c"])
        self.assertEqual(h.sweep_params, ["r1"])
        self.assertEqual(h.dtype, np.dtype("<f8"))

    def test_matches_old_ascii_reader(self):
        from hspice_parser.hspiceParser import signal_file_ascii_read

        old = signal_file_ascii_read(str(self.path))
        ts = read_traces(self.path)
        self.assertEqual(ts.selected, ["TIME", "v_a", "v_b", "i_c"])
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0]])
        self.assertFalse(ts.truncated)
        for name in ts.selected:
            for i in range(2):
                np.testing.assert_array_equal(ts.data[name][i], np.asarray(old[name][i]))

    def test_matches_generator_to_7_digits(self):
        ts = read_traces(self.path)
        for col, name in enumerate(ts.selected):
            for i, (_, data) in enumerate(self.sweeps):
                np.testing.assert_allclose(ts.data[name][i], data[:, col], rtol=1e-6)

    def test_selection_and_sweeps_on_ascii(self):
        ts = read_traces(self.path, ["v(b)"], sweeps=[1])
        self.assertEqual(ts.selected, ["TIME", "v_b"])
        self.assertEqual(ts.sweep_values, [[2000.0]])
        self.assertEqual(ts.data["v_b"][0].size, 5)

    def test_first_data_line_without_exponent_raises(self):
        lines = self.path.read_text().splitlines()
        lines[4] = "0.1000000+00 0.2000000+00"          # no 'E': field width undeterminable
        self.path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "cannot determine ASCII field width"):
            read_traces(self.path)

    def test_truncated_ascii(self):
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join(lines[:-1]) + "\n")   # drop the last line (holds the final sentinel)
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(self.path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 2)
        self.assertEqual(ts.data["TIME"][0].size, 7)


class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_traces(self):
        info = api.list_traces(HERE / "test_9601.tr0")
        self.assertEqual(info, {
            "path": str(HERE / "test_9601.tr0"), "format": "9601", "analysis": "tr", "x": "TIME",
            "traces": ["v_0", "v_vo", "v_vs", "i_vs"], "sweep_params": [], "sweep_count_hint": 0,
        })
        info = api.list_traces(self.path)
        self.assertEqual(info["sweep_params"], ["r1"])
        self.assertEqual(info["sweep_count_hint"], 3)

    def test_decimation_indices(self):
        self.assertIsNone(api.decimation_indices(10, None))
        self.assertIsNone(api.decimation_indices(10, 10))
        self.assertIsNone(api.decimation_indices(10, 20))
        np.testing.assert_array_equal(api.decimation_indices(10, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(api.decimation_indices(10, 2), [0, 9])
        with self.assertRaises(ValueError):
            api.decimation_indices(10, 1)

    def test_extract_arrays_equals_read_traces(self):
        ts = api.extract(self.path, ["v_a"])
        ref = read_traces(self.path, ["v_a"])
        self.assertEqual(ts.selected, ref.selected)
        for i in range(3):
            np.testing.assert_array_equal(ts.data["v_a"][i], ref.data["v_a"][i])

    def test_extract_arrays_downsampled(self):
        ts = api.extract(self.path, ["v_a"], downsample=5)
        for i, (_, data) in enumerate(self.sweeps):
            self.assertEqual(ts.data["TIME"][i].size, 5)
            self.assertEqual(ts.data["v_a"][i].size, 5)
            self.assertEqual(ts.data["v_a"][i][0], data[0, 1])
            self.assertEqual(ts.data["v_a"][i][-1], data[-1, 1])

    def test_summary(self):
        s = api.extract(self.path, ["v_a"], output="summary", downsample=4)
        self.assertEqual(s["analysis"], "tr")
        self.assertEqual(s["x"], "TIME")
        self.assertEqual(s["sweeps"], 3)
        self.assertEqual(s["sweep_values"], [[1000.0], [2000.0], [3000.0]])
        self.assertFalse(s["truncated"])
        self.assertEqual(list(s["traces"]), ["v_a"])
        entry = s["traces"]["v_a"][1]
        col = self.sweeps[1][1][:, 1]
        self.assertEqual(entry["count"], 977)
        self.assertEqual(entry["min"], float(col.min()))
        self.assertEqual(entry["max"], float(col.max()))
        self.assertAlmostEqual(entry["mean"], float(col.mean()), places=12)
        self.assertEqual(entry["first"], float(col[0]))
        self.assertEqual(entry["last"], float(col[-1]))
        self.assertEqual(len(entry["x"]), 4)
        self.assertEqual(len(entry["y"]), 4)
        self.assertEqual(entry["x"][0], float(self.sweeps[1][1][0, 0]))
        self.assertEqual(entry["x"][-1], float(self.sweeps[1][1][-1, 0]))
        json.dumps(s)   # must be JSON serialisable

    def test_summary_without_downsample_has_no_points(self):
        s = api.extract(self.path, ["v_a"], output="summary")
        self.assertNotIn("x", s["traces"]["v_a"][0])

    def test_summary_on_sample_file(self):
        s = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="summary", downsample=10)
        self.assertEqual(s["traces"]["v_vo"][0]["count"], 2605)
        self.assertEqual(len(s["traces"]["v_vo"][0]["y"]), 10)

    def test_bad_output(self):
        with self.assertRaisesRegex(ValueError, "output must be one of"):
            api.extract(self.path, output="xlsx")

    def test_bad_downsample_rejected_before_reading(self):
        # a path that cannot be opened: the ValueError proves nothing was read first
        with self.assertRaisesRegex(ValueError, "downsample must be at least 2"):
            api.extract(self.dir / "missing.tr0", ["v_a"], downsample=1)
        with self.assertRaisesRegex(ValueError, "downsample must be at least 2"):
            api.extract(self.path, ["v_a"], downsample=1)


class TestFileOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_dest(self):
        self.assertEqual(api.default_dest("/x/y/test_9601.tr0", "csv"), "/x/y/test_9601_tr0_traces.csv")
        self.assertEqual(api.default_dest("run.ac0", "npz"), "run_ac0_traces.npz")

    def test_csv_single_sweep(self):
        dest = self.dir / "single.csv"
        out = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="csv", dest=dest)
        self.assertEqual(out, str(dest))
        lines = dest.read_text().splitlines()
        self.assertEqual(lines[0], "TIME,v_vo")
        self.assertEqual(len(lines), 1 + 2605)
        ref = read_traces(HERE / "test_9601.tr0", ["v(vo"])
        table = np.loadtxt(dest, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(table[:, 0], ref.data["TIME"][0].astype(np.float64))
        np.testing.assert_array_equal(table[:, 1], ref.data["v_vo"][0].astype(np.float64))

    def test_csv_multi_sweep(self):
        dest = self.dir / "multi.csv"
        api.extract(self.path, ["v_a"], output="csv", dest=dest)
        lines = dest.read_text().splitlines()
        self.assertEqual(lines[0], "sweep,r1,TIME,v_a")
        self.assertEqual(len(lines), 1 + 1500 + 977 + 2310)
        table = np.loadtxt(dest, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(table[:1500, 0], 0)
        np.testing.assert_array_equal(table[1500:1500 + 977, 0], 1)
        np.testing.assert_array_equal(table[1500:1500 + 977, 1], 2000.0)
        np.testing.assert_array_equal(table[1500:1500 + 977, 3], self.sweeps[1][1][:, 1])

    def test_csv_downsampled(self):
        dest = self.dir / "ds.csv"
        api.extract(self.path, ["v_a"], output="csv", downsample=10, dest=dest)
        self.assertEqual(len(dest.read_text().splitlines()), 1 + 30)

    def test_default_csv_dest_is_beside_input(self):
        out = api.extract(self.path, ["v_a"], output="csv")
        self.assertEqual(out, str(self.dir / "multi_2001_tr0_traces.csv"))
        self.assertTrue(Path(out).exists())

    def test_npz_single_sweep(self):
        dest = self.dir / "single.npz"
        api.extract(HERE / "test_9601.tr0", ["v(vo"], output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertEqual(sorted(z.files), ["TIME", "__sweep_params__", "__sweep_values__", "v_vo"])
            ref = read_traces(HERE / "test_9601.tr0", ["v(vo"])
            np.testing.assert_array_equal(z["v_vo"], ref.data["v_vo"][0])
            self.assertEqual(z["v_vo"].dtype, np.dtype("<f4"))
            self.assertEqual(z["__sweep_values__"].shape, (1, 0))
            self.assertEqual(z["__sweep_params__"].size, 0)

    def test_npz_multi_sweep(self):
        dest = self.dir / "multi.npz"
        api.extract(self.path, ["v_a"], output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertEqual(sorted(z.files), sorted(
                ["TIME@0", "TIME@1", "TIME@2", "v_a@0", "v_a@1", "v_a@2", "__sweep_params__", "__sweep_values__"]))
            np.testing.assert_array_equal(z["v_a@2"], self.sweeps[2][1][:, 1])
            np.testing.assert_array_equal(z["__sweep_values__"], [[1000.0], [2000.0], [3000.0]])
            self.assertEqual(list(z["__sweep_params__"]), ["r1"])

    def test_csv_sweep_column_uses_original_index(self):
        dest = self.dir / "sweep2.csv"
        api.extract(self.path, ["v_a"], sweeps=[2], output="csv", dest=dest)
        table = np.loadtxt(dest, delimiter=",", skiprows=1)
        self.assertEqual(dest.read_text().splitlines()[0], "sweep,r1,TIME,v_a")
        np.testing.assert_array_equal(table[:, 0], 2)
        np.testing.assert_array_equal(table[:, 1], 3000.0)
        np.testing.assert_array_equal(table[:, 3], self.sweeps[2][1][:, 1])

    def test_npz_keys_use_original_sweep_indices(self):
        dest = self.dir / "sub.npz"
        api.extract(self.path, ["v_a"], sweeps=[1, 2], output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertIn("v_a@1", z.files)
            self.assertIn("v_a@2", z.files)
            self.assertNotIn("v_a@0", z.files)
            np.testing.assert_array_equal(z["v_a@2"], self.sweeps[2][1][:, 1])

    def test_csv_writer_memory_stays_bounded(self):
        rng = np.random.default_rng(8)
        npoints, ncols = 200_000, 5
        data = rng.random((npoints, ncols), dtype=np.float32)
        path = self.dir / "wide.tr0"
        fixtures.write_binary(path, "9601", ["TIME"] + [f"v({i}" for i in range(ncols - 1)],
                              [1] * ncols, [([], data)])
        del data
        dest = self.dir / "wide.csv"
        tracemalloc.start()
        tracemalloc.reset_peak()
        api.extract(path, ["v(2"], output="csv", dest=dest)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        selected_bytes_as_float64 = 2 * npoints * 8            # TIME + v_2 widened
        self.assertLess(peak, 2 * selected_bytes_as_float64 + 4_000_000, f"peak {peak} bytes")
        self.assertEqual(len(dest.read_text().splitlines()), 1 + npoints)

    def test_write_file_rejects_other_outputs(self):
        ts = read_traces(self.path, ["v_a"])
        with self.assertRaises(ValueError):
            api.write_file(ts, "summary")


class TestPackage(unittest.TestCase):
    def test_exports(self):
        import hspice_parser as hp

        for name in ["convert", "Header", "TraceSet", "read_header", "read_traces",
                     "list_traces", "extract", "write_file", "summarize"]:
            self.assertTrue(hasattr(hp, name), name)

    def test_pyproject_declares_mcp_extra_and_script(self):
        text = (HERE.parent / "pyproject.toml").read_text()
        self.assertIn('hsp-mcp = "hspice_parser.mcp_server:main"', text)
        self.assertIn('mcp = ["mcp>=1.0"]', text)

    def test_usage_documents_api_and_mcp(self):
        text = (HERE.parent / "Usage.md").read_text()
        self.assertIn("hsp-mcp", text)
        self.assertIn("list_traces", text)
        self.assertIn("extract(", text)


class TestMcp(unittest.TestCase):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from hspice_parser import mcp_server

        self.m = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_tools_registered(self):
        names = {t.name for t in asyncio.run(self.m.server.list_tools())}
        self.assertEqual(names, {"list_traces", "extract"})

    def test_list_traces_tool(self):
        out = self.m.list_traces(str(HERE / "test_9601.tr0"))
        self.assertEqual(out["traces"], ["v_0", "v_vo", "v_vs", "i_vs"])

    def test_arrays_rejected(self):
        with self.assertRaisesRegex(ValueError, "not available over MCP"):
            self.m.extract(str(HERE / "test_9601.tr0"), output="arrays")

    def test_bad_output_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, "summary"):
            self.m.extract(str(HERE / "test_9601.tr0"), output="xlsx")

    def test_summary_is_json(self):
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], downsample=20)
        json.dumps(out)
        self.assertEqual(len(out["traces"]["v_vo"][0]["x"]), 20)

    def test_csv_tool_returns_path_and_counts(self):
        dest = self.dir / "o.csv"
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv", dest=str(dest))
        self.assertEqual(out["path"], str(dest))
        self.assertEqual(out["traces"], ["TIME", "v_vo"])
        self.assertEqual(out["sweeps"], 1)
        self.assertEqual(out["points"], [2605])
        self.assertEqual(out["sweep_indices"], [0])
        self.assertFalse(out["truncated"])
        self.assertTrue(dest.exists())
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
