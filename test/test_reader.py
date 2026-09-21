"""Tests for the streaming reader, API and MCP server. Fixtures are generated in temp dirs."""
import asyncio
import json
import pickle
import re
import shutil
import struct
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

    def test_bulk_buffer_is_clamped_to_remaining_file_bytes(self):
        # A 1 MB block size makes the fixture a single ~268 KB block. Without the clamp on
        # remaining file bytes the bulk reader would allocate _MAX_CHUNK_BYTES // frame = 3
        # frames of that size (~3 MB); the clamp must bring the read buffer down to one frame.
        path, sweeps = make_multi(self.dir, "2001", block_bytes=1 << 20)
        tracemalloc.start()
        try:
            ts = read_traces(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2 * path.stat().st_size + 1_000_000, f"peak {peak} bytes")
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


    def test_growth_does_not_strand_old_buffers(self):
        # First block 512 bytes, later blocks 1 MB: the capacity estimate from the first block is
        # low, the buffers grow once, and finish() must not leave the old buffers pinned.
        path, sweeps = make_multi(self.dir, "2001", block_bytes=[512, 1 << 20])
        ts = read_traces(path, ["v_a"])
        npts = 1500 + 977 + 2310
        for name, col in (("TIME", 0), ("v_a", 1)):
            arrs = ts.data[name]
            bases = {id(a.base): a.base for a in arrs if a.base is not None}
            retained = sum(b.nbytes for b in bases.values()) + sum(a.nbytes for a in arrs if a.base is None)
            self.assertLessEqual(retained, int(npts * 8 * 1.1) + 64, f"{name} retains {retained} bytes")
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(arrs[i], data[:, col])

    def test_oversized_declared_block_takes_the_per_block_path(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off = 20 + n0                                  # head of data block 1
        b[off + 12:off + 16] = struct.pack("<i", 1 << 28)
        path.write_bytes(bytes(b))

        def must_not_run(*args, **kwargs):
            raise AssertionError("bulk path used for a block larger than _MAX_CHUNK_BYTES")

        original = reader._feed_bulk_frames
        reader._feed_bulk_frames = must_not_run
        self.addCleanup(setattr, reader, "_feed_bulk_frames", original)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ts = read_traces(path)
        self.assertTrue(ts.truncated)

    def test_byte_cap_binds_the_bulk_buffer(self):
        path, sweeps = make_multi(self.dir, "2001")
        original = reader._MAX_CHUNK_BYTES
        reader._MAX_CHUNK_BYTES = 2 * (8192 + 20)      # two frames per readinto
        self.addCleanup(setattr, reader, "_MAX_CHUNK_BYTES", original)
        tracemalloc.start()
        try:
            ts = read_traces(path, ["v_b"])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        npts = 1500 + 977 + 2310
        # 200 KB slack: with the byte-cap term removed the buffer is ~271 KB and the peak ~650 KB.
        self.assertLess(peak, 2 * npts * 8 + 3 * reader._MAX_CHUNK_BYTES + 200_000, f"peak {peak} bytes")
        for i, (_, data) in enumerate(sweeps):
            np.testing.assert_array_equal(ts.data["v_b"][i], data[:, 2])

    def _read_both_ways(self, path):
        def run():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    ts, err = read_traces(path), None
                except ValueError as e:
                    ts, err = None, str(e)
            return ts, err, [str(w.message) for w in caught]

        bulk = run()
        original = reader._peek_block_payload
        reader._peek_block_payload = lambda f: 0          # forces the per-block path
        try:
            per_block = run()
        finally:
            reader._peek_block_payload = original
        return bulk, per_block

    def test_bulk_and_per_block_paths_agree(self):
        cases = {
            "uniform": {},
            "mixed": {"block_bytes": [8192, 4096]},
            "tiny": {"block_bytes": 64},
            "partial_point": {"drop_tail": 3},
            "no_final_sentinel": {"final_sentinel": False},
        }
        for label, kw in cases.items():
            sub = self.dir / label
            sub.mkdir()
            path, _ = make_multi(sub, "9601", **kw)
            if label == "uniform":                          # also a corrupt-tail variant
                bad = bytearray(path.read_bytes())
                n0 = struct.unpack("<i", bad[12:16])[0]
                off = 20 + n0 + 2 * (8192 + 20)
                size = struct.unpack("<i", bad[off + 12:off + 16])[0]
                tail = off + 16 + size
                bad[tail:tail + 4] = struct.pack("<i", size + 4)
                (sub / "corrupt.tr0").write_bytes(bytes(bad))
                cases_paths = [path, sub / "corrupt.tr0"]
            else:
                cases_paths = [path]
            for p in cases_paths:
                with self.subTest(case=label, file=p.name):
                    (ts_b, err_b, warn_b), (ts_p, err_p, warn_p) = self._read_both_ways(p)
                    self.assertEqual(err_b, err_p)
                    self.assertEqual(warn_b, warn_p)
                    if ts_b is not None:
                        self.assertEqual(ts_b.truncated, ts_p.truncated)
                        self.assertEqual(ts_b.sweep_indices, ts_p.sweep_indices)
                        self.assertEqual(ts_b.sweep_values, ts_p.sweep_values)
                        for name in ts_b.selected:
                            self.assertEqual(len(ts_b.data[name]), len(ts_p.data[name]))
                            for a, b in zip(ts_b.data[name], ts_p.data[name]):
                                np.testing.assert_array_equal(a, b)


    def test_incomplete_point_is_dropped_at_a_sweep_boundary(self):
        # Sweep 0 ends with 3 stray values before its sentinel; sweep 1 must start clean.
        path, _ = make_multi(self.dir, "2001")
        header = read_header(path)
        c = reader._Collector(header, list(range(7)), None, capacity=8)
        sweep0 = [1000.0] + list(range(14)) + [90.0, 91.0, 92.0] + [1e30]
        sweep1 = [2000.0] + [float(v) for v in range(100, 107)] + [1e30]
        c.feed(np.array(sweep0 + sweep1, dtype=np.float64))
        self.assertFalse(c.finish())
        self.assertEqual([a.size for a in c.data[0]], [2, 1])
        np.testing.assert_array_equal(np.stack([c.data[k][1] for k in range(7)], axis=1)[0],
                                      np.arange(100.0, 107.0))
        np.testing.assert_array_equal(np.stack([c.data[k][0] for k in range(7)], axis=1),
                                      np.arange(14.0).reshape(2, 7))

    def test_partial_sweep_parameter_prefix_is_padded_with_nan(self):
        names = ["TIME", "v(a", "p0", "p1"]
        d = np.arange(12.0).reshape(6, 2)
        sweeps = [([0.0, 10.0], d), ([1.0, 11.0], d), ([2.0, 12.0], np.empty((0, 2)))]
        path = self.dir / "ragged.tr0"
        fixtures.write_binary(path, "2001", names, [1, 1], sweeps, final_sentinel=False, drop_tail=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(ts.sweep_values[:2], [[0.0, 10.0], [1.0, 11.0]])
        self.assertEqual(ts.sweep_values[2][0], 2.0)
        self.assertTrue(np.isnan(ts.sweep_values[2][1]))
        self.assertEqual(ts.data["v_a"][2].size, 0)

    def test_nan_values_do_not_hide_the_sentinel(self):
        d1 = np.arange(21.0).reshape(3, 7)
        d1[1, 3] = np.nan
        d2 = np.arange(100.0, 114.0).reshape(2, 7)
        path = self.dir / "nan.tr0"
        fixtures.write_binary(path, "2001", MULTI_NAMES, [1, 1, 1, 1, 1, 8, 8], [([1.0], d1), ([2.0], d2)])
        ts = read_traces(path)
        self.assertEqual(ts.sweep_values, [[1.0], [2.0]])
        np.testing.assert_array_equal(ts.data["v_c"][0], d1[:, 3])      # NaN compares equal here
        np.testing.assert_array_equal(ts.data["TIME"][1], d2[:, 0])

    def test_resolve_columns_is_fast_for_wide_headers(self):
        import time
        header = read_header(make_multi(self.dir, "2001")[0])
        names = ["TIME"] + [f"v_{i}" for i in range(20000)]
        wide = reader.Header(path=header.path, is_binary=True, version="2001", analysis="tr", ncols=len(names),
                             nsweepparam=0, sweep_count_hint=0, x_name="TIME", names=names,
                             raw_names=names, col_raw_names=names, type_codes=[1] * len(names), sweep_params=[])
        t = time.perf_counter()
        cols = reader.resolve_columns(wide, [f"v_{i}" for i in range(0, 20000, 2)])
        self.assertEqual(len(cols), 10001)
        self.assertLess(time.perf_counter() - t, 1.0)

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
        # Deliberately tight (about 1.1x headroom): a bump to FRAMES_PER_CHUNK or
        # _MAX_CHUNK_BYTES should fail here.
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


    def test_chunk_size_does_not_change_ascii_results(self):
        # 13 bytes is one field, 52 one point, 66 one line of the fixture: every chunk size
        # must give the same values, whether it splits a field, a point or a line.
        full = read_traces(self.path)
        original = reader.ASCII_CHUNK_BYTES
        self.addCleanup(setattr, reader, "ASCII_CHUNK_BYTES", original)
        for chunk in (7, 13, 52, 66, 4096):
            reader.ASCII_CHUNK_BYTES = chunk
            small = read_traces(self.path)
            self.assertEqual(small.sweep_values, full.sweep_values, chunk)
            self.assertEqual(small.sweep_indices, full.sweep_indices, chunk)
            self.assertFalse(small.truncated, chunk)
            for name in full.selected:
                for a, b in zip(full.data[name], small.data[name]):
                    np.testing.assert_array_equal(a, b, err_msg=f"chunk {chunk}")

    def test_crlf_line_endings(self):
        full = read_traces(self.path)
        crlf = self.dir / "crlf.tr0"
        crlf.write_bytes(self.path.read_bytes().replace(b"\n", b"\r\n"))
        original = reader.ASCII_CHUNK_BYTES
        self.addCleanup(setattr, reader, "ASCII_CHUNK_BYTES", original)
        for chunk in (7, 4096):
            reader.ASCII_CHUNK_BYTES = chunk
            ts = read_traces(crlf)
            self.assertEqual(ts.selected, full.selected)
            self.assertEqual(ts.sweep_values, full.sweep_values)
            self.assertFalse(ts.truncated)
            for name in full.selected:
                for a, b in zip(full.data[name], ts.data[name]):
                    np.testing.assert_array_equal(a, b, err_msg=f"chunk {chunk}")

    def test_junk_field_names_the_file(self):
        lines = self.path.read_text().splitlines()
        lines[6] = lines[6][:13] + "X" * 13 + lines[6][26:]    # a field that is not a number
        self.path.write_text("\n".join(lines) + "\n")
        original = reader.ASCII_CHUNK_BYTES
        self.addCleanup(setattr, reader, "ASCII_CHUNK_BYTES", original)
        for chunk in (7, 4096):
            reader.ASCII_CHUNK_BYTES = chunk
            with self.assertRaisesRegex(ValueError, "non-numeric field"):
                read_traces(self.path)
            with self.assertRaisesRegex(ValueError, re.escape(str(self.path))):
                read_traces(self.path)

    def test_file_cut_mid_field(self):
        text = self.path.read_text().rstrip("\n")
        self.path.write_text(text[:-7])                  # the final sentinel is now a part field
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(self.path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 2)
        self.assertEqual(ts.data["TIME"][0].size, 7)
        self.assertEqual(ts.data["TIME"][1].size, 5)     # the partial field is dropped, not parsed
        np.testing.assert_allclose(ts.data["v_a"][1], self.sweeps[1][1][:, 1], rtol=1e-6)

    def test_batching_does_not_change_ascii_results(self):
        path, sweeps = make_ascii(self.dir)
        full = read_traces(path)
        original = reader.ASCII_BATCH_VALUES
        reader.ASCII_BATCH_VALUES = 7                      # not a multiple of the column count
        self.addCleanup(setattr, reader, "ASCII_BATCH_VALUES", original)
        small = read_traces(path)
        self.assertEqual(full.sweep_values, small.sweep_values)
        for name in full.selected:
            for a, b in zip(full.data[name], small.data[name]):
                np.testing.assert_array_equal(a, b)

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
            "plot": 0, "plots": [],
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


    def _x_window(self):
        x0 = self.sweeps[0][1][:, 0]
        return float(np.percentile(x0, 10)), float(np.percentile(x0, 60))

    def test_xrange_selects_rows_in_every_sweep(self):
        lo, hi = self._x_window()
        ts = api.extract(self.path, ["v_a"], xrange=(lo, hi))
        for i, (_, data) in enumerate(self.sweeps):
            mask = (data[:, 0] >= lo) & (data[:, 0] <= hi)
            self.assertGreater(int(mask.sum()), 0)
            self.assertLess(int(mask.sum()), data.shape[0])
            np.testing.assert_array_equal(ts.data["TIME"][i], data[mask, 0])
            np.testing.assert_array_equal(ts.data["v_a"][i], data[mask, 1])

    def test_summary_counts_follow_xrange(self):
        lo, hi = self._x_window()
        out = api.extract(self.path, ["v_a"], output="summary", xrange=(lo, hi))
        for i, (_, data) in enumerate(self.sweeps):
            mask = (data[:, 0] >= lo) & (data[:, 0] <= hi)
            self.assertEqual(out["traces"]["v_a"][i]["count"], int(mask.sum()))

    def test_yrange_does_not_change_arrays(self):
        plain = api.extract(self.path, ["v_a"])
        clipped = api.extract(self.path, ["v_a"], yrange=(0.0, 0.5))
        for i in range(3):
            np.testing.assert_array_equal(plain.data["v_a"][i], clipped.data["v_a"][i])

    def test_ranges_are_validated_before_reading(self):
        for bad in [(1.0, 0.0), (1.0,), "0,1", (0.0, "a")]:
            with self.assertRaisesRegex(ValueError, "xrange"):
                api.extract(self.dir / "missing.tr0", xrange=bad)
            with self.assertRaisesRegex(ValueError, "yrange"):
                api.extract(self.dir / "missing.tr0", yrange=bad)


    def test_summary_points_are_capped(self):
        original = api.SUMMARY_MAX_POINTS
        api.SUMMARY_MAX_POINTS = 100
        self.addCleanup(setattr, api, "SUMMARY_MAX_POINTS", original)
        out = api.extract(self.path, ["v_a"], output="summary", downsample=10_000_000)
        self.assertEqual(len(out["traces"]["v_a"][2]["x"]), 100 // 3)     # budget shared by 3 sweeps
        self.assertEqual(out["traces"]["v_a"][2]["count"], 2310)

    def test_summary_json_has_no_nan_tokens(self):
        d = np.arange(12.0).reshape(6, 2)
        d[2, 1] = np.nan
        sweeps = [([0.0, 10.0], d), ([1.0, 11.0], d), ([2.0, 12.0], np.empty((0, 2)))]
        path = self.dir / "ragged.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "p0", "p1"], [1, 1], sweeps,
                              final_sentinel=False, drop_tail=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out = api.extract(path, output="summary", downsample=100)
        json.dumps(out, allow_nan=False)                        # strict JSON must accept it
        self.assertEqual(out["sweep_values"][2], [2.0, None])
        self.assertIsNone(out["traces"]["v_a"][0]["y"][2])
        self.assertIsNone(out["traces"]["v_a"][0]["min"])      # NaN poisons the statistics too

    def test_summary_cap_is_a_total_budget(self):
        original = api.SUMMARY_MAX_POINTS
        api.SUMMARY_MAX_POINTS = 60                             # 2 traces x 3 sweeps -> 10 points each
        self.addCleanup(setattr, api, "SUMMARY_MAX_POINTS", original)
        out = api.extract(self.path, ["v_a", "v_b"], output="summary", downsample=1000)
        sizes = [len(e["x"]) for name in ("v_a", "v_b") for e in out["traces"][name]]
        self.assertEqual(sizes, [10] * 6)

    def test_default_caps(self):
        self.assertEqual(api.SUMMARY_MAX_POINTS, 20000)
        self.assertEqual(api.PLOT_MAX_POINTS, 5000)

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


    def test_csv_rows_follow_xrange(self):
        x0 = self.sweeps[0][1][:, 0]
        lo, hi = float(np.percentile(x0, 10)), float(np.percentile(x0, 60))
        dest = self.dir / "window.csv"
        api.extract(self.path, ["v_a"], output="csv", dest=dest, xrange=(lo, hi))
        lines = dest.read_text().splitlines()
        expected = sum(int(((d[:, 0] >= lo) & (d[:, 0] <= hi)).sum()) for _, d in self.sweeps)
        self.assertEqual(lines[0], "sweep,r1,TIME,v_a")
        self.assertEqual(len(lines), 1 + expected)

    def test_png_default_dest(self):
        self.assertEqual(api.default_dest("run.tr0", "png"), "run_tr0_traces.png")

    def test_png_written(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        x0 = self.sweeps[0][1][:, 0]
        lo, hi = float(np.percentile(x0, 10)), float(np.percentile(x0, 60))
        dest = self.dir / "plot.png"
        out = api.extract(self.path, ["v_a", "i_e"], output="png", dest=dest,
                          xrange=(lo, hi), yrange=(-1.0, 1.0))
        self.assertEqual(out, str(dest))
        data = dest.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(data), 1000)

    def test_png_ac_uses_log_axis(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        ts = read_traces(HERE / "test_9601.ac0", ["v(vo)"])
        fig = api.make_figure(ts)
        self.assertEqual(fig.axes[0].get_xscale(), "log")
        self.assertEqual(len(fig.axes[0].get_lines()), 2)


    def test_csv_lead_column_without_sweep_params(self):
        d = np.arange(12.0).reshape(4, 3)
        path = self.dir / "nosp.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b"], [1, 1, 1], [([], d), ([], d + 100)])
        dest = self.dir / "nosp.csv"
        api.extract(path, output="csv", dest=dest)
        lines = dest.read_text().splitlines()
        self.assertEqual(lines[0], "sweep,TIME,v_a,v_b")
        self.assertEqual(len(lines), 9)
        self.assertTrue(lines[-1].startswith("1,"))

    def test_npz_survives_partial_sweep_parameter_prefix(self):
        names = ["TIME", "v(a", "p0", "p1"]
        d = np.arange(12.0).reshape(6, 2)
        sweeps = [([0.0, 10.0], d), ([1.0, 11.0], d), ([2.0, 12.0], np.empty((0, 2)))]
        path = self.dir / "ragged.tr0"
        fixtures.write_binary(path, "2001", names, [1, 1], sweeps, final_sentinel=False, drop_tail=1)
        dest = self.dir / "ragged.npz"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api.extract(path, output="npz", dest=dest)
        with np.load(dest) as z:
            sv = z["__sweep_values__"]
        self.assertEqual(sv.shape, (3, 2))
        self.assertEqual(sv[2, 0], 2.0)
        self.assertTrue(np.isnan(sv[2, 1]))

    def test_figure_applies_yrange(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        ts = read_traces(self.path, ["v_a"])
        fig = api.make_figure(ts, yrange=(-1.0, 1.0))
        self.assertEqual(fig.axes[0].get_ylim(), (-1.0, 1.0))

    def test_figure_decimates_long_traces(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        original = api.PLOT_MAX_POINTS
        api.PLOT_MAX_POINTS = 50
        self.addCleanup(setattr, api, "PLOT_MAX_POINTS", original)
        ts = read_traces(self.path, ["v_a"])
        fig = api.make_figure(ts)
        sizes = [line.get_xdata().size for line in fig.axes[0].get_lines()]
        self.assertEqual(sizes, [50, 50, 50])
        first = fig.axes[0].get_lines()[0]
        self.assertEqual(first.get_xdata()[0], ts.data["TIME"][0][0])
        self.assertEqual(first.get_xdata()[-1], ts.data["TIME"][0][-1])

    def test_npz_trace_names_that_collide_with_savez_parameters(self):
        d = np.arange(12.0).reshape(4, 3)
        path = self.dir / "names.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "file", "allow_pickle"], [1, 1, 1], [([], d)])
        dest = self.dir / "names.npz"
        api.extract(path, output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertEqual(set(z.files), {"TIME", "file", "allow_pickle", "__sweep_values__", "__sweep_params__"})
            np.testing.assert_array_equal(z["allow_pickle"], d[:, 2])
            self.assertEqual(z["__sweep_values__"].shape, (1, 0))

    def test_csv_header_quotes_names_with_commas(self):
        import csv
        d = np.arange(8.0).reshape(4, 2)
        path = self.dir / "diff.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a,b"], [1, 1], [([], d)])
        dest = self.dir / "diff.csv"
        ts = api.extract(path)
        self.assertIn(",", ts.selected[1])
        api.extract(path, output="csv", dest=dest)
        rows = list(csv.reader(dest.read_text().splitlines()))
        self.assertEqual(rows[0], ts.selected)
        self.assertTrue(all(len(r) == 2 for r in rows))

class TestXrangeWhileReading(unittest.TestCase):
    """read_traces(xrange=...) must equal reading everything and cutting afterwards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_matches_post_hoc_cut(self, path, lo, hi, **kw):
        """The windowed read equals the full read masked with the same closed interval."""
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
        """A window that keeps some but not all rows of the first sweep."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = read_traces(path, **kw).data[read_header(path).x_name][0]
        return float(np.percentile(x, 20)), float(np.percentile(x, 70))

    def test_sample_files(self):
        for name in ("test_9601.tr0", "test_2001.tr0", "test_9601.sw0", "test_9601.ac0"):
            path = HERE / name
            lo, hi = self.middle_window(path)
            _, win = self.assert_matches_post_hoc_cut(path, lo, hi)
            self.assertGreater(win.data[win.selected[0]][0].size, 0)

    def test_multi_sweep_and_selective_sweeps(self):
        for version in ("9601", "2001"):
            path, sweeps = make_multi(self.dir, version)
            lo, hi = self.middle_window(path)
            _, win = self.assert_matches_post_hoc_cut(path, lo, hi)
            self.assertEqual(len(win.sweep_values), 3)
            self.assert_matches_post_hoc_cut(path, lo, hi, sweeps=[0, 2])
            self.assert_matches_post_hoc_cut(path, lo, hi, names=["v_a"], sweeps=[1])

    def test_ascii(self):
        path, _ = make_ascii(self.dir)
        lo, hi = self.middle_window(path)
        self.assert_matches_post_hoc_cut(path, lo, hi)
        self.assert_matches_post_hoc_cut(path, lo, hi, names=["v_a"], sweeps=[1])

    def test_truncated_file(self):
        path, _ = make_multi(self.dir, "2001", final_sentinel=False, drop_tail=3)
        lo, hi = self.middle_window(path)
        _, win = self.assert_matches_post_hoc_cut(path, lo, hi)
        self.assertTrue(win.truncated)

    def test_window_covering_everything_matches_no_window(self):
        path, _ = make_multi(self.dir, "2001")
        plain = read_traces(path, ["v_a"])
        wide = read_traces(path, ["v_a"], xrange=(-1e30, 1e30))
        self.assertEqual(wide.sweep_indices, plain.sweep_indices)
        for i in range(3):
            for name in plain.selected:
                np.testing.assert_array_equal(wide.data[name][i], plain.data[name][i])

    def test_empty_window_keeps_the_sweeps_with_empty_arrays(self):
        path, _ = make_multi(self.dir, "2001")
        ts = read_traces(path, ["v_a"], xrange=(2.0, 3.0))       # x is in [0, 1)
        self.assertEqual(ts.sweep_indices, [0, 1, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0], [3000.0]])
        self.assertFalse(ts.truncated)
        for i in range(3):
            for name in ts.selected:
                self.assertEqual(ts.data[name][i].size, 0)

    def test_window_boundaries_are_inclusive(self):
        path, sweeps = make_multi(self.dir, "2001")
        x = sweeps[0][1][:, 0]
        lo, hi = float(np.sort(x)[5]), float(np.sort(x)[9])
        ts = read_traces(path, ["v_a"], sweeps=[0], xrange=(lo, hi))
        got = np.sort(ts.data["TIME"][0])
        self.assertEqual(got.size, 5)
        self.assertEqual(got[0], lo)
        self.assertEqual(got[-1], hi)

    def test_memory_follows_the_window(self):
        rng = np.random.default_rng(17)
        npoints, ncols = 200_000, 20
        data = rng.random((npoints, ncols))
        data[:, 0] = np.arange(npoints) * 1e-12
        path = self.dir / "wide.tr0"
        fixtures.write_binary(path, "2001", ["TIME"] + [f"v({i}" for i in range(ncols - 1)],
                              [1] * ncols, [([], data)])
        lo, hi = 0.0, float(data[npoints // 20, 0])              # 5% of the rows
        del data
        original = reader.FRAMES_PER_CHUNK
        reader.FRAMES_PER_CHUNK = 8                              # ~66 KB: isolate the buffers
        self.addCleanup(setattr, reader, "FRAMES_PER_CHUNK", original)
        peaks = []
        for xrange in (None, (lo, hi)):
            tracemalloc.start()
            tracemalloc.reset_peak()
            ts = read_traces(path, ["v(7"], xrange=xrange)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
        self.assertEqual(ts.data["v_7"][0].size, npoints // 20 + 1)
        self.assertLess(peaks[1], 0.25 * peaks[0], f"windowed {peaks[1]} vs full {peaks[0]} bytes")


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
        with self.assertRaisesRegex(self.m.ToolError, "not available over MCP"):
            self.m.extract(str(HERE / "test_9601.tr0"), output="arrays")

    def test_bad_output_rejected_before_reading(self):
        with self.assertRaisesRegex(self.m.ToolError, "summary"):
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


    def test_png_tool_returns_path_and_size(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        dest = self.dir / "o.png"
        ref = read_traces(HERE / "test_9601.tr0")
        x = ref.data["TIME"][0]
        lo, hi = float(x[100]), float(x[900])
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="png", dest=str(dest),
                             xrange=[lo, hi], yrange=[-2.0, 2.0])
        self.assertEqual(out["output"], "png")
        self.assertEqual(out["path"], str(dest))
        self.assertEqual(out["points"], [int(((x >= lo) & (x <= hi)).sum())])   # 802: a repeated time point
        self.assertEqual(len(out["size"]), 2)
        self.assertTrue(all(isinstance(v, int) and v > 0 for v in out["size"]))
        self.assertTrue(dest.exists())
        json.dumps(out)

    def test_csv_tool_applies_xrange(self):
        dest = self.dir / "w.csv"
        ref = read_traces(HERE / "test_9601.tr0")
        x = ref.data["TIME"][0]
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv", dest=str(dest),
                             xrange=[float(x[10]), float(x[19])])
        self.assertEqual(out["points"], [10])
        self.assertEqual(len(dest.read_text().splitlines()), 11)

    def test_bad_range_rejected_before_reading(self):
        with self.assertRaisesRegex(self.m.ToolError, "xrange"):
            self.m.extract(str(self.dir / "missing.tr0"), output="csv", xrange=[1.0])


    def test_error_text_reaches_the_client(self):
        with self.assertRaisesRegex(Exception, "available traces"):
            asyncio.run(self.m.server.call_tool("extract", {"path": str(HERE / "test_9601.tr0"),
                                                             "names": ["nope"], "output": "csv"}))
        with self.assertRaisesRegex(Exception, "must be one of"):
            asyncio.run(self.m.server.call_tool("extract", {"path": str(HERE / "test_9601.tr0"),
                                                             "output": "xlsx"}))

    def test_unexpected_errors_carry_their_type(self):
        original = self.m.api.list_traces
        def boom(path, plot=0):
            raise KeyError("boom")
        self.m.api.list_traces = boom
        self.addCleanup(setattr, self.m.api, "list_traces", original)
        with self.assertRaisesRegex(self.m.ToolError, "KeyError: 'boom'"):
            self.m.list_traces("whatever")

if __name__ == "__main__":
    unittest.main()
