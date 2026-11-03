"""Tests for the streaming reader, API and MCP server. Fixtures are generated in temp dirs."""
import asyncio
import dataclasses
import json
import pickle
import re
import shutil
import struct
import sys
import tempfile
import time
import tracemalloc
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402

from spice_result_parser.reader import read_header, read_traces  # noqa: E402
from spice_result_parser import reader  # noqa: E402
from spice_result_parser import api  # noqa: E402


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
            data = (HERE / "test_9601.tr0").read_bytes().replace(b"9601    *", b"9007    *", 1)
            bad.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "unsupported post_version '9007'; 9601, 2001 and 2013"):
                read_header(bad)
            # 2013 is written where 2001 is (columns 20-23), never where 9601 is
            bad.write_bytes((HERE / "test_9601.tr0").read_bytes().replace(b"9601    *", b"2013    *", 1))
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

    def test_ac_9601_sample_file(self):
        ts = read_traces(HERE / "test_9601.ac0", ac_format="realimag")
        self.assertEqual(ts.selected[:3], ["HERTZ", "v_0_Re", "v_0_Im"])
        self.assertEqual(ts.data["HERTZ"][0].size, 41)
        np.testing.assert_allclose(ts.data["v_vo_Re"][0][:2], [0.99996054, 0.9999374], rtol=1e-6)
        np.testing.assert_allclose(ts.data["v_vo_Im"][0][:2], [-0.00628294, -0.00790957], rtol=1e-6)

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

    def test_header_over_several_blocks(self):
        # A header longer than one block (about 8 KB of text: about 330 short names) goes on
        # in the next blocks; only the last one holds the terminator.
        names = ["TIME"] + [f"v(n{i}" for i in range(400)]
        data = np.arange(3 * 401, dtype=np.float64).reshape(3, 401)
        for version in ("9601", "2001"):
            path = self.dir / f"wide{version}.tr0"
            fixtures.write_binary(path, version, names, [1] * 401, [([], data)], header_bytes=1000)
            self.assertGreater(path.stat().st_size // 1000, 6)
            h = read_header(path)
            self.assertEqual(h.raw_names, names)
            ts = read_traces(path, ["v(n0", "v(n399"])
            np.testing.assert_array_equal(ts.data["v_n399"][0], data[:, 400])
            np.testing.assert_array_equal(ts.data["TIME"][0], data[:, 0])
        path = self.dir / "cut.tr0"
        path.write_bytes(fixtures._block(fixtures.header_text("9601", names, [1] * 401, 0, 0).encode()[:1000]))
        with self.assertRaisesRegex(ValueError, "terminator"):
            read_header(path)
        # The terminator split over two blocks.
        text = fixtures.header_text("9601", ["TIME", "v(a"], [1, 1], 0, 0)
        path = self.dir / "split.tr0"
        fixtures.write_binary(path, "9601", ["TIME", "v(a"], [1, 1], [([], np.ones((2, 2)))],
                              header_bytes=text.index("$&%#") + 2)
        self.assertEqual(read_header(path).raw_names, ["TIME", "v(a"])
        np.testing.assert_array_equal(read_traces(path).data["v_a"][0], [1.0, 1.0])

    def test_header_without_terminator(self):
        # A header block without $&%#, then data blocks: read as it is, as before the header could
        # span blocks, and fast: the data blocks are not read as text.
        path = self.dir / "noterm.tr0"
        text = fixtures.header_text("9601", ["TIME", "v(a"], [1, 1], 0, 0).replace("$&%#", "    ")
        values = np.arange(1, 2049, dtype="<f4")
        with open(path, "wb") as f:
            f.write(fixtures._block(text.encode()))
            for _ in range(1200):                   # about 10 MB of data blocks
                f.write(fixtures._block(values.tobytes()))
        start = time.perf_counter()
        self.assertEqual(read_header(path).raw_names, ["TIME", "v(a"])
        self.assertLess(time.perf_counter() - start, 2.0)
        with self.assertWarnsRegex(RuntimeWarning, "file ended before the sweep terminator"):
            ts = read_traces(path)
        np.testing.assert_array_equal(ts.data["TIME"][0][:3], [1.0, 3.0, 5.0])

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
        header = dataclasses.replace(read_header(path), sweep_count_hint=2)    # the 2 sweeps fed below
        c = reader._Collector(header, list(range(7)), None, capacity=8)
        sweep0 = [1000.0] + list(range(14)) + [90.0, 91.0, 92.0] + [1e30]
        sweep1 = [2000.0] + [float(v) for v in range(100, 107)] + [1e30]
        with self.assertWarnsRegex(RuntimeWarning, "sweep 0 ends with 3 values that do not make a whole "
                                                   "point of 7; they are dropped"):
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
        with self.assertRaisesRegex(ValueError, r"^unknown trace 'nope'\. Available traces: TIME, v_a, v_b"):
            read_traces(self.path, ["nope"])

    def test_case_is_ignored_when_nothing_matches_exactly(self):
        # SPICE names are case-insensitive: a netlist's V(B) is the file's v(b
        for req in ("V(B)", "V_B", "v(B", "V_*"):
            ts = read_traces(self.path, [req])
            self.assertIn("v_b", ts.selected, req)
        self.assertEqual(read_traces(self.path, ["time"]).selected, ["TIME"])
        self.assertEqual(read_traces(self.path, ["I_*"]).selected, ["TIME", "i_e", "i_f"])

    def test_exact_case_wins_over_folded_case(self):
        names = ["TIME", "v(a", "V(A"]
        path = self.dir / "cases.tr0"
        fixtures.write_binary(path, "2001", names, [1, 1, 1], [([], np.arange(9.0).reshape(3, 3))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertEqual(read_traces(path, ["V(A)"]).selected, ["TIME", "V_A"])
            self.assertEqual(read_traces(path, ["v_a"]).selected, ["TIME", "v_a"])

    def test_unknown_name_suggests_near_names(self):
        with self.assertRaisesRegex(ValueError, r"unknown trace 'v\(bb\)'; did you mean v_b\? Available"):
            read_traces(self.path, ["v(bb)"])
        with self.assertRaisesRegex(ValueError, r"unknown trace 'i_ef'; did you mean i_e, i_f\? Available"):
            read_traces(self.path, ["i_ef"])

    def test_ac_base_name_says_how_to_select_the_pair(self):
        with self.assertRaisesRegex(ValueError, r"unknown trace 'v_vo': complex traces are split in two, "
                                                r"v_vo_Mag and v_vo_Phase; 'v\(vo\)' or 'v_vo\*' selects both"):
            read_traces(HERE / "test_9601.ac0", ["v_vo"])
        with self.assertRaisesRegex(ValueError, "v_vo_dB and v_vo_Phase"):
            read_traces(HERE / "test_9601.ac0", ["V_VO"], ac_format="db")
        with self.assertRaisesRegex(ValueError, r"v_out_Mag and v_out_Phase; 'v\(out\)'"):
            read_traces(HERE / "nutmeg_rc_bin.raw", ["v_out"], plot=1)

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

    def write(self, name, raw_names):
        path = self.dir / name
        data = np.arange(3.0 * len(raw_names)).reshape(3, -1)
        fixtures.write_binary(path, "2001", raw_names, [1] * len(raw_names), [([], data)])
        return path

    def test_raw_globs_keep_to_their_level(self):
        # "." and "_" both sanitize to "_": a glob on raw names keeps to its level, one on
        # sanitized names does not.
        path = self.write("levels.tr0", ["TIME", "v(x1.a.n1", "v(x1.a.n2", "v(x1.a_b.n1", "i(x1.a.m1", "v(x2.a.n1"])
        def selected(req):
            return read_traces(path, [req]).selected[1:]
        for req in ("v(x1.a.*", "v(x1.a.*)", "V(X1.A.*"):                  # paren or not; case folded
            self.assertEqual(selected(req), ["v_x1_a_n1", "v_x1_a_n2"], req)
        self.assertEqual(selected("v_x1_a_*"), ["v_x1_a_n1", "v_x1_a_n2", "v_x1_a_b_n1"])
        self.assertEqual(selected("*(x1.a.*"), ["v_x1_a_n1", "v_x1_a_n2", "i_x1_a_m1"])
        self.assertEqual(selected("v(x?.a.n1"), ["v_x1_a_n1", "v_x2_a_n1"])

    def test_brackets_in_a_glob_match_themselves(self):
        # Bus bits: fnmatch would read [*] as a class that holds "*".
        path = self.write("bus.tr0", ["TIME", "v(d[0]", "v(d[1]", "v(d0", "v(x[2].n1"])
        def selected(req):
            return read_traces(path, [req]).selected[1:]
        self.assertEqual(selected("v(d[*]"), ["v_d[0]", "v_d[1]"])
        self.assertEqual(selected("v_d[?]"), ["v_d[0]", "v_d[1]"])
        self.assertEqual(selected("v(d[1])"), ["v_d[1]"])
        self.assertEqual(selected("v(x[2].*"), ["v_x[2]_n1"])

    def test_nutmeg_names_with_or_without_the_closing_paren(self):
        # HSPICE writes v(out and Nutmeg v(out): either form finds both, in a glob too.
        nutmeg = HERE / "nutmeg_rc_bin.raw"
        for req in ("v(out", "v(out)", "V(OUT"):
            self.assertEqual(read_traces(nutmeg, [req]).selected, ["time", "v_out"], req)
        for req in ("v(*", "v(*)"):
            self.assertEqual(read_traces(nutmeg, [req]).selected, ["time", "v_in", "v_out"], req)

    def test_an_unknown_level_suggests_the_near_branch(self):
        names = ["TIME"] + [f"v(xtop.xcore{c}.n{i}" for c in range(3) for i in range(10)] + ["v(out"]
        path = self.write("chip.tr0", names)
        with self.assertRaisesRegex(ValueError, r"^unknown trace 'v\(xtop\.xcor1\.\*'; did you mean "
                                                r"v\(xtop\.xcore1\.\*\? Available"):
            read_traces(path, ["v(xtop.xcor1.*"])
        with self.assertRaisesRegex(ValueError, r"did you mean v\(xtop\.xcore2\.n7\? "):
            read_traces(path, ["V(XTOP.XCORE2.N77"])                     # the case of the file
        with self.assertRaisesRegex(ValueError, r"did you mean v\(xtop\.\*\? "):
            read_traces(path, ["v(xtopp.xcore1.*"])
        # After the first 20 traces of a larger file, the error counts them all by name.
        with self.assertRaisesRegex(ValueError, r"\.\.\. and 12 more\. By name: v\(xtop\.xcore0\.\* \(10\), "
                                                r"v\(xtop\.xcore1\.\* \(10\), v\(xtop\.xcore2\.\* \(10\), "
                                                r"v\(out \(1\)$"):
            read_traces(path, ["nope"])


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

class TestNameGroups(unittest.TestCase):
    """reader.name_groups: counts of names by prefix, as requests that select them."""

    @staticmethod
    def header(raw_names, ac=False):
        """A header whose traces have these raw names (ac: two columns each), x first."""
        names, col_raw = ["x"], ["x"]
        for raw in raw_names:
            base = reader.parse_var_name(raw.replace(")", ""))
            names += reader.ac_names(base, "magphase") if ac else [base]
            col_raw += [raw, raw] if ac else [raw]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)             # names that sanitize alike
            names = reader._uniquify("test", names)
        return reader.Header(path="test", is_binary=True, version="2001", analysis="ac" if ac else "tr",
                             ncols=len(names), nsweepparam=0, sweep_count_hint=0, x_name="x", names=names,
                             raw_names=["x", *raw_names], col_raw_names=col_raw, type_codes=[], sweep_params=[],
                             ncomplex=len(raw_names) if ac else 0)

    def assert_partition(self, raw_names, limit=reader.GROUPS_MAX, ac=False):
        """Each key selects its count of columns, and the keys select every column once."""
        header = self.header(raw_names, ac)
        groups = reader.name_groups(header.col_raw_names[1:], limit)
        self.assertLessEqual(len(groups), max(2, limit))
        seen = set()
        for key, count in groups.items():
            cols = set(reader.resolve_columns(header, key)[1:])
            self.assertEqual(len(cols), count, key)
            self.assertFalse(cols & seen, key)
            seen |= cols
        self.assertEqual(seen, set(range(1, header.ncols)))
        return groups

    def test_the_keys_select_every_name_once(self):
        cores = [f"v(xtop.xcore{c}.n{i}" for c in range(10) for i in range(30)] + ["v(out", "i(vdd"]
        functions = [f"{f}(xtop.x{c}.m{i}" for f in ("v", "i1", "i2", "p", "vdb") for c in range(4) for i in range(8)]
        cases = {
            "cores": cores,
            "functions": functions,
            "flat": [f"v(n{i}" for i in range(500)],
            "array": [f"v(xtop.xbit{i}.n{j}" for i in range(200) for j in range(3)] + ["v(xtop.xctl.en"],
            "bare": ["vo", "xt.vo", "outnoise", "top.u1.clk", "top.u1.d", "top/u2/d", "top/u2/q"]
                    + [f"top.u3.s{i}" for i in range(60)],
            "ltspice": [f"V(xu{u}:n{i:03d}" for u in range(5) for i in range(20)] + ["I(R1", "V(n002"],
            "bus": [f"v(d[{i}]" for i in range(100)] + [f"v(x[{k}].n1" for k in range(4)],
            "nutmeg": [f"v(x{k}.n{i})" for k in range(6) for i in range(12)] + ["i(v1)"],
            "cases": ["v(a", "V(B", "v(c.d", "V(C.E"],
        }
        for name, raw in cases.items():
            for limit in (2, 10, 50):
                with self.subTest(name, limit=limit):
                    self.assert_partition(raw, limit)
        with self.subTest("ac"):
            groups = self.assert_partition(cores, ac=True)
            self.assertEqual(groups["v(xtop.xcore0.*"], 60)            # a count is of columns

    def test_the_groups_follow_the_hierarchy(self):
        # Below the part that all names share (v(xtop.), the largest groups split while
        # the limit allows.
        cores = [f"v(xtop.xcore{c}.n{i}" for c in range(10) for i in range(300)] + ["v(out", "i(vdd"]
        self.assertEqual(reader.name_groups(cores), {**{f"v(xtop.xcore{c}.*": 300 for c in range(10)},
                                                      "v(out": 1, "i(vdd": 1})
        self.assertEqual(reader.name_groups(cores, 10), {"v(xtop.*": 3000, "v(out": 1, "i(vdd": 1})
        # A group of names without levels below it does not split into names.
        small = [f"v(xtop.xcore{c}.n{i}" for c in range(2) for i in range(3)] + [f"i(m{i}" for i in range(4)]
        self.assertEqual(reader.name_groups(small), {"v(xtop.xcore0.*": 3, "v(xtop.xcore1.*": 3, "i(*": 4})

    def test_groups_of_one_count_split_together(self):
        # Three like cores stay three groups: splitting one by characters and not the
        # others would hide that they are alike.
        cores = [f"v(xtop.xcore{c}.n{i}" for c in range(3) for i in range(300)] + ["v(out"]
        self.assertEqual(reader.name_groups(cores), {"v(xtop.xcore0.*": 300, "v(xtop.xcore1.*": 300,
                                                      "v(xtop.xcore2.*": 300, "v(out": 1})

    def test_a_wide_level_splits_by_characters(self):
        # 1000 instances do not fit: their names split by characters, a level when it can.
        array = [f"v(xtop.xbit{i}.n{j}" for i in range(1000) for j in range(6)] + ["v(xtop.xctl.en"]
        groups = reader.name_groups(array)
        self.assertEqual(list(groups)[:3], ["v(xtop.xbit0.*", "v(xtop.xbit1*", "v(xtop.xbit2*"])
        self.assertEqual((groups["v(xtop.xbit0.*"], groups["v(xtop.xbit1*"]), (6, 666))
        inner = reader.name_groups([n for n in array if n.startswith("v(xtop.xbit1")])
        self.assertEqual(list(inner)[:3], ["v(xtop.xbit1.*", "v(xtop.xbit10*", "v(xtop.xbit11*"])
        # A flat list is the only group, so it splits by characters too.
        flat = reader.name_groups([f"v(n{i}" for i in range(3000)])
        self.assertEqual(list(flat.items())[:3], [("v(n0", 1), ("v(n1*", 1111), ("v(n2*", 1111)])
        # A wide level of leaves beside other groups does not.
        leaves = [f"v(xtop.n{i}" for i in range(300)] + ["i(vdd"]
        self.assertEqual(reader.name_groups(leaves), {"v(xtop.*": 300, "i(vdd": 1})

    def test_the_groups_keep_the_order_of_the_names(self):
        raw = ["v(zz.n1", "v(aa.n1", "v(zz.n2", "v(mm.n1", "v(aa.n2", "v(mm.n2"]
        self.assertEqual(reader.name_groups(raw), {"v(zz.*": 2, "v(aa.*": 2, "v(mm.*": 2})
        self.assertEqual(reader.name_groups([]), {})
        self.assertEqual(reader.name_groups(["v(a"]), {"v(a": 1})

    def test_groups_are_fast_for_many_names(self):
        raw = [f"v(xtop.xcore{c}.xblk{c % 7}.n{i}" for c in range(100) for i in range(2000)]
        start = time.perf_counter()
        groups = reader.name_groups(raw)
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertEqual(sum(groups.values()), len(raw))


AC_NAMES = ["HERTZ", "v(a", "i(b"]
AC_CODES = [2, 1, 8]


def make_ac(directory, version="2001", name="ac.ac0"):
    """AC fixture: HSPICE stores each complex value as a (real, imaginary) pair."""
    rng = np.random.default_rng(21)
    f = np.logspace(0, 6, 31)
    z = rng.uniform(-2, 2, (31, 2)) + 1j * rng.uniform(-2, 2, (31, 2))
    data = np.column_stack([f, z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag])
    path = Path(directory) / name
    fixtures.write_binary(path, version, AC_NAMES, AC_CODES, [([], data)])
    return path, f, z


class TestAcFormat(unittest.TestCase):
    """HSPICE AC data are (real, imaginary) pairs; ac_format chooses what the pair becomes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.f, self.z = make_ac(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_magnitude_and_phase_in_degrees(self):
        ts = read_traces(self.path)
        self.assertEqual(ts.selected, ["HERTZ", "v_a_Mag", "v_a_Phase", "i_b_Mag", "i_b_Phase"])
        np.testing.assert_allclose(ts.data["v_a_Mag"][0], np.abs(self.z[:, 0]), rtol=1e-12)
        np.testing.assert_allclose(ts.data["v_a_Phase"][0], np.angle(self.z[:, 0], deg=True), rtol=1e-12)
        np.testing.assert_allclose(ts.data["i_b_Mag"][0], np.abs(self.z[:, 1]), rtol=1e-12)
        np.testing.assert_array_equal(ts.data["HERTZ"][0], self.f)

    def test_realimag_returns_the_stored_parts(self):
        ts = read_traces(self.path, ac_format="realimag")
        self.assertEqual(ts.selected, ["HERTZ", "v_a_Re", "v_a_Im", "i_b_Re", "i_b_Im"])
        np.testing.assert_array_equal(ts.data["v_a_Re"][0], self.z[:, 0].real)
        np.testing.assert_array_equal(ts.data["i_b_Im"][0], self.z[:, 1].imag)

    def test_db(self):
        ts = read_traces(self.path, ["v(a"], ac_format="db")
        self.assertEqual(ts.selected, ["HERTZ", "v_a_dB", "v_a_Phase"])
        np.testing.assert_allclose(ts.data["v_a_dB"][0], 20 * np.log10(np.abs(self.z[:, 0])), rtol=1e-12)
        np.testing.assert_allclose(ts.data["v_a_Phase"][0], np.angle(self.z[:, 0], deg=True), rtol=1e-12)

    def test_one_half_of_a_pair_still_uses_both_stored_parts(self):
        ts = read_traces(self.path, ["v_a_Phase"])
        self.assertEqual(ts.selected, ["HERTZ", "v_a_Phase"])
        np.testing.assert_allclose(ts.data["v_a_Phase"][0], np.angle(self.z[:, 0], deg=True), rtol=1e-12)

    def test_xrange_applies_before_conversion(self):
        ts = read_traces(self.path, ["v(a"], xrange=(10.0, 1e4))
        keep = (self.f >= 10.0) & (self.f <= 1e4)
        np.testing.assert_allclose(ts.data["v_a_Mag"][0], np.abs(self.z[keep, 0]), rtol=1e-12)

    def test_9601_keeps_float32(self):
        path, _, z = make_ac(self.dir, "9601", "ac32.ac0")
        ts = read_traces(path)
        self.assertEqual(ts.data["v_a_Mag"][0].dtype, np.dtype("<f4"))
        np.testing.assert_allclose(ts.data["v_a_Mag"][0], np.abs(z[:, 0]), rtol=1e-5)

    def test_header_and_list_traces_follow_the_format(self):
        self.assertEqual(read_header(self.path, ac_format="db").names[1:3], ["v_a_dB", "v_a_Phase"])
        self.assertEqual(api.list_traces(self.path, ac_format="realimag")["traces"],
                         ["v_a_Re", "v_a_Im", "i_b_Re", "i_b_Im"])

    def test_extract_passes_the_format(self):
        ts = api.extract(self.path, ["i(b"], ac_format="realimag")
        self.assertEqual(ts.selected, ["HERTZ", "i_b_Re", "i_b_Im"])

    def test_unknown_format_rejected(self):
        with self.assertRaisesRegex(ValueError, "ac_format must be one of"):
            read_traces(self.path, ac_format="polar")
        with self.assertRaisesRegex(ValueError, "ac_format must be one of"):
            read_header(self.path, ac_format="polar")

    def test_non_ac_file_ignores_the_format(self):
        ts = read_traces(HERE / "test_9601.tr0", ["v_vo"], ac_format="db")
        self.assertEqual(ts.selected, ["TIME", "v_vo"])

    def test_sample_file_is_an_rc_lowpass(self):
        # test_9601.ac0 is a first-order RC low-pass: H = 1 / (1 + j f/fc), so Re/Im gives
        # the same fc at every frequency, the magnitude is never negative, and the phase
        # heads to -90 degrees. Read as magnitude/phase the stored pairs fail all three.
        ri = read_traces(HERE / "test_9601.ac0", ["v(vo"], ac_format="realimag")
        f = ri.data["HERTZ"][0].astype(np.float64)
        re, im = (ri.data[n][0].astype(np.float64) for n in ("v_vo_Re", "v_vo_Im"))
        fc = -re * f / im
        self.assertLess(np.std(fc) / np.mean(fc), 1e-3)
        ts = read_traces(HERE / "test_9601.ac0")
        self.assertTrue((ts.data["i_vs_Mag"][0] >= 0).all())
        self.assertLess(ts.data["v_vo_Phase"][0][-1], -85.0)
        np.testing.assert_allclose(ts.data["v_vo_Mag"][0], 1 / np.sqrt(1 + (f / fc.mean()) ** 2), rtol=1e-3)


class TestHeaderReadsOnlyTheHead(unittest.TestCase):
    def test_ascii_header_does_not_scan_the_data(self):
        import builtins

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.tr0"
            t = np.linspace(0, 1e-6, 100_000)
            fixtures.write_ascii(path, ["TIME", "v(out"], [1, 1], [([], np.column_stack([t, t]))])
            size = path.stat().st_size
            total = [0]
            real_open = builtins.open

            class Counting:
                def __init__(self, f):
                    self.f = f

                def read(self, *a):
                    data = self.f.read(*a)
                    total[0] += len(data)
                    return data

                def readline(self, *a):
                    data = self.f.readline(*a)
                    total[0] += len(data)
                    return data

                def __getattr__(self, name):
                    return getattr(self.f, name)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self.f.close()

            builtins.open = lambda *a, **k: Counting(real_open(*a, **k))
            try:
                h = read_header(path)
            finally:
                builtins.open = real_open
            self.assertEqual(h.names, ["TIME", "v_out"])
            self.assertGreater(size, 2_000_000)
            self.assertLess(total[0], 10_000)



ASCII_NAMES = ["TIME", "v(a", "v(b", "i(c", "r1"]
ASCII_CODES = [1, 1, 1, 8]


def make_ascii(directory):
    rng = np.random.default_rng(2)
    sweeps = [([1000.0], rng.uniform(-1, 1, (7, 4))), ([2000.0], rng.uniform(-1, 1, (5, 4)))]
    path = Path(directory) / "ascii.tr0"
    fixtures.write_ascii(path, ASCII_NAMES, ASCII_CODES, sweeps)
    return path, sweeps


class TestStatsIgnoreNan(unittest.TestCase):
    """One NaN in a trace must not blank its min/max/mean in summary and text output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nan.tr0"
        d = np.column_stack([np.arange(5.0), [1.0, np.nan, -2.0, 4.0, 0.5]])
        fixtures.write_binary(self.path, "2001", ["TIME", "v(a"], [1, 1], [([], d)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary(self):
        entry = api.summarize(read_traces(self.path))["stats"]["v_a"][0]
        self.assertEqual((entry["min"], entry["max"], entry["mean"]), (-2.0, 4.0, 0.875))

    def test_text(self):
        from spice_result_parser import mcp_server

        text = mcp_server.format_text(read_traces(self.path))
        self.assertIn(["v_a", "-2", "4"], [line.split() for line in text.splitlines()])

    def test_all_nan_is_null_without_warning(self):
        d = np.column_stack([np.arange(3.0), [np.nan] * 3])
        fixtures.write_binary(self.path, "2001", ["TIME", "v(a"], [1, 1], [([], d)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            entry = api.summarize(read_traces(self.path))["stats"]["v_a"][0]
        self.assertIsNone(entry["min"])


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
            "path": str(HERE / "test_9601.tr0"), "format": "9601", "analysis": "tr", "plot": 0, "plots": [],
            "x": "TIME", "x_range": [0.0, 0.01], "points": 2605, "sweeps": 1, "sweep_params": [],
            "sweep_values": [[]], "truncated": False, "trace_count": 4, "traces": ["v_0", "v_vo", "v_vs", "i_vs"],
            "ranges": {"v_0": [0.0, 0.0], "v_vo": [0.0, 1.8477405e-05], "v_vs": [0.0, 5.0],
                       "i_vs": [-0.005, 1.8477404e-08]},
        })
        json.dumps(info, allow_nan=False)
        info = api.list_traces(self.path)
        self.assertEqual(info["sweep_params"], ["r1"])
        self.assertEqual(info["sweeps"], 3)
        self.assertEqual(info["sweep_values"], [[1000.0], [2000.0], [3000.0]])
        self.assertEqual(info["points"], sum(d.shape[0] for _, d in self.sweeps))
        everything = np.concatenate([d for _, d in self.sweeps])
        self.assertEqual(info["x_range"], [everything[:, 0].min(), everything[:, 0].max()])
        for col, name in enumerate(info["traces"], 1):
            self.assertEqual(info["ranges"][name], [everything[:, col].min(), everything[:, col].max()])

    def test_list_traces_header_only(self):
        info = api.list_traces(self.path, header_only=True)
        self.assertEqual(list(info), ["path", "format", "analysis", "plot", "plots", "x", "sweeps",
                                      "sweep_params", "trace_count", "traces"])
        self.assertEqual(info["sweeps"], 3)                                 # what the header announces
        self.assertEqual(api.list_traces(HERE / "test_9601.tr0", header_only=True)["sweeps"], 1)
        with mock.patch.object(api, "scan", side_effect=AssertionError("data read")):
            api.list_traces(self.path, header_only=True)

    def test_list_traces_names_narrow_traces_and_ranges(self):
        info = api.list_traces(self.path, names=["v(b", "i_*"])
        self.assertEqual(info["traces"], ["v_b", "i_e", "i_f"])
        self.assertEqual(list(info["ranges"]), ["v_b", "i_e", "i_f"])
        self.assertEqual(api.list_traces(self.path, names="TIME")["traces"], [])
        self.assertEqual(list(reader.scan(self.path, names=["v_c"]).ranges), ["v_c"])
        # an unknown name fails before the data is read
        with mock.patch.object(reader, "_feed_data", side_effect=AssertionError("data read")):
            with self.assertRaisesRegex(ValueError, "did you mean v_a"):
                api.list_traces(self.path, names=["v_aa"])
        from spice_result_parser import nutmeg
        with mock.patch.object(nutmeg, "_read_data", side_effect=AssertionError("data read")):
            with self.assertRaisesRegex(ValueError, "did you mean v_out"):
                api.list_traces(HERE / "nutmeg_rc_bin.raw", names=["v_outt"])

    def chip(self):
        """A transient file of three cores of 40 nodes and v(out: 121 traces."""
        names = ["TIME"] + [f"v(xtop.xcore{c}.n{i}" for c in range(3) for i in range(40)] + ["v(out"]
        data = np.random.default_rng(8).uniform(-1, 1, (6, len(names)))
        data[:, 0] = np.arange(6) * 1e-9
        path = self.dir / "chip.tr0"
        fixtures.write_binary(path, "2001", names, [1] * len(names), [([], data)])
        return path

    def test_list_traces_lists_at_most_max_traces(self):
        path = self.chip()
        whole = api.list_traces(path)
        self.assertNotIn("groups", whole)                               # no limit: every name, no groups
        info = api.list_traces(path, max_traces=50)
        self.assertEqual(list(info), ["path", "format", "analysis", "plot", "plots", "x", "x_range", "points",
                                      "sweeps", "sweep_params", "sweep_values", "truncated", "trace_count",
                                      "groups", "traces", "ranges", "note"])
        self.assertEqual(info["trace_count"], 121)
        self.assertEqual(info["traces"], whole["traces"][:50])
        self.assertEqual(info["ranges"], {name: whole["ranges"][name] for name in info["traces"]})
        self.assertEqual(info["groups"], {"v(xtop.xcore0.*": 40, "v(xtop.xcore1.*": 40, "v(xtop.xcore2.*": 40,
                                          "v(out": 1})
        self.assertEqual(info["note"], "traces lists the first 50 of 121 traces, and groups counts them all by "
                                       "name: pass a key of groups as names (to list_traces or extract) for its "
                                       "traces")
        head = api.list_traces(path, max_traces=50, header_only=True)
        self.assertEqual((head["groups"], head["traces"]), (info["groups"], info["traces"]))
        self.assertEqual(api.list_traces(path, max_traces=121), {**whole, "trace_count": 121})
        for bad in (0, -1, True, 2.5, "10"):
            with self.assertRaisesRegex(ValueError, "max_traces must be"):
                api.list_traces(path, max_traces=bad)

    def test_an_agent_walks_from_the_whole_file_to_one_block(self):
        # Each call lists at most 20 names: the groups lead to the names wanted, and every
        # key selects in extract the traces that list_traces counts.
        path = self.chip()
        info = api.list_traces(path, max_traces=20)
        key = "v(xtop.xcore1.*"
        self.assertEqual(info["groups"][key], 40)
        block = api.list_traces(path, names=key, max_traces=20)
        self.assertEqual(block["trace_count"], 40)
        self.assertEqual(block["groups"], {"v(xtop.xcore1.n0": 1, "v(xtop.xcore1.n1*": 11, "v(xtop.xcore1.n2*": 11,
                                           "v(xtop.xcore1.n3*": 11, **{f"v(xtop.xcore1.n{i}": 1 for i in range(4, 10)}})
        for sub, count in block["groups"].items():
            listed = api.list_traces(path, names=sub, max_traces=20)
            self.assertEqual((listed["trace_count"], len(listed["traces"])), (count, count), sub)
            self.assertNotIn("groups", listed)
            self.assertEqual(api.extract(path, sub, output="summary")["traces"], listed["traces"], sub)
        ts = api.extract(path, "v(xtop.xcore1.n1*")
        self.assertEqual(ts.selected[1:3], ["v_xtop_xcore1_n1", "v_xtop_xcore1_n10"])

    def test_list_traces_reads_only_the_header_of_a_large_file(self):
        size = self.path.stat().st_size
        cost = api.read_cost(self.path, scan=True)
        facts = (f"the file has {api._size(size)}, about {api._amount(cost.points)} points, more than the 1 kB "
                 f"that list_traces reads by default: a pass over the data goes through all of it for any names, "
                 f"sweeps and xrange, {api.read_time(cost)}")
        with mock.patch.object(api, "SCAN_MAX_BYTES", 1000), \
                mock.patch.object(api, "scan", side_effect=AssertionError("data read")):
            info = api.list_traces(self.path)
        self.assertEqual(list(info), ["path", "format", "analysis", "plot", "plots", "x", "sweeps",
                                      "sweep_params", "trace_count", "traces", "note"])
        self.assertEqual(info["note"], f"list_traces read only the header: {facts}; header_only=False makes it, for "
                                       f"x_range, points, sweep_values, truncated and ranges")
        with mock.patch.object(api, "SCAN_MAX_BYTES", 1000):
            asked = api.list_traces(self.path, header_only=False)                  # asked for: read it all
            self.assertEqual((asked["note"], "ranges" in asked), (facts, True))
            self.assertEqual(api.list_traces(self.path, header_only=True)["note"], facts)   # told it is large
        self.assertNotIn("note", api.list_traces(self.path))                        # small: read, no note
        self.assertEqual([api._size(n) for n in (999, 1000, 1_500_000, 10 ** 11)],
                         ["999 bytes", "1 kB", "1.5 MB", "100 GB"])

    def test_read_cost_comes_from_the_header(self):
        rng = np.random.default_rng(4)
        files = {"2001": make_multi(self.dir, "2001")[0], "9601": make_multi(self.dir, "9601")[0],
                 "ascii": self.dir / "a.tr0"}
        fixtures.write_ascii(files["ascii"], ASCII_NAMES, ASCII_CODES, [([1.0], rng.random((3000, 4)))])
        nutmeg = self.dir / "n.raw"
        fixtures.write_nutmeg(nutmeg, [("Transient Analysis", ["time", "v(a)", "v(b)"], rng.random((3000, 3)),
                                        False)], binary=False)
        files["nutmeg ascii"] = nutmeg
        from spice_result_parser import nutmeg as nut
        for kind, path in files.items():
            with self.subTest(kind), mock.patch.object(reader, "_feed_data", side_effect=AssertionError("read")), \
                    mock.patch.object(nut, "_read_data", side_effect=AssertionError("read")):
                cost = api.read_cost(path, names=["TIME", "v_a"] if kind != "nutmeg ascii" else ["v(a)"])
            points = reader.scan(path).points
            self.assertAlmostEqual(cost.points / points, 1, delta=0.1)                  # from the size alone
            header = reader.read_header(path)
            self.assertEqual(cost.memory, cost.points * (header.x_dtype.itemsize + header.dtype.itemsize))
            self.assertEqual((cost.engine, cost.traces, cost.values, cost.large), ("python", 1, 0, False))
            self.assertLessEqual(cost.seconds[0], cost.seconds[1])
            self.assertEqual(cost.bytes, Path(path).stat().st_size)
        with mock.patch.object(api, "SCAN_MAX_BYTES", 1000):
            self.assertTrue(api.read_cost(files["2001"]).large)
        with self.assertRaisesRegex(ValueError, "did you mean v_a"):               # names fail first
            api.read_cost(files["2001"], names="v_aa")

    def test_read_time_in_words(self):
        def cost(seconds, engine="python"):
            return api.ReadCost(engine, 10 ** 11, 10 ** 9, 2, 0, 8 * 10 ** 9, seconds, True)
        disk = "(the file in memory, or on a disk that reads 200 MB/s)"
        self.assertEqual(api.read_time(cost((67, 500))), f"about 67 s to 8 min {disk}")
        self.assertEqual(api.read_time(cost((130, 290))), f"about 2 to 5 min {disk}")
        self.assertEqual(api.read_time(cost((13, 13))), "about 13 s")
        self.assertEqual(api.read_time(cost((0.2, 0.4))), "about 1 s")
        self.assertEqual(api.read_time(cost((9000, 9000))), "about 2.5 h")
        self.assertEqual([api._amount(n) for n in (3200, 2_500_000, 1_250_000_000)], ["3200", "2.5e6", "1.25e9"])

    def test_list_traces_and_summary_share_their_keys(self):
        info = api.list_traces(self.path)
        summary = api.extract(self.path, output="summary")
        shared = ["path", "format", "analysis", "plot", "plots", "x", "x_range", "points", "sweeps",
                  "sweep_params", "sweep_values", "truncated", "traces", "ranges"]
        self.assertEqual([k for k in info if k in shared], shared)
        self.assertEqual([k for k in summary if k in shared], shared)
        for key in shared:
            self.assertEqual(info[key], summary[key], key)              # whole file: same facts
        self.assertEqual(list(summary)[-1], "stats")

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
        self.assertEqual(s["traces"], ["v_a"])
        self.assertEqual(list(s["stats"]), ["v_a"])
        everything = np.concatenate([d for _, d in self.sweeps])
        self.assertEqual(s["points"], everything.shape[0])
        self.assertEqual(s["x_range"], [everything[:, 0].min(), everything[:, 0].max()])
        self.assertEqual(s["ranges"], {"v_a": [everything[:, 1].min(), everything[:, 1].max()]})
        entry = s["stats"]["v_a"][1]
        col = self.sweeps[1][1][:, 1]
        self.assertEqual(entry["count"], 977)
        self.assertEqual(entry["min"], float(col.min()))
        self.assertEqual(entry["max"], float(col.max()))
        self.assertAlmostEqual(entry["mean"], float(col.mean()), places=12)
        self.assertEqual(entry["first"], float(col[0]))
        self.assertEqual(entry["last"], float(col[-1]))
        # random data cannot be followed in 4 points: a peak-detect envelope of 2 slices instead
        self.assertNotIn("x", entry)
        env = entry["envelope"]
        self.assertEqual([len(env[k]) for k in ("from", "to", "min", "max")], [2, 2, 2, 2])
        x = self.sweeps[1][1][:, 0]
        self.assertAlmostEqual(env["from"][0], float(x.min()), places=5)
        self.assertAlmostEqual(env["to"][-1], float(x.max()), places=5)
        self.assertEqual((min(env["min"]), max(env["max"])), (float(col.min()), float(col.max())))
        json.dumps(s)   # must be JSON serialisable

    def test_summary_without_downsample_has_no_points(self):
        s = api.extract(self.path, ["v_a"], output="summary")
        self.assertNotIn("x", s["stats"]["v_a"][0])

    def test_summary_on_sample_file(self):
        s = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="summary", downsample=10)
        self.assertEqual(s["stats"]["v_vo"][0]["count"], 2605)
        self.assertEqual(len(s["stats"]["v_vo"][0]["envelope"]["max"]), 5)    # rippled: 10 points, 5 slices
        followed = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="summary", tolerance=0.05)
        self.assertLessEqual(followed["stats"]["v_vo"][0]["max_error"], 0.05)

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
            self.assertEqual(out["stats"]["v_a"][i]["count"], int(mask.sum()))
        inside = np.concatenate([d[(d[:, 0] >= lo) & (d[:, 0] <= hi)] for _, d in self.sweeps])
        self.assertEqual(out["points"], inside.shape[0])
        self.assertEqual(out["x_range"], [inside[:, 0].min(), inside[:, 0].max()])
        self.assertEqual(out["ranges"]["v_a"], [inside[:, 1].min(), inside[:, 1].max()])

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
        self.assertEqual(len(out["stats"]["v_a"][2]["envelope"]["min"]), 100 // 3 // 2)   # budget shared by 3 sweeps
        self.assertEqual(out["stats"]["v_a"][2]["count"], 2310)

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
        self.assertIsNone(out["stats"]["v_a"][0]["y"][2])
        self.assertEqual(out["stats"]["v_a"][0]["min"], 1.0)  # statistics skip the NaN

    def test_summary_cap_is_a_total_budget(self):
        original = api.SUMMARY_MAX_POINTS
        api.SUMMARY_MAX_POINTS = 60                             # 2 traces x 3 sweeps -> 10 points each
        self.addCleanup(setattr, api, "SUMMARY_MAX_POINTS", original)
        out = api.extract(self.path, ["v_a", "v_b"], output="summary", downsample=1000)
        sizes = [len(e["envelope"]["max"]) for name in ("v_a", "v_b") for e in out["stats"][name]]
        self.assertEqual(sizes, [5] * 6)                         # random data: 10 points become 5 slices

    def test_default_caps(self):
        self.assertEqual(api.SUMMARY_MAX_POINTS, 1000)
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
        self.assertTrue(all(8 < s <= 50 for s in sizes), sizes)
        first = fig.axes[0].get_lines()[0]
        self.assertEqual(first.get_xdata()[0], ts.data["TIME"][0][0])
        self.assertEqual(first.get_xdata()[-1], ts.data["TIME"][0][-1])
        for line, y in zip(fig.axes[0].get_lines(), ts.data["v_a"]):     # the envelope keeps the extremes
            self.assertEqual((line.get_ydata().min(), line.get_ydata().max()), (y.min(), y.max()))

    def test_figure_keeps_a_one_sample_spike(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        t = np.linspace(0, 1e-6, 100_001)
        v = np.zeros_like(t)
        v[54_321] = 1.0                                          # narrower than any stride
        path = self.dir / "spike.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a"], [1, 1], [([], np.c_[t, v])])
        line = api.make_figure(read_traces(path)).axes[0].get_lines()[0]
        self.assertLessEqual(line.get_xdata().size, api.PLOT_MAX_POINTS)
        self.assertEqual(line.get_ydata().max(), 1.0)
        self.assertIn(t[54_321], line.get_xdata())

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


class TestScan(unittest.TestCase):
    """reader.scan: one pass that keeps none of the data; and the plot extent read_traces records."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_scan_matches_read(self, path, **kw):
        s = reader.scan(path, **kw)
        ts = read_traces(path, **kw)
        x = ts.header.x_name
        self.assertEqual(s.header.names, ts.header.names)
        self.assertEqual(s.sweep_values, ts.sweep_values)
        self.assertEqual(s.sweep_points, [a.size for a in ts.data[x]])
        self.assertEqual(s.points, ts.plot_points)
        whole = {name: np.concatenate(ts.data[name]) for name in ts.selected}
        np.testing.assert_array_equal(s.x_range, (np.nanmin(whole[x]), np.nanmax(whole[x])))
        np.testing.assert_array_equal(s.x_range, ts.plot_x_range)
        self.assertEqual(list(s.ranges), ts.selected[1:])
        for name in ts.selected[1:]:
            np.testing.assert_array_equal(s.ranges[name], (np.nanmin(whole[name]), np.nanmax(whole[name])),
                                          err_msg=name)
            self.assertEqual(s.ranges[name][0].dtype, ts.data[name][0].dtype)   # bounds keep the dtype
        self.assertEqual(s.truncated, ts.truncated)
        return s

    def test_sample_files_in_every_ac_format(self):
        for name in ("test_9601.tr0", "test_2001.tr0", "test_9601.sw0", "test_9601.ac0"):
            for ac_format in reader.AC_FORMATS:
                with self.subTest(name=name, ac_format=ac_format):
                    self.assert_scan_matches_read(HERE / name, ac_format=ac_format)

    def test_sweeps_binary_and_ascii(self):
        for version in ("9601", "2001"):
            path, _ = make_multi(self.dir, version)
            self.assertEqual(self.assert_scan_matches_read(path).sweep_points, [1500, 977, 2310])
        path, _ = make_ascii(self.dir)
        self.assertEqual(self.assert_scan_matches_read(path).sweep_values, [[1000.0], [2000.0]])
        path, _, _ = make_ac(self.dir)
        for ac_format in reader.AC_FORMATS:
            self.assert_scan_matches_read(path, ac_format=ac_format)

    def test_truncated_file(self):
        path, _ = make_multi(self.dir, "2001", final_sentinel=False, drop_tail=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(self.assert_scan_matches_read(path).truncated)
        self.assertTrue(any("partial" in str(w.message) for w in caught))

    def test_nan_is_skipped_and_an_all_nan_trace_has_no_range(self):
        d = np.column_stack([np.arange(4.0), [1.0, np.nan, -2.0, 4.0], [np.nan] * 4])
        path = self.dir / "nan.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b"], [1, 1, 1], [([], d)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = reader.scan(path)
            info = api.list_traces(path)
        self.assertEqual(s.ranges["v_a"], (-2.0, 4.0))
        self.assertTrue(np.isnan(s.ranges["v_b"]).all())
        self.assertEqual(info["ranges"], {"v_a": [-2.0, 4.0], "v_b": None})
        json.dumps(info, allow_nan=False)

    def test_file_without_data(self):
        path = self.dir / "empty.tr0"
        text = fixtures.header_text("2001", ["TIME", "v(a"], [1, 1], 0, 0)
        path.write_bytes(fixtures._block(text.encode("utf-8")))
        s = reader.scan(path)
        self.assertEqual((s.sweep_points, s.points, s.truncated), ([], 0, False))
        self.assertTrue(np.isnan(s.x_range).all())
        info = api.list_traces(path)
        self.assertEqual((info["sweeps"], info["x_range"], info["ranges"]), (0, None, {"v_a": None}))

    def test_memory_does_not_follow_the_file(self):
        data = np.random.default_rng(1).random((400_000, 8))
        path = self.dir / "big.tr0"
        fixtures.write_binary(path, "2001", ["TIME"] + [f"v(n{i}" for i in range(7)], [1] * 8, [([], data)])
        tracemalloc.start()
        try:
            s = reader.scan(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(s.points, 400_000)
        self.assertLess(peak, data.nbytes / 4, f"scan peak {peak} bytes for {data.nbytes} bytes of data")

    def test_read_traces_records_the_plot_extent(self):
        path, sweeps = make_multi(self.dir, "2001")
        whole = np.concatenate([d for _, d in sweeps])
        ts = read_traces(path, ["v_a"], sweeps=[1], xrange=(0.2, 0.3))
        self.assertEqual(ts.plot_points, whole.shape[0])               # every sweep, no window
        self.assertEqual(ts.plot_x_range, (whole[:, 0].min(), whole[:, 0].max()))
        self.assertEqual(ts.xrange, (0.2, 0.3))
        self.assertIsNone(read_traces(path, ["v_a"]).xrange)

    def test_float32_values_are_written_short_in_json(self):
        data = np.array([[0.0, 0.1], [1e-9, 0.7], [2e-9, 0.3]])
        path = self.dir / "short.tr0"
        fixtures.write_binary(path, "9601", ["TIME", "v(a", "vdd"], [1, 1], [([1.1], data)])
        summary = api.extract(path, output="summary", downsample=3)
        self.assertEqual(summary["sweep_values"], [[1.1]])             # not 1.100000023841858
        self.assertEqual(summary["x_range"], [0.0, 2e-09])
        self.assertEqual(summary["ranges"], {"v_a": [0.1, 0.7]})
        self.assertEqual(summary["stats"]["v_a"][0]["y"], [0.1, 0.7, 0.3])
        self.assertEqual(api.list_traces(path)["ranges"], {"v_a": [0.1, 0.7]})
        # still exact: the short decimal reads back as the very float32 the file holds
        self.assertEqual(np.float32(summary["stats"]["v_a"][0]["max"]), read_traces(path).data["v_a"][0].max())
        self.assertEqual(api._json_floats(np.array([np.nan, np.inf, 1.5], np.float32)), [None, None, 1.5])


class TestFilesCutAnywhere(unittest.TestCase):
    """A file the simulator is still writing can end at any byte: the complete points are
    kept and `truncated` is set; nothing half-written becomes data."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, path, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return read_traces(path, **kw), api.list_traces(path)

    def test_cut_inside_a_block_head(self):
        path = self.dir / "run.tr0"
        data = np.random.default_rng(0).random((2000, 2))
        fixtures.write_binary(path, "2001", ["TIME", "v(out"], [1, 1], [([], data)])
        raw = path.read_bytes()
        first = 20 + struct.unpack("<i", raw[12:16])[0]
        boundary = first + 2 * (16 + 8192 + 4)                    # after two whole data blocks
        for extra in range(1, 16):                                 # every byte of the third block's head
            path.write_bytes(raw[:boundary + extra])
            ts, info = self.read(path)
            self.assertTrue(ts.truncated, extra)
            self.assertEqual(ts.data["v_out"][0].size, 1024)
            self.assertTrue(info["truncated"])
            np.testing.assert_array_equal(ts.data["v_out"][0], data[:1024, 1])

    def test_cut_just_after_a_sweep_terminator(self):
        rng = np.random.default_rng(3)
        sweeps = [([1000.0], rng.random((100, 3))), ([2000.0], rng.random((100, 3))), ([3000.0], rng.random((100, 3)))]
        path = self.dir / "sweep.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b", "r1"], [1, 1, 1], sweeps)
        raw = path.read_bytes()
        start = 20 + struct.unpack("<i", raw[12:16])[0] + 16
        terminator = start + 302 * 8                                # sweep 0: 1 parameter, 300 values, 1e30
        for cut in (terminator + 3, terminator):                    # mid-value, and exactly at the terminator
            path.write_bytes(raw[:cut])
            ts, info = self.read(path)
            self.assertEqual(len(ts.sweep_indices), 1)
            self.assertTrue(ts.truncated, cut - terminator)        # 1 of the 3 announced sweeps
            self.assertEqual((info["sweeps"], info["truncated"]), (1, True))
        apath = self.dir / "sweep_a.tr0"
        fixtures.write_ascii(apath, ["TIME", "v(a", "r1"], [1, 1], [([1.0], rng.random((3, 2)) + .1),
                                                                     ([2.0], rng.random((3, 2)) + .1)])
        text = apath.read_bytes()
        end = text.index(b"0.1000000E+31") + 13
        apath.write_bytes(text[:end + 1 + 6])                       # half of sweep 1's parameter
        ts, _ = self.read(apath)
        self.assertEqual((len(ts.sweep_indices), ts.truncated), (1, True))

    def test_a_complete_file_is_not_truncated(self):
        path, _ = make_multi(self.dir, "2001")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertFalse(read_traces(path).truncated)

    def test_cut_inside_the_first_ascii_field(self):
        path = self.dir / "live.tr0"
        fixtures.write_ascii(path, ["TIME", "v(a"], [1, 1], [([], np.array([[1e-9, 0.5], [2e-9, 0.7]]))])
        raw = path.read_bytes()
        start = raw.index(b"$&%#\n") + 5
        for n in range(1, 13):                                      # inside the first 13-character field
            path.write_bytes(raw[:start + n])
            ts, info = self.read(path)
            self.assertTrue(ts.truncated, n)
            self.assertEqual(info["points"], 0)

    def test_a_cut_header_says_so(self):
        path = self.dir / "hdr.tr0"
        fixtures.write_binary(path, "9601", ["TIME", "v(in", "i(vdd"], [1, 1, 8], [([], np.ones((5, 3)))])
        raw = path.read_bytes()
        path.write_bytes(raw[:raw.index(b"i(vdd") + 3])             # inside the last name
        with self.assertRaisesRegex(ValueError, "the header is incomplete"):
            read_header(path)

    def test_a_title_saying_reserved(self):
        path = self.dir / "corp.tr0"
        data = np.random.default_rng(0).random((5, 2))
        fixtures.write_binary(path, "9601", ["TIME", "v(out"], [1, 1], [([], data)])
        raw = path.read_bytes()
        n0 = struct.unpack("<i", raw[12:16])[0]
        text = raw[16:16 + n0].decode().replace(" fixture.sp ", " * (c) ACME. All Rights Reserved. ", 1).encode()
        path.write_bytes(raw[:12] + struct.pack("<i", len(text)) + text + struct.pack("<i", len(text)) + raw[20 + n0:])
        ts = read_traces(path)
        self.assertEqual(ts.selected, ["TIME", "v_out"])
        np.testing.assert_allclose(ts.data["v_out"][0], data[:, 1], rtol=1e-7)

    def test_ascii_capacity_follows_the_field_width(self):
        # 9601-style ASCII writes 11-character fields; the estimate must still bound the points
        def field11(v):
            if v == 0.0:
                return "0.00000E+00"
            e = int(np.floor(np.log10(abs(v)))) + 1
            m = f"{abs(v) / 10.0 ** e:.5f}"
            if m.startswith("1"):
                m, e = "0.10000", e + 1
            return ("-" if v < 0 else "0") + m[1:] + f"E{e:+03d}"
        data = np.column_stack([np.linspace(1e-12, 1e-6, 5000), np.linspace(0.1, 0.9, 5000)])
        path = self.dir / "a9601.tr0"
        fields = [field11(v) for v in data.ravel().tolist() + [1e30]]
        with open(path, "w") as f:
            f.write("   2   0   0    9601 * x.sp\n01/01/2026 00:00:00 Copyright (c) 2026 X. All Rights Reserved."
                    "\n  0\n1 1 TIME v(a $&%#\n")
            f.writelines("".join(fields[i:i + 5]) + "\n" for i in range(0, len(fields), 5))
        with open(path, "rb") as f:
            header = reader._read_ascii_header(f, str(path))
            self.assertGreaterEqual(reader._data_capacity(f, header), 5000)
        np.testing.assert_allclose(read_traces(path).data["v_a"][0], data[:, 1], rtol=1e-4)


class TestNamesAndValues(unittest.TestCase):
    def test_phase_stays_in_its_documented_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inv.ac0"
            fixtures.write_binary(path, "2001", ["HERTZ", "v(out"], [2, 1],
                                  [([], np.array([[1e3, -5.0, 0.0], [1e4, -5.0, -0.0]]))])
            phase = read_traces(path, ["v(out"]).data["v_out_Phase"][0]
            np.testing.assert_array_equal(phase, [180.0, 180.0])           # (-180, 180]: never -180
            self.assertEqual(reader.scan(path).ranges["v_out_Phase"], (180.0, 180.0))

    def test_a_repeat_suffix_never_takes_an_existing_name(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            names = reader._uniquify("f", ["time", "x#3", "x", "x"])
        self.assertEqual(names, ["time", "x#3", "x", "x#3_2"])
        self.assertEqual(len(set(names)), len(names))

    def test_names_must_be_strings(self):
        for names in ([1], [None], [b"TIME"], 5):
            with self.assertRaisesRegex(ValueError, "names must be"):
                read_traces(HERE / "test_9601.tr0", names)
        from spice_result_parser import read_measures
        with self.assertRaisesRegex(ValueError, "names must be measure names"):
            read_measures(HERE / "test.mt0", [1])


class TestArgumentChecks(unittest.TestCase):
    """Arguments the reader and the API reject, with an error that names the problem."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, _ = make_multi(self.dir, "2001")             # 3 sweeps

    def tearDown(self):
        self.tmp.cleanup()

    def test_sweep_index_out_of_range(self):
        with self.assertRaisesRegex(ValueError, r"sweep index 3 out of range; the file holds 3 sweeps \(0 to 2\)"):
            read_traces(self.path, ["v_a"], sweeps=[0, 3])
        with self.assertRaisesRegex(ValueError, "sweep index -1 out of range"):
            api.extract(self.path, ["v_a"], sweeps=[-1], output="csv", dest=self.dir / "o.csv")
        self.assertFalse((self.dir / "o.csv").exists())

    def test_sweep_index_of_a_truncated_file(self):
        path, _ = make_multi(self.dir, "2001", final_sentinel=False, drop_tail=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, r"sweep index 5 out of range.*\(the file is truncated\)"):
                read_traces(path, sweeps=[5])

    def test_sweeps_forms(self):
        self.assertEqual(read_traces(self.path, ["v_a"], sweeps=1).sweep_indices, [1])   # a bare int
        self.assertEqual(read_traces(self.path, ["v_a"], sweeps=[]).sweep_indices, [])   # none kept
        for bad in ([1.5], True, [True]):
            with self.assertRaisesRegex(ValueError, "integer sweep indices"):
                read_traces(self.path, ["v_a"], sweeps=bad)

    def test_read_traces_validates_xrange(self):
        for bad in ((5, 1), (0, 1, 2), "12", (float("nan"), 1)):
            with self.assertRaisesRegex(ValueError, "xrange"):
                read_traces(self.path, ["v_a"], xrange=bad)
        ts = read_traces(self.path, ["v_a"], xrange=(float("-inf"), float("inf")))   # an open bound
        self.assertEqual(sum(a.size for a in ts.data["v_a"]), 1500 + 977 + 2310)

    def test_yrange_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "yrange must hold finite numbers"):
            api.extract(self.path, ["v_a"], yrange=(0, float("inf")))

    def test_downsample_must_be_an_integer_of_at_least_2(self):
        for bad in (2.5, True, "5"):
            with self.assertRaisesRegex(ValueError, "downsample must be an integer"):
                api.extract(self.path, ["v_a"], downsample=bad)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            api.extract(self.path, ["v_a"], downsample=1)
        self.assertEqual(api.extract(self.path, ["v_a"], downsample=np.int64(5)).data["v_a"][0].size, 5)

    def test_not_a_result_file(self):
        for name, content in (("text.tr0", b"hello world\nfoo\n"), ("bin.tr0", b"\x00abcd" * 64)):
            path = self.dir / name
            path.write_bytes(content)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # the garbage block is short
                with self.assertRaisesRegex(ValueError, "not an HSPICE result file .* or a Nutmeg rawfile"):
                    read_header(path)


class TestDestChecks(unittest.TestCase):
    """dest is checked before the read, and never replaces the input."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "run.tr0"
        shutil.copy(HERE / "test_9601.tr0", self.path)
        self.original = self.path.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_dest_equal_to_the_input_is_refused(self):
        for output in ("csv", "npz", "png"):
            with self.assertRaisesRegex(ValueError, "is the input file"):
                api.extract(self.path, ["v_vo"], output=output, dest=self.path)
        ts = read_traces(self.path, ["v_vo"])
        with self.assertRaisesRegex(ValueError, "is the input file"):
            api.write_file(ts, "csv", str(self.dir / "." / "run.tr0"))
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_unwritable_dest_fails_before_the_read(self):
        with self.assertRaisesRegex(FileNotFoundError, "missing.tr0"):
            api.extract(self.dir / "nodir" / "missing.tr0", output="csv")   # not its default dest
        with mock.patch.object(api, "read_traces") as read:
            with self.assertRaisesRegex(ValueError, "is a directory"):
                api.extract(self.path, output="csv", dest=self.dir)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                api.extract(self.path, output="csv", dest=self.dir / "nope" / "o.csv")
        read.assert_not_called()

    def test_dest_without_a_file_output_is_refused(self):
        for output in ("arrays", "summary"):
            with self.assertRaisesRegex(ValueError, "dest applies to csv, npz and png only"):
                api.extract(self.path, output=output, dest=self.dir / "o.csv")

    def test_write_file_passes_title_to_the_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        with mock.patch.object(api, "make_figure", wraps=api.make_figure) as figure:
            dest = api.write_file(read_traces(self.path, ["v_vo"]), "png", self.dir / "t.png", title="run 7")
        self.assertEqual(figure.call_args.args[2], "run 7")
        self.assertTrue(Path(dest).exists())


class TestPackage(unittest.TestCase):
    def test_exports(self):
        import spice_result_parser as hp

        exported = {"Header", "TraceSet", "Scan", "read_header", "read_traces", "scan", "list_traces",
                    "extract", "write_file", "summarize", "MeasureSet", "read_measures"}
        self.assertEqual(set(hp.__all__), exported)
        for name in exported:
            self.assertTrue(hasattr(hp, name), name)

    def test_old_converter_is_gone(self):
        import spice_result_parser as hp

        self.assertFalse(hasattr(hp, "convert"))
        with self.assertRaises(ImportError):
            import spice_result_parser.hspiceParser  # noqa: F401
        text = (HERE.parent / "pyproject.toml").read_text()
        self.assertNotIn("srp-parser", text)
        self.assertNotIn("scipy", text)

    def test_import_writes_nothing_to_stdout(self):
        # srp-mcp speaks JSON-RPC on stdout: an import-time print would corrupt the stream.
        import importlib
        import subprocess

        src = Path(importlib.import_module("spice_result_parser").__file__).parent.parent
        out = subprocess.run([sys.executable, "-c", "import sys; sys.modules['scipy'] = None\n"
                              "import spice_result_parser, spice_result_parser.mcp_server"],
                             check=True, cwd=str(src), capture_output=True, text=True)
        self.assertEqual(out.stdout, "")

    def test_module_runs_the_server(self):
        # `python -m spice_result_parser.mcp_server` must start the server, not exit silently.
        import subprocess

        code = ("import runpy, sys; sys.modules['mcp'] = None\n"
                "runpy.run_module('spice_result_parser.mcp_server', run_name='__main__')")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("spice_result_parser[mcp]", out.stderr)

    def test_pixi_test_task_runs_every_suite(self):
        text = (HERE.parent / "pixi.toml").read_text()
        self.assertIn('unittest discover -s test -p "test_*.py"', text)

    def test_pyproject_declares_mcp_extra_and_script(self):
        text = (HERE.parent / "pyproject.toml").read_text()
        self.assertIn('srp-mcp = "spice_result_parser.mcp_server:main"', text)
        self.assertIn('mcp = ["mcp>=1.0"]', text)

    def test_usage_documents_api_and_mcp(self):
        text = (HERE.parent / "Usage.md").read_text()
        self.assertIn("srp-mcp", text)
        self.assertIn("list_traces", text)
        self.assertIn("extract(", text)


class TestMcp(unittest.TestCase):
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

    def test_tools_registered(self):
        names = {t.name for t in asyncio.run(self.m.server.list_tools())}
        self.assertEqual(names, {"list_traces", "extract", "read_measures"})

    def test_server_name(self):
        self.assertEqual(self.m.server.name, "spice-result-parser")

    def test_ac_format_over_mcp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = make_ac(tmp)
            out = self.m.list_traces(str(path))
            self.assertEqual(out["traces"][:2], ["v_a_Mag", "v_a_Phase"])
            params = {t.name: list(self.schema(t)["properties"])
                      for t in asyncio.run(self.m.server.list_tools())}
            # list_traces takes the parameters extract has for naming and ranging traces, in the same order
            self.assertEqual(params["list_traces"], ["path", "names", "plot", "ac_format", "header_only", "engine"])
            self.assertEqual([p for p in params["extract"] if p in params["list_traces"]],
                             ["path", "names", "plot", "ac_format", "engine"])
            db = self.m.list_traces(str(path), ac_format="db")
            self.assertEqual(db["traces"][:2], ["v_a_dB", "v_a_Phase"])
            ts = read_traces(path, ["v(a"], ac_format="db")
            self.assertEqual(db["ranges"]["v_a_dB"], [float(ts.data["v_a_dB"][0].min()),
                                                      float(ts.data["v_a_dB"][0].max())])
            text = self.m.extract(str(path), ["v(a"], ac_format="realimag")
            self.assertIn(self.m.AC_LEGEND["realimag"], text)
            self.assertIn("v_a_Re", text)

    def tools(self):
        return {t.name: t for t in asyncio.run(self.m.server.list_tools())}

    @staticmethod
    def schema(tool):
        return getattr(tool, "inputSchema", None) or tool.input_schema     # mcp 1.x / 2.x

    def test_every_parameter_is_described_in_the_schema(self):
        for name, tool in self.tools().items():
            for param, spec in self.schema(tool)["properties"].items():
                self.assertTrue(spec.get("description"), f"{name}.{param} has no description")

    def test_descriptions_are_short_and_start_with_usage(self):
        tools = self.tools()
        for name, tool in tools.items():
            self.assertLess(len(tool.description), 1400, name)
        self.assertIn("list_traces", tools["extract"].description)       # says what to call first
        self.assertIn("read_measures", tools["list_traces"].description)

    def test_list_traces_documents_its_result(self):
        text = self.tools()["list_traces"].description
        out = self.m.list_traces(str(HERE / "test_9601.tr0"))
        self.assertNotIn("sweep_count_hint", out)
        for field in out:
            self.assertRegex(text, rf"\b{field}\b", field)

    def test_extract_documents_its_json_results(self):
        text = self.tools()["extract"].description
        summary = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="summary")
        csv_out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv",
                                 dest=str(self.dir / "o.csv"), sweeps=[0])
        for field in [*summary, *csv_out]:
            if field not in ("path", "format", "analysis", "plot", "plots", "sweep_params"):
                self.assertRegex(text, rf"\b{field}\b", field)

    def test_list_traces_tool_gives_the_extents(self):
        out = self.m.list_traces(str(HERE / "test_9601.tr0"), names=["v(vo)", "I(VS)"])
        self.assertEqual(out["x_range"], [0.0, 0.01])
        self.assertEqual(out["points"], 2605)
        self.assertEqual(out["traces"], ["v_vo", "i_vs"])               # case is ignored as a fallback
        self.assertEqual(out["ranges"], {"v_vo": [0.0, 1.8477405e-05], "i_vs": [-0.005, 1.8477404e-08]})
        self.assertNotIn("note", out)
        quick = self.m.list_traces(str(HERE / "test_9601.tr0"), header_only=True)
        self.assertNotIn("ranges", quick)
        self.assertEqual(quick["sweeps"], 1)

    def test_wide_results_cap_the_ranges(self):
        names = ["TIME"] + [f"v(n{i}" for i in range(150)]
        data = np.random.default_rng(2).random((7, len(names)))
        data[:, 0] = np.arange(7) * 1e-9
        path = self.dir / "wide.tr0"
        fixtures.write_binary(path, "2001", names, [1] * len(names), [([], data)])
        out = self.m.list_traces(str(path))
        self.assertEqual((out["trace_count"], len(out["traces"])), (150, self.m.MCP_MAX_TRACES))
        self.assertEqual(list(out["ranges"]), out["traces"])            # the ranges of the names listed
        self.assertEqual(sum(out["groups"].values()), 150)              # the groups count every trace
        self.assertEqual(out["note"], f"traces lists the first {self.m.MCP_MAX_TRACES} of 150 traces, and groups "
                                      f"counts them all by name: pass a key of groups as names (to list_traces "
                                      f"or extract) for its traces")
        self.assertNotIn("note", self.m.list_traces(str(path), names=["v_n149"]))
        summary = self.m.extract(str(path), output="summary")
        self.assertEqual(len(summary["ranges"]), self.m.MCP_MAX_RANGES)
        self.assertEqual(len(summary["stats"]), 150)
        self.assertIn("note", summary)
        text = self.m.extract(str(path))
        self.assertIn(f"-- {150 - self.m.TEXT_MAX_STATS} more traces not shown; select fewer names --", text)
        self.assertIn(f"the tables below show the first {self.m.TEXT_MAX_TRACES} of 150 traces", text)
        heading = [line.split() for line in text.splitlines() if line.lstrip().startswith("TIME")]
        self.assertEqual(heading, [["TIME"] + out["traces"][:self.m.TEXT_MAX_TRACES]])   # x and 16 traces

    def test_list_traces_tool_reads_only_the_header_of_a_large_file(self):
        tool = self.tools()["list_traces"]
        header_only = self.schema(tool)["properties"]["header_only"]["description"]
        for text in (tool.description, header_only):                # the limits that the tool states
            self.assertIn(f"over {api._size(api.SCAN_MAX_BYTES)}" if text is tool.description
                          else f"at most {api._size(api.SCAN_MAX_BYTES)}", text)
        self.assertIn(f"at most {api.WAVEVIEW_MAX_TRACES} traces and {api.WAVEVIEW_SCAN_MAX_VALUES / 1e6:g} M values",
                      header_only)
        with mock.patch.object(api, "SCAN_MAX_BYTES", 1000):
            out = self.m.list_traces(str(HERE / "test_9601.tr0"))
            self.assertNotIn("ranges", out)
            self.assertIn("header_only=False makes it", out["note"])
            self.assertTrue(out["note"].endswith("; extract reads this file only with allow_large=True"))
            self.assertEqual(out["engine"], "python")
            out = self.m.list_traces(str(HERE / "test_9601.tr0"), header_only=True)
            self.assertIn("extract reads this file only with allow_large=True", out["note"])
            self.assertIn("ranges", self.m.list_traces(str(HERE / "test_9601.tr0"), header_only=False))
        self.assertNotIn("note", self.m.list_traces(str(HERE / "test_9601.tr0")))

    def test_extract_asks_before_a_large_read(self):
        path = str(HERE / "test_9601.tr0")
        size = (HERE / "test_9601.tr0").stat().st_size
        with mock.patch.object(api, "SCAN_MAX_BYTES", 1000):
            cost = api.read_cost(path, names="v(vo")
            for output in ("text", "summary", "csv"):
                with self.subTest(output), \
                        mock.patch.object(reader, "_feed_data", side_effect=AssertionError("data read")):
                    with self.assertRaises(self.m.ToolError) as caught:
                        self.m.extract(path, "v(vo", output=output, dest=str(self.dir / "o.csv") if output == "csv"
                                       else None)
                then = " Then text and summary reduce each point that they keep." if output != "csv" else ""
                self.assertEqual(str(caught.exception),
                                 f"extract did not read the file: it has {api._size(size)}, about "
                                 f"{api._amount(cost.points)} points, more than the 1 kB that extract reads without "
                                 f"allow_large=True. Each extract goes through all of it, for any names, sweeps "
                                 f"and xrange: {api.read_time(cost)}.{then} Without xrange, x and the 1 selected "
                                 f"trace take about {api._size(cost.memory)} of memory; with xrange, only their "
                                 f"window. Pass allow_large=True to read it")
            with self.assertRaisesRegex(self.m.ToolError, r"With this xrange, memory holds only the window of x "
                                                          r"and the 2 selected traces\. Pass allow_large=True"):
                self.m.extract(path, ["v(vo", "v(vs"], xrange=[0, 0.001])
            took = re.compile(rf"this read took \d+ s: each extract reads all {re.escape(api._size(size))} of the "
                              rf"file again, for any names, sweeps and xrange")
            text = self.m.extract(path, "v(vo", allow_large=True)
            self.assertRegex(text.splitlines()[-1], rf"^-- {took.pattern} --$")    # the time, after the data
            self.assertRegex(self.m.extract(path, "v(vo", output="summary", allow_large=True)["note"], took)
            written = self.m.extract(path, "v(vo", output="csv", dest=str(self.dir / "o.csv"), allow_large=True)
            self.assertRegex(written["note"], took)
        self.assertNotIn("this read took", self.m.extract(path, "v(vo"))           # small: no note

    def test_a_summary_of_too_many_traces_fails_before_the_read(self):
        names = ["TIME"] + [f"v(n{i}" for i in range(self.m.MCP_MAX_SERIES + 1)]
        path = self.dir / "wide.tr0"
        fixtures.write_binary(path, "2001", names, [1] * len(names), [([], np.ones((3, len(names))))])
        with mock.patch.object(reader, "_feed_data", side_effect=AssertionError("data read")):
            with self.assertRaisesRegex(self.m.ToolError, rf"^a summary of {self.m.MCP_MAX_SERIES + 1} traces \(a "
                                                          rf"series per trace and sweep\) is too large"):
                self.m.extract(str(path), output="summary")

    def test_the_tools_state_what_a_large_read_costs(self):
        props = self.schema(self.tools()["extract"])["properties"]
        self.assertIn("the read still goes through the whole file", props["xrange"]["description"])
        self.assertNotIn("cheap", props["xrange"]["description"])
        self.assertIn(f"over {api._size(api.SCAN_MAX_BYTES)}", props["allow_large"]["description"])
        self.assertIn(f"over {api.WAVEVIEW_MAX_TRACES} traces or {api.WAVEVIEW_SCAN_MAX_VALUES / 1e6:g} M values",
                      props["allow_large"]["description"])

    def test_list_traces_documents_its_cut_result(self):
        names = ["TIME"] + [f"v(x{c}.n{i}" for c in range(3) for i in range(50)]
        path = self.dir / "cut.tr0"
        fixtures.write_binary(path, "2001", names, [1] * len(names), [([], np.ones((3, len(names))))])
        out = self.m.list_traces(str(path))
        self.assertEqual(out["groups"], {"v(x0.*": 50, "v(x1.*": 50, "v(x2.*": 50})
        text = self.tools()["list_traces"].description
        for field in out:
            self.assertRegex(text, rf"\b{field}\b", field)

    def test_a_missed_window_says_where_the_data_is(self):
        hint = "the whole plot has 2605 points with TIME in [0, 0.01]"
        text = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], xrange=[1, 2])
        self.assertIn(f"no points in xrange [1, 2]; {hint}", text)
        summary = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="summary", xrange=[1, 2])
        self.assertEqual((summary["points"], summary["x_range"]), (0, None))
        self.assertEqual(summary["note"], f"no points in xrange [1, 2]; {hint}")
        inside = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], xrange=[0.001, 0.002])
        self.assertIn(f"window: xrange [0.001, 0.002]; {hint}", inside)
        self.assertNotIn("note", self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="summary",
                                                xrange=[0.001, 0.002]))

    def test_true_false_and_strings_are_not_numbers(self):
        # the schema's lax parsing would make true 1 and "0.5" 0.5 before the tool's own checks
        path = str(HERE / "test_9601.tr0")
        for args in ({"tolerance": True}, {"sweeps": True}, {"plot": True}, {"xrange": [False, True]},
                     {"tolerance": "0.5"}, {"downsample": "20"}, {"yrange": [0, True]}):
            with self.assertRaisesRegex(Exception, "expected a number", msg=str(args)):
                asyncio.run(self.m.server.call_tool("extract", {"path": path, **args}))
        with self.assertRaisesRegex(Exception, "expected a number"):
            asyncio.run(self.m.server.call_tool("read_measures", {"path": str(HERE / "test.mt0"), "rows": True}))
        result = asyncio.run(self.m.server.call_tool("extract", {"path": path, "names": "v(vo", "tolerance": 0,
                                                                  "sweeps": [0], "xrange": [0, 0.001]}))
        content = getattr(result, "content", None) or (result[0] if isinstance(result, tuple) else result)
        self.assertIn("265 points", content[0].text)                # real numbers still pass

    def test_many_sweeps_stay_within_one_reply(self):
        t = np.linspace(0, 1e-8, 50)
        sweeps = [([float(k)], np.c_[t, np.tanh(t / 1e-9) * (1 + k / 1000)]) for k in range(300)]
        path = self.dir / "mc.tr0"
        fixtures.write_binary(path, "2001", ["TIME", "v(o", "mc"], [1, 1], sweeps)
        info = self.m.list_traces(str(path))
        self.assertEqual(info["sweeps"], 300)
        self.assertEqual(len(info["sweep_values"]), self.m.MCP_MAX_SWEEP_VALUES)
        self.assertIn(f"sweep_values lists the first {self.m.MCP_MAX_SWEEP_VALUES} of 300 sweeps", info["note"])
        with self.assertRaisesRegex(self.m.ToolError, "300 series is too large .* sweeps=\\[...\\]"):
            self.m.extract(str(path), output="summary")
        summary = self.m.extract(str(path), output="summary", sweeps=list(range(150)))
        self.assertEqual(len(summary["stats"]["v_o"]), 150)
        self.assertLess(len(self.m.extract(str(path))), 30_000)       # text is bounded by itself

    def test_relative_path_errors_name_the_server_directory(self):
        import os

        with self.assertRaisesRegex(self.m.ToolError, "server's working directory, " + re.escape(os.getcwd())):
            self.m.list_traces("no_such_run.tr0")
        with self.assertRaises(self.m.ToolError) as caught:
            self.m.list_traces(str(self.dir / "no_such_run.tr0"))
        self.assertNotIn("working directory", str(caught.exception))

    def test_plot_on_hspice_is_documented_as_an_error(self):
        props = self.schema(self.tools()["extract"])["properties"]
        self.assertNotIn("Ignored for HSPICE", props["plot"]["description"])
        with self.assertRaisesRegex(self.m.ToolError, "Nutmeg rawfiles only"):
            self.m.extract(str(HERE / "test_9601.tr0"), plot=1)

    def test_read_measures_tool(self):
        text = self.m.read_measures(str(HERE / "test.mt0"), ["rchg"])
        self.assertTrue(text.startswith("measures | 1 row"))
        self.assertIn("1.623e-08", text)
        self.assertNotIn("tmech", text)
        with self.assertRaisesRegex(self.m.ToolError, "not an HSPICE measure file"):
            self.m.read_measures(str(HERE / "test_9601.tr0"))

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
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="summary", downsample=20)
        json.dumps(out, allow_nan=False)
        self.assertEqual(len(out["stats"]["v_vo"][0]["envelope"]["from"]), 10)

    def test_csv_tool_returns_dest_and_counts(self):
        dest = self.dir / "o.csv"
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv", dest=str(dest))
        self.assertEqual(list(out), ["output", "dest", "x", "x_range", "points", "sweeps", "truncated",
                                     "traces", "ranges", "engine"])
        self.assertEqual(out["dest"], str(dest))          # the parameter's name: path is always the input
        self.assertEqual(out["x"], "TIME")
        self.assertEqual(out["x_range"], [0.0, 0.01])
        self.assertEqual(out["traces"], ["v_vo"])
        self.assertEqual(out["ranges"], {"v_vo": [0.0, 1.8477405e-05]})
        self.assertEqual(out["sweeps"], 1)
        self.assertEqual(out["points"], 2605)
        self.assertNotIn("sweep_indices", out)            # only when sweeps= was given
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
        self.assertEqual(out["dest"], str(dest))
        self.assertEqual(out["points"], int(((x >= lo) & (x <= hi)).sum()))   # 802: a repeated time point
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
        self.assertEqual(out["points"], 10)
        self.assertEqual(len(dest.read_text().splitlines()), 11)

    def test_summary_defaults_to_statistics_only(self):
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="summary")
        entry = out["stats"]["v_vo"][0]
        self.assertEqual(entry["count"], 2605)
        self.assertNotIn("x", entry)

    def test_downsample_below_2_is_an_error_for_every_output(self):
        for output in ("text", "summary", "csv"):
            with self.assertRaisesRegex(self.m.ToolError, "at least 2"):
                self.m.extract(str(HERE / "test_9601.tr0"), output=output, downsample=1,
                               dest=str(self.dir / "o.csv") if output == "csv" else None)

    def test_dest_with_text_is_an_error(self):
        with self.assertRaisesRegex(self.m.ToolError, "dest applies to csv, npz and png only"):
            self.m.extract(str(HERE / "test_9601.tr0"), dest=str(self.dir / "o.csv"))

    def test_dest_equal_to_the_input_is_an_error(self):
        path = self.dir / "run.tr0"
        shutil.copy(HERE / "test_9601.tr0", path)
        with self.assertRaisesRegex(self.m.ToolError, "is the input file"):
            self.m.extract(str(path), output="csv", dest=str(path))
        self.assertEqual(path.read_bytes(), (HERE / "test_9601.tr0").read_bytes())

    def test_relative_dest_is_returned_absolute(self):
        import os

        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv", dest="rel.csv", sweeps=[0])
        finally:
            os.chdir(cwd)
        self.assertEqual(out["dest"], str((self.dir / "rel.csv").resolve()))
        self.assertEqual(out["sweep_indices"], [0])

    def test_sweep_out_of_range_reaches_the_client(self):
        with self.assertRaisesRegex(self.m.ToolError, "sweep index 99 out of range"):
            self.m.extract(str(HERE / "test_9601.tr0"), sweeps=[99])

    def test_schema_enums_and_string_names(self):
        props = self.schema(self.tools()["extract"])["properties"]
        self.assertEqual(props["output"]["enum"], ["text", "summary", "csv", "npz", "png"])
        self.assertEqual(props["ac_format"]["enum"], ["magphase", "db", "realimag"])
        result = asyncio.run(self.m.server.call_tool("extract", {"path": str(HERE / "test_9601.tr0"),
                                                                  "names": "v(vo"}))
        content = getattr(result, "content", None) or (result[0] if isinstance(result, tuple) else result)
        self.assertIn("v_vo", content[0].text)

    def test_tool_annotations(self):
        if self.m.ToolAnnotations is None:
            self.skipTest("mcp before 1.7 has no tool annotations")
        tools = self.tools()
        for name in ("list_traces", "read_measures"):
            self.assertTrue(tools[name].annotations.read_only_hint
                            if hasattr(tools[name].annotations, "read_only_hint")
                            else tools[name].annotations.readOnlyHint, name)
        hints = tools["extract"].annotations
        self.assertTrue(getattr(hints, "destructive_hint", None) or getattr(hints, "destructiveHint", None))

    def test_read_measures_returns_text_only(self):
        tool = self.tools()["read_measures"]
        self.assertIsNone(getattr(tool, "output_schema", None) or getattr(tool, "outputSchema", None))
        with self.assertRaisesRegex(self.m.ToolError, "rows must be at least 2"):
            self.m.read_measures(str(HERE / "test.mt0"), rows=1)

    def test_bad_range_rejected_before_reading(self):
        with self.assertRaisesRegex(self.m.ToolError, "xrange"):
            self.m.extract(str(self.dir / "missing.tr0"), output="csv", xrange=[1.0])


    def test_error_text_reaches_the_client(self):
        with self.assertRaisesRegex(Exception, "Available traces"):
            asyncio.run(self.m.server.call_tool("extract", {"path": str(HERE / "test_9601.tr0"),
                                                             "names": ["nope"], "output": "csv"}))
        with self.assertRaisesRegex(Exception, "Input should be 'text'"):    # the schema's enum
            asyncio.run(self.m.server.call_tool("extract", {"path": str(HERE / "test_9601.tr0"),
                                                             "output": "xlsx"}))

    def test_unexpected_errors_carry_their_type(self):
        original = self.m.api.list_traces
        def boom(*args, **kwargs):
            raise KeyError("boom")
        self.m.api.list_traces = boom
        self.addCleanup(setattr, self.m.api, "list_traces", original)
        with self.assertRaisesRegex(self.m.ToolError, "KeyError: 'boom'"):
            self.m.list_traces("whatever")

if __name__ == "__main__":
    unittest.main()
