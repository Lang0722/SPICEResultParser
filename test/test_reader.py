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

    def test_short_payload_raises(self):
        path, _ = make_multi(self.dir, "9601")
        b = path.read_bytes()
        path.write_bytes(b[:-10])
        with self.assertRaisesRegex(ValueError, "head promises"):
            read_traces(path)

if __name__ == "__main__":
    unittest.main()
