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

from hspice_parser.reader import read_header  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
