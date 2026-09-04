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


if __name__ == "__main__":
    unittest.main()
