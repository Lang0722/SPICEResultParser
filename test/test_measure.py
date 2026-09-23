"""Tests for the HSPICE measure-file reader (.mt#, .ms#, .ma#)."""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from spice_result_parser import measure  # noqa: E402
from spice_result_parser.measure import read_measures  # noqa: E402
from spice_result_parser.reader import read_header, read_traces  # noqa: E402

# A swept measure file: an index column, one swept parameter, rows wrapped over two
# lines each, and a measure that failed in one row.
SWEPT = """$DATA1 SOURCE='HSPICE' VERSION='P-2019.06 linux64' PARAM_COUNT=1
.TITLE '* inverter delay'
 index            vdd              tpd              tpw
                  temper           alter#
 1.0000           0.9000           2.345e-11        1.000e-10
                  25.0000          1.0000
 2.0000           1.0000           failed           1.100e-10
                  25.0000          1.0000
 3.0000           1.1000           1.845e-11        1.200e-10
                  25.0000          1.0000
"""


class TestSampleFile(unittest.TestCase):
    def test_names_and_values(self):
        ms = read_measures(HERE / "test.mt0")
        self.assertEqual(ms.names, ["rchg", "tmech", "trl", "tmechbb", "trlbb", "temper", "alter#"])
        self.assertEqual(ms.params, [])
        self.assertEqual(ms.title, "* dc sweep of a relay to show that it works")
        self.assertEqual(ms.rows, 1)
        np.testing.assert_array_equal(ms.values["rchg"], [1.623e-08])
        np.testing.assert_array_equal(ms.values["temper"], [25.0])
        np.testing.assert_array_equal(ms.values["alter#"], [1.0])


class TestSweptFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "inv.mt0"
        self.path.write_text(SWEPT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wrapped_rows_and_params(self):
        ms = read_measures(self.path)
        self.assertEqual(ms.names, ["index", "vdd", "tpd", "tpw", "temper", "alter#"])
        self.assertEqual(ms.params, ["vdd"])
        self.assertEqual(ms.rows, 3)
        np.testing.assert_array_equal(ms.values["vdd"], [0.9, 1.0, 1.1])
        np.testing.assert_array_equal(ms.values["tpw"], [1.0e-10, 1.1e-10, 1.2e-10])

    def test_failed_measure_is_nan(self):
        ms = read_measures(self.path)
        np.testing.assert_array_equal(np.isnan(ms.values["tpd"]), [False, True, False])
        self.assertEqual(ms.failed, {"tpd": 1})

    def test_selection_keeps_index_and_params(self):
        ms = read_measures(self.path, ["tp*"])
        self.assertEqual(ms.names, ["index", "vdd", "tpd", "tpw"])
        ms = read_measures(self.path, ["tpw"])
        self.assertEqual(list(ms.values), ["index", "vdd", "tpw"])

    def test_unknown_measure_lists_available(self):
        with self.assertRaisesRegex(ValueError, "unknown measure 'nope'.*tpd"):
            read_measures(self.path, ["nope"])

    def test_partial_last_row_is_dropped_with_warning(self):
        self.path.write_text(SWEPT + " 4.0000           1.2000\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ms = read_measures(self.path)
        self.assertEqual(ms.rows, 3)
        self.assertTrue(any("incomplete" in str(w.message) for w in caught))

    def test_non_numeric_value_raises(self):
        self.path.write_text(SWEPT.replace("1.845e-11", "garbage"))
        with self.assertRaisesRegex(ValueError, "garbage"):
            read_measures(self.path)

    def test_not_a_measure_file(self):
        with self.assertRaisesRegex(ValueError, "not an HSPICE measure file"):
            read_measures(HERE / "test_9601.tr0")


class TestTraceReaderPointsToMeasures(unittest.TestCase):
    def test_read_header_names_read_measures(self):
        with self.assertRaisesRegex(ValueError, "measure file.*read_measures"):
            read_header(HERE / "test.mt0")

    def test_read_traces_names_read_measures(self):
        with self.assertRaisesRegex(ValueError, "measure file.*read_measures"):
            read_traces(HERE / "test.mt0")


class TestFormatMeasures(unittest.TestCase):
    """mcp_server.format_measures: the agent-readable rendering; runs without the mcp package."""

    def setUp(self):
        from spice_result_parser import mcp_server

        self.m = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "inv.mt0"
        self.path.write_text(SWEPT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_swept_table(self):
        lines = self.m.format_measures(read_measures(self.path)).splitlines()
        self.assertEqual(lines[0], "measures | 3 rows | params: vdd")
        self.assertEqual(lines[1], "failed: tpd in 1 of 3 rows")
        self.assertEqual(lines[3].split(), ["index", "vdd", "tpd", "tpw", "temper", "alter#"])
        self.assertEqual(lines[5].split()[2], "failed")
        self.assertEqual(len(lines), 7)

    def test_rows_are_decimated(self):
        lines = self.m.format_measures(read_measures(self.path), rows=2).splitlines()
        self.assertEqual(lines[0], "measures | 3 rows (2 shown) | params: vdd")
        self.assertEqual([l.split()[0] for l in lines[-2:]], ["1", "3"])

    def test_single_row_without_params(self):
        lines = self.m.format_measures(read_measures(HERE / "test.mt0")).splitlines()
        self.assertEqual(lines[0], "measures | 1 row")
        self.assertEqual(lines[2].split()[0], "rchg")
        self.assertEqual(lines[3].split()[0], "1.623e-08")


class TestPackage(unittest.TestCase):
    def test_exported(self):
        import spice_result_parser as srp

        self.assertIs(srp.read_measures, measure.read_measures)
        self.assertIs(srp.MeasureSet, measure.MeasureSet)


if __name__ == "__main__":
    unittest.main()
