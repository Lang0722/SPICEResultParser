"""HSPICE header layouts across releases, and AC probe outputs.

HSPICE writes its header in fixed columns (fixtures.hspice_header_text reproduces the
headers of 2011, 2013, 2020 and 2021 files byte for byte). Only the copyright notice in
it changes with the release: H-2013.03 blanks all of it but "erved.".
"""
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402

from spice_result_parser import api, reader  # noqa: E402
from spice_result_parser.reader import read_header, read_traces  # noqa: E402

TR_NAMES = ["TIME", "v(in", "v(out", "i(vdd"]
TR_CODES = [1, 1, 1, 8]


class TestReleaseHeaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        t = np.linspace(0, 1e-8, 40)
        self.data = np.column_stack([t, np.random.default_rng(13).uniform(0, 1.2, (40, 3))])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, version="9601", **header):
        path = self.dir / name
        fixtures.write_binary(path, version, TR_NAMES, TR_CODES, [([], self.data)], layout="hspice", **header)
        return path

    def test_2013_files_read_without_a_copyright_notice(self):
        for version, rtol in (("9601", 1e-6), ("2001", 0)):
            path = self.write(f"r2013_{version}.tr0", version, notice=fixtures.NOTICE_2013)
            self.assertNotIn(b"Reserved.", path.read_bytes())       # nothing to search for
            ts = read_traces(path)
            self.assertEqual(ts.selected, ["TIME", "v_in", "v_out", "i_vdd"])
            self.assertFalse(ts.truncated)
            np.testing.assert_allclose(ts.data["v_out"][0], self.data[:, 2], rtol=rtol)
            info = api.list_traces(path)
            self.assertEqual((info["format"], info["points"], info["sweeps"]), (version, 40, 1))

    def test_every_release_reads_alike(self):
        notices = {"2011": fixtures.NOTICE_2011, "2013": fixtures.NOTICE_2013,
                   "2020": fixtures.NOTICE_2020, "blank": ""}
        for release, notice in notices.items():
            ts = read_traces(self.write(f"r{release}.tr0", notice=notice))
            np.testing.assert_allclose(ts.data["i_vdd"][0], self.data[:, 3], rtol=1e-6, err_msg=release)

    def test_a_2013_sweep_file(self):
        # as an H-2013.03 .sw0 holds it: an outer sweep parameter, the count in the header columns
        names, codes = ["VOLTS", "v(out", "i(vin", "0:vin2"], [3, 1, 8]
        x = np.linspace(0, 1, 11)
        sweeps = [([v], np.column_stack([x, x * v, -x])) for v in (0.0, 0.5, 1.0)]
        path = self.dir / "s2013.sw0"
        fixtures.write_binary(path, "9601", names, codes, sweeps, layout="hspice", notice=fixtures.NOTICE_2013)
        h = read_header(path)
        self.assertEqual((h.analysis, h.sweep_count_hint, h.sweep_params), ("sw", 3, ["0_vin2"]))
        ts = read_traces(path)
        self.assertEqual(ts.sweep_values, [[0.0], [0.5], [1.0]])
        np.testing.assert_allclose(ts.data["v_out"][2], x, rtol=1e-6)
        self.assertFalse(ts.truncated)

    def test_a_2013_sweep_file_that_stops_early(self):
        # the header announces 5 sweeps; the file holds 2 and the start of a third
        names, codes = ["VOLTS", "v(out", "vin2"], [3, 1]
        x = np.linspace(0, 1, 11)
        sweeps = [([0.0], np.column_stack([x, x])), ([0.5], np.column_stack([x, x])),
                  ([1.0], np.column_stack([x, x])[:4])]
        path = self.dir / "stopped.sw0"
        fixtures.write_binary(path, "9601", names, codes, sweeps, layout="hspice", final_sentinel=False,
                              announce=5, notice=fixtures.NOTICE_2013)
        with self.assertWarnsRegex(RuntimeWarning, "announces 5 sweeps, the file holds 3"):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual([len(v) for v in ts.data["v_out"]], [11, 11, 4])

    def test_ascii_2013_file(self):
        path = self.dir / "a2013.tr0"
        fixtures.write_ascii(path, TR_NAMES, TR_CODES, [([], self.data)], layout="hspice",
                             notice=fixtures.NOTICE_2013)
        self.assertIn("erved.", path.read_text().splitlines()[2])    # HSPICE wraps the header every 80 columns
        ts = read_traces(path)
        self.assertEqual(ts.selected, ["TIME", "v_in", "v_out", "i_vdd"])
        np.testing.assert_allclose(ts.data["v_in"][0], self.data[:, 1], rtol=1e-6)

    def test_a_title_that_says_reserved_and_a_number(self):
        names, codes = ["VOLTS", "v(out", "p"], [3, 1]
        x = np.linspace(0, 1, 5)
        sweeps = [([v], np.column_stack([x, x * v])) for v in (1.0, 2.0)]
        path = self.dir / "title.sw0"
        fixtures.write_binary(path, "2001", names, codes, sweeps, layout="hspice",
                              title="* All Rights Reserved. 7 stages", notice=fixtures.NOTICE_2013)
        self.assertEqual(read_header(path).sweep_count_hint, 2)    # the columns, not the title's 7
        self.assertEqual(read_traces(path).sweep_values, [[1.0], [2.0]])

    def test_a_utf8_title_keeps_the_byte_columns(self):
        path = self.write("utf8.tr0", title="* 放大器 – 1 µA bias", notice=fixtures.NOTICE_2013)
        ts = read_traces(path)
        self.assertEqual(ts.selected, ["TIME", "v_in", "v_out", "i_vdd"])
        np.testing.assert_allclose(ts.data["v_in"][0], self.data[:, 1], rtol=1e-6)

    def test_the_sample_files_are_read_by_their_columns(self):
        for name in ("test_9601.tr0", "test_2001.tr0", "test_9601.sw0", "test_9601.ac0"):
            raw = (HERE / name).read_bytes()
            payload = raw[16:16 + struct.unpack("<i", raw[12:16])[0]]
            tokens = reader._sweep_count_and_variables(name, payload, "")   # "": the notice is not searched
            self.assertEqual((tokens[0], tokens[1], tokens[-1]), ("0", "2" if "ac" in name else
                                                                  "3" if "sw" in name else "1", "$&%#"))

    def test_a_header_over_several_blocks_keeps_its_columns(self):
        # HSPICE ends a header block at a name field, at about 8 KB; the columns hold in the joined text
        names = ["TIME"] + [f"v(n{i}" for i in range(600)]
        data = np.arange(3 * 601, dtype=np.float64).reshape(3, 601)
        path = self.dir / "wide.tr0"
        fixtures.write_binary(path, "2001", names, [1] * 601, [([], data)], layout="hspice",
                              header_bytes=8176, notice=fixtures.NOTICE_2013)
        self.assertEqual(read_header(path).raw_names, names)
        np.testing.assert_array_equal(read_traces(path, ["v(n599"]).data["v_n599"][0], data[:, 600])
        real = read_header(HERE / "waveview" / "l340_9.tr0")       # HSPICE P-2019.06: 8184 + 344 bytes of header
        self.assertEqual((len(real.names), real.names[-1]), (344, "i_vin"))

    def test_a_file_cut_inside_a_header_block(self):
        path = self.dir / "cut.tr0"
        fixtures.write_binary(path, "2001", ["TIME"] + [f"v(n{i}" for i in range(600)], [1] * 601,
                              [([], np.ones((2, 601)))], layout="hspice", header_bytes=8176)
        raw = path.read_bytes()
        for end in (8176 + 20 + 10, 8176 + 20 + 16 + 100):         # in the second block's head, in its payload
            path.write_bytes(raw[:end])
            with self.assertRaisesRegex(ValueError, "the header is incomplete .*terminator"):
                read_header(path)

    def test_other_writers_still_read_by_the_notice(self):
        # the free layout of fixtures.header_text keeps no columns, as another writer might not
        path = self.dir / "free.tr0"
        fixtures.write_binary(path, "9601", TR_NAMES, TR_CODES, [([], self.data)])
        np.testing.assert_allclose(read_traces(path).data["v_out"][0], self.data[:, 2], rtol=1e-6)
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"Reserved.", b"Reserved,", 1))      # same length: the framing holds
        with self.assertRaisesRegex(ValueError, "header has no sweep count: it is not in columns 177-187"):
            read_header(path)


AC_NAMES = ["HERTZ", "v(out", "i(vin", "vdb(out", "vp(out"]
AC_CODES = [2, 1, 8, 6, 5]          # the type codes HSPICE gives vdb and vp


def probe_ac_data(n=36):
    """An AC table as HSPICE stores it: the frequency, (re, im) per complex variable, then
    one real value per probe output (here vdb and vp of v(out), as HSPICE computes them)."""
    rng = np.random.default_rng(7)
    f = np.logspace(2, 9, n)
    z = rng.uniform(-2, 2, (n, 2)) + 1j * rng.uniform(-2, 2, (n, 2))
    vdb, vp = 20 * np.log10(np.abs(z[:, 0])), np.angle(z[:, 0], deg=True)
    return f, z, np.column_stack([f, z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag, vdb, vp])


class TestAcProbes(unittest.TestCase):
    """In an AC file only the variables of the header's first count are complex; the probe
    outputs of the second count (vm, vdb, vp ... from .probe and .print) are real."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.f, self.z, self.data = probe_ac_data()
        self.path = self.dir / "probe.ac0"
        fixtures.write_binary(self.path, "2001", AC_NAMES, AC_CODES, [([], self.data)], nprobe=2,
                              layout="hspice")

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_outputs_are_one_real_column_each(self):
        h = read_header(self.path)
        self.assertEqual(h.names, ["HERTZ", "v_out_Mag", "v_out_Phase", "i_vin_Mag", "i_vin_Phase",
                                   "vdb_out", "vp_out"])
        self.assertEqual((h.ncols, h.ncomplex), (7, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # whole points only: no value is left over
            ts = read_traces(self.path)
        self.assertFalse(ts.truncated)
        np.testing.assert_array_equal(ts.data["HERTZ"][0], self.f)
        np.testing.assert_allclose(ts.data["v_out_Mag"][0], np.abs(self.z[:, 0]), rtol=1e-12)
        np.testing.assert_allclose(ts.data["i_vin_Phase"][0], np.angle(self.z[:, 1], deg=True), rtol=1e-12)
        np.testing.assert_array_equal(ts.data["vdb_out"][0], self.data[:, 5])
        np.testing.assert_array_equal(ts.data["vp_out"][0], self.data[:, 6])
        # the probes agree with the complex variable they describe
        np.testing.assert_allclose(20 * np.log10(ts.data["v_out_Mag"][0]), ts.data["vdb_out"][0], rtol=1e-12)

    def test_every_ac_format_leaves_the_probes_alone(self):
        for ac_format, pair in (("db", ["v_out_dB", "v_out_Phase"]), ("realimag", ["v_out_Re", "v_out_Im"])):
            ts = read_traces(self.path, ac_format=ac_format)
            self.assertEqual(ts.selected[1:3], pair)
            self.assertEqual(ts.selected[-2:], ["vdb_out", "vp_out"])
            np.testing.assert_array_equal(ts.data["vp_out"][0], self.data[:, 6])

    def test_selecting_a_probe_reads_only_its_column(self):
        h = read_header(self.path)
        self.assertEqual(reader._stored_columns(h, [0, 5]), [0, 5])
        self.assertEqual(reader._stored_columns(h, [0, 2]), [0, 1, 2])      # a phase needs its pair
        ts = read_traces(self.path, ["vdb(out"])
        self.assertEqual(ts.selected, ["HERTZ", "vdb_out"])
        np.testing.assert_array_equal(ts.data["vdb_out"][0], self.data[:, 5])

    def test_ranges_follow_the_columns(self):
        info = api.list_traces(self.path)
        self.assertEqual(info["x_range"], [100.0, 1e9])
        self.assertEqual(info["ranges"]["vdb_out"], [self.data[:, 5].min(), self.data[:, 5].max()])
        mag = np.abs(self.z[:, 0])
        np.testing.assert_allclose(info["ranges"]["v_out_Mag"], [mag.min(), mag.max()], rtol=1e-12)

    def test_9601_and_ascii_files(self):
        path = self.dir / "probe32.ac0"
        fixtures.write_binary(path, "9601", AC_NAMES, AC_CODES, [([], self.data)], nprobe=2)
        ts = read_traces(path)
        np.testing.assert_allclose(ts.data["vdb_out"][0], self.data[:, 5], rtol=1e-6)
        np.testing.assert_allclose(ts.data["v_out_Mag"][0], np.abs(self.z[:, 0]), rtol=1e-6)
        for layout in ("free", "hspice"):
            path = self.dir / f"probe_{layout}.ac0"
            fixtures.write_ascii(path, AC_NAMES, AC_CODES, [([], self.data)], nprobe=2, layout=layout)
            ts = read_traces(path)
            self.assertEqual(ts.selected[-2:], ["vdb_out", "vp_out"])
            np.testing.assert_allclose(ts.data["HERTZ"][0], self.f, rtol=1e-6)
            np.testing.assert_allclose(ts.data["vp_out"][0], self.data[:, 6], rtol=1e-6)

    def test_a_swept_ac_file_with_probes(self):
        sweeps = [([c], self.data * [1, c, c, 1, 1, 1, 1]) for c in (1.0, 2.0)]
        path = self.dir / "swept.ac0"
        fixtures.write_binary(path, "2001", AC_NAMES + ["cload"], AC_CODES, sweeps, nprobe=2, layout="hspice")
        ts = read_traces(path, ["v(out", "vdb(out"])
        self.assertEqual(ts.sweep_values, [[1.0], [2.0]])
        np.testing.assert_allclose(ts.data["v_out_Mag"][1], 2 * np.abs(self.z[:, 0]), rtol=1e-12)
        np.testing.assert_array_equal(ts.data["vdb_out"][1], self.data[:, 5])

    def test_a_file_of_probes_only(self):
        path = self.dir / "only.ac0"
        fixtures.write_binary(path, "2001", ["HERTZ", "vdb(out", "vp(out"], [2, 6, 5],
                              [([], self.data[:, [0, 5, 6]])], nprobe=2, layout="hspice")
        h = read_header(path)
        self.assertEqual((h.names, h.ncomplex), (["HERTZ", "vdb_out", "vp_out"], 0))
        np.testing.assert_array_equal(read_traces(path).data["vp_out"][0], self.data[:, 6])

    def test_text_report(self):
        from spice_result_parser import mcp_server
        text = mcp_server.format_text(api.extract(self.path, ["v(out", "vdb(out"], output="arrays"))
        lines = text.splitlines()
        self.assertEqual(lines[0], "ac | x: HERTZ in [100, 1e+09] | 36 points")
        self.assertEqual(lines[1], "complex traces are split into _Mag (linear magnitude) and _Phase (degrees)")
        stats = next(line.split() for line in lines if line.split()[:1] == ["vdb_out"])
        self.assertEqual(stats, ["vdb_out", f"{self.data[:, 5].min():.6g}", f"{self.data[:, 5].max():.6g}"])
        self.assertNotIn("vdb_out_Mag", text)


class TestDataSweeps(unittest.TestCase):
    """A .data sweep (.tran ... sweep data=pts) names its table before its columns
    (pts rval cval); each sweep starts with a value per column, none for the table."""

    def test_real_files(self):
        wv = HERE / "waveview"                  # HSPICE P-2019.06: two9.tr0 binary 9601, twoa.tr0 ASCII
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # read one value off, the sweeps would end in stray values
            binary, text = read_traces(wv / "two9.tr0"), read_traces(wv / "twoa.tr0")
        for ts in (binary, text):
            self.assertEqual(ts.header.sweep_params, ["rval", "cval"])
            self.assertEqual([len(a) for a in ts.data["TIME"]], [46, 46, 46])
            np.testing.assert_allclose(ts.sweep_values, [[1000, 1e-9], [2000, 1e-9], [2000, 2e-9]], rtol=1e-7)
            np.testing.assert_allclose(ts.data["TIME"][0][:4], [0, 2.5e-10, 5e-10, 1e-9], rtol=1e-7)
            np.testing.assert_array_equal(ts.data["v_in"][0][:4], [0, 0.25, 0.5, 1])     # the 1 ns rise of the pulse
        for name in binary.selected:
            for a, b in zip(text.data[name], binary.data[name]):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12, err_msg=name)
        info = api.list_traces(wv / "two9.tr0")
        self.assertEqual((info["sweeps"], info["sweep_params"]), (3, ["rval", "cval"]))

    def test_fixture_in_hspice_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.tr0"
            x = np.linspace(0, 1e-9, 5)
            sweeps = [([r, c], np.column_stack([x, x * r])) for r, c in ((1e3, 1e-9), (2e3, 2e-9))]
            fixtures.write_binary(path, "9601", ["TIME", "v(out", "rval", "cval"], [1, 1], sweeps, layout="hspice")
            self.assertIn(b"pts             rval            cval", path.read_bytes())
            ts = read_traces(path)
            self.assertEqual((ts.header.nsweepparam, ts.header.sweep_params), (2, ["rval", "cval"]))
            np.testing.assert_allclose(ts.data["v_out"][1], x * 2e3, rtol=1e-6)


class TestStrayValues(unittest.TestCase):
    def test_a_wrong_column_count_is_reported(self):
        # 7 values per point under a header that declares 5: the values before the
        # terminator do not divide into points, which the reader says instead of hiding
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.ac0"
            data = probe_ac_data(n=7)[2][:, :6]
            fixtures.write_binary(path, "2001", ["HERTZ", "v(a", "i(b"], [2, 1, 8], [([], data)])
            with self.assertWarnsRegex(RuntimeWarning, "sweep 0 ends with 2 values that do not make a "
                                                       "whole point of 5"):
                read_traces(path)


if __name__ == "__main__":
    unittest.main()
