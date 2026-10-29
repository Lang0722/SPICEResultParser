"""End-to-end tests of the WaveView engine against the real WaveView. Skipped by default.

On the Mac, with the Parallels VM "Windows 11" running:

    SRP_E2E_WAVEVIEW=1 PYTHONPATH=src python3 -m unittest discover -s test -p "test_waveview_e2e.py"

SRP_WAVEVIEW defaults to test/waveview/wv_vm.py, which forwards each WaveView run to the
VM with prlctl. The VM sees only the Mac home directory, so the temporary files go to a
directory under it (SRP_E2E_TMP, default ~/.cache/srp_e2e). On a machine with WaveView,
set SRP_WAVEVIEW to the wv binary instead.

The tests compare both engines on every file the Python reader can read, and check that
the WaveView engine reads the HSPICE post_version 2013 files, which the Python reader
rejects. Known differences are asserted as such, so that a change on either side shows.
"""
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).parent.resolve()
WV = HERE / "waveview"
ENABLED = os.environ.get("SRP_E2E_WAVEVIEW") == "1"

from spice_result_parser import api, measure, waveview  # noqa: E402
from spice_result_parser.reader import read_traces  # noqa: E402
from spice_result_parser.waveview import WaveViewError  # noqa: E402

_saved = {}


def setUpModule():
    if not ENABLED:
        raise unittest.SkipTest("set SRP_E2E_WAVEVIEW=1 to run the tests against the real WaveView")
    tmp = Path(os.environ.get("SRP_E2E_TMP", Path.home() / ".cache" / "srp_e2e"))
    tmp.mkdir(parents=True, exist_ok=True)
    _saved["tempdir"] = tempfile.tempdir
    _saved["env"] = mock.patch.dict(os.environ)
    _saved["env"].start()
    tempfile.tempdir = str(tmp)
    os.environ.setdefault("SRP_WAVEVIEW", str(WV / "wv_vm.py"))
    for name in ("SRP_ENGINE", "SRP_FAKE_WV"):
        os.environ.pop(name, None)


def tearDownModule():
    if "env" in _saved:
        _saved["env"].stop()
        tempfile.tempdir = _saved["tempdir"]


# Files the Python reader reads correctly, with the tolerance of the comparison of the
# values. 0 means equal. WaveView's 17-digit scan gives the exact values of binary files;
# of ASCII files it can be 1 ulp off the Python parse of the same text.
PARITY = {
    HERE / "test_9601.tr0": 0.0,
    HERE / "test_9601.sw0": 0.0,
    HERE / "test_2001.tr0": 0.0,
    HERE / "nutmeg_rc_bin.raw": 0.0,
    HERE / "nutmeg_rc_ascii.raw": 1e-15,
    WV / "rc9.tr0": 0.0,
    WV / "rc9.sw0": 0.0,
    WV / "rca.tr0": 1e-15,
    WV / "rca.sw0": 1e-15,
    WV / "odd.tr0": 0.0,            # p(rbus) and myexpr=par(...): WaveView lists power(rbus), par(myexpr)
    WV / "dc2.sw0": 0.0,            # sweep parameter 0:vdd
    WV / "ng_dc.raw": 0.0,          # ngspice: x is v(v-sweep), WaveView's xname says voltage
    WV / "dims.raw": 1e-15,         # Dimensions: 3,2
    WV / "named3_9.tr0": 0.0,       # pw=p(r1) gmx=lx7(m1) idm=i1(m1) vdiff=v(in,out): WaveView lists power(pw)...
    WV / "expr.tr0": 0.0,           # vd=v(in,out) dv=deriv(...) x1.loc=par(...)
    WV / "diff2_9.tr0": 0.0,        # v(x1.xl.n1,x1.xl.b): WaveView lists x1.xl.v(n1,b)
    WV / "l340_9.tr0": 0.0,         # 344 names: the header goes on in a second block
}
POST_2013 = ("rc.tr0", "rc.ac0", "rc.sw0", "two.tr0", "single.tr0", "single.ac0", "single.sw0")
MEASURES = (HERE / "test.mt0", WV / "rc.mt0", WV / "rc.ma0", WV / "rc.ms0", WV / "two.mt0", WV / "d1.mt0",
            WV / "mdup.mt0")          # mdup: measures named vmax (a .data parameter), index and temper


class Base(unittest.TestCase):
    def assert_values(self, got, want, rtol, atol=0.0, msg=""):
        want = np.asarray(want, dtype=np.float64)
        self.assertEqual(got.dtype, np.float64, msg)
        if rtol or atol:
            np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=msg)
        else:
            np.testing.assert_array_equal(got, want, err_msg=msg)

    def assert_same(self, got, want, rtol, atol=0.0, names=None):
        names = names or dict(zip(got.selected, want.selected))
        self.assertEqual(got.sweep_indices, want.sweep_indices)
        for g, w in names.items():
            self.assertEqual(len(got.data[g]), len(want.data[w]), g)
            for i, (a, b) in enumerate(zip(got.data[g], want.data[w])):
                self.assertEqual(a.size, b.size, f"{g} sweep {i}")
                self.assert_values(a, b, rtol, atol, f"{g} sweep {i}")


class TestParity(Base):
    def test_listing_and_values(self):
        for path, rtol in PARITY.items():
            with self.subTest(path.name):
                want_list = api.list_traces(path)
                got_list = api.list_traces(path, engine="waveview")
                self.assertEqual(got_list.pop("format"), "waveview")
                want_list.pop("format")
                if path.suffix == ".raw":                      # WaveView opens plot 0 only
                    self.assertEqual(got_list.pop("plots"), want_list.pop("plots")[:1])
                self.assertEqual(got_list, want_list)
                got, want = waveview.read_traces(path), read_traces(path)
                self.assertEqual((got.selected, got.sweep_values), (want.selected, want.sweep_values))
                self.assert_same(got, want, rtol)

    def test_repeated_x_values_are_kept(self):
        # 1170 of the 2605 points of test_9601.tr0 repeat the previous x value (float32 time);
        # sx_export_data would drop most of them, the scan keeps them all.
        ts = waveview.read_traces(HERE / "test_9601.tr0", ["v(vo"])
        x = ts.data["TIME"][0]
        self.assertEqual(x.size, 2605)
        self.assertEqual(int((np.diff(x) == 0).sum()), 1170)

    def test_ac_formats(self):
        # 9601: WaveView's real and imaginary parts are about 1 ulp of the magnitude off
        # (absolute); the Python engine computes magnitude, phase and dB in float32.
        # Nutmeg: double precision.
        cases = ((HERE / "test_9601.ac0", {"realimag": (1e-13, 0.0), "magphase": (5e-7, 0.0), "db": (5e-7, 1e-5)}),
                 (WV / "multi_bin.raw", {"realimag": (1e-13, 1e-16), "magphase": (1e-13, 1e-12), "db": (1e-13, 1e-12)}),
                 (WV / "multi_asc.raw", {"realimag": (1e-13, 1e-16), "magphase": (1e-13, 1e-12), "db": (1e-13, 1e-12)}))
        for path, formats in cases:
            for ac_format, (rtol, atol) in formats.items():
                with self.subTest(f"{path.name} {ac_format}"):
                    got = waveview.read_traces(path, ac_format=ac_format)
                    want = read_traces(path, ac_format=ac_format)
                    self.assertEqual(got.selected[1:], want.selected[1:])
                    self.assert_same(got, want, rtol, atol)

    def test_nutmeg_x_name(self):
        # WaveView's xname of a Nutmeg plot is the type of variable 0 (voltage); the engine
        # takes the name from the header.
        got = api.list_traces(WV / "multi_bin.raw", engine="waveview")
        self.assertEqual((got["x"], got["analysis"], got["plots"]), ("frequency", "ac", ["AC Analysis"]))
        self.assertEqual(api.list_traces(WV / "multi_bin.raw")["x"], "frequency")

    def test_xrange_bounds_of_a_9601_file(self):
        # Both engines compare the float32 x of a 9601 file with float32 bounds.
        for window, size in (((0.003, 0.006), 786), ((0.005, 0.01), 1301)):
            with self.subTest(window):
                got = waveview.read_traces(HERE / "test_9601.tr0", ["v(vo"], xrange=window)
                want = read_traces(HERE / "test_9601.tr0", ["v(vo"], xrange=window)
                self.assertEqual(got.data["TIME"][0].size, size)
                self.assert_same(got, want, 0.0)

    def test_power_and_par_selection(self):
        for names in (["p(rbus)"], ["p_rbus"], ["myexpr"]):
            with self.subTest(names):
                got = waveview.read_traces(WV / "odd.tr0", names)
                want = read_traces(WV / "odd.tr0", names)
                self.assertEqual(got.selected, want.selected)
                self.assert_same(got, want, 0.0)

    def test_selection_sweeps_and_xrange(self):
        path = WV / "rc9.tr0"
        x = read_traces(path).data["TIME"][1]
        window = (float(x[10]), float(x[40]))
        got = waveview.read_traces(path, ["v(out)", "i_*"], sweeps=[2, 1], xrange=window)
        want = read_traces(path, ["v(out)", "i_*"], sweeps=[2, 1], xrange=window)
        self.assertEqual((got.selected, got.sweep_values), (want.selected, want.sweep_values))
        self.assert_same(got, want, 0.0)

    def test_deep_hierarchy(self):
        # The ASCII header wraps v(xt.xm.xl.n1 over two lines. The Python reader takes the
        # two parts for two names, so each later name labels the column before it and i(vin)
        # is lost. WaveView gives every name. The values agree column by column.
        got = waveview.read_traces(WV / "deep.tr0")
        want = read_traces(WV / "deep.tr0")
        self.assertEqual(got.selected[:7], want.selected[:7])
        self.assertEqual(want.selected[7:9], ["v_xt_xm_", "xl_n1"])
        self.assertEqual(got.selected[7], "v_xt_xm_xl_n1")
        self.assertEqual(want.selected[9:], got.selected[8:11])
        self.assertEqual(got.selected[-1], "i_vin")
        self.assertNotIn("i_vin", want.selected)
        self.assert_same(got, want, 1e-15, names=dict(zip(got.selected, want.selected)))

    def test_data_sweep_parameters(self):
        # The Python reader takes the .data table name (pts) for a sweep parameter: three
        # parameters [1000, 1e-09, 0] and shifted values. WaveView: rval, cval.
        for path in (WV / "two9.tr0", WV / "twoa.tr0"):
            with self.subTest(path.name):
                self.assertEqual(api.list_traces(path)["sweep_params"], ["pts", "rval", "cval"])
                got = waveview.read_traces(path, ["v(x1.mid"])
                self.assertEqual(got.header.sweep_params, ["rval", "cval"])
                self.assertEqual(got.sweep_values, [[1000.0, 1e-9], [2000.0, 1e-9], [2000.0, 2e-9]])
                np.testing.assert_allclose(read_traces(path, ["v(x1.mid"]).sweep_values[0], [1000.0, 1e-9, 0.0],
                                           rtol=1e-7)
                self.assertEqual([a.size for a in got.data["TIME"]], [46, 46, 46])

    def test_noise_columns(self):
        # WaveView lists outnoise as noise(outnoise). The Python reader takes these real
        # columns for complex pairs (the known AC real-column error), but its names are right.
        got = waveview.read_traces(WV / "noi.ac0", ["outnoise", "innoise"])
        self.assertEqual(got.selected, ["HERTZ", "outnoise", "innoise"])
        self.assertEqual(read_traces(WV / "noi.ac0").header.raw_names[-2:], ["outnoise", "innoise"])
        np.testing.assert_allclose(got.data["outnoise"][0][:3], 1.6465e-17, rtol=1e-4)    # 4kTR, 1 kOhm

    def test_analysis_of_an_ac_file_without_its_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acreal.bin"
            shutil.copy(WV / "acreal_9.ac0", path)
            self.assertEqual(api.list_traces(path, engine="waveview")["analysis"], "ac")

    def test_real_probe_in_an_ac_file(self):
        # vdb(out) is one real value per point. The Python reader takes it for a complex
        # pair, so its columns shift; WaveView lists it as one real trace.
        for path in (WV / "rc9.ac0", WV / "rca.ac0"):
            with self.subTest(path.name):
                self.assertIn("vdb_out_Mag", api.list_traces(path)["traces"])
                self.assertEqual(read_traces(path).data["HERTZ"][0][1], 0.0)    # a shifted column
                got = waveview.read_traces(path, ["v(out)", "vdb(out)"], ac_format="db")
                self.assertEqual(got.selected, ["HERTZ", "v_out_dB", "v_out_Phase", "vdb_out"])
                for i in range(2):
                    self.assertEqual(got.data["HERTZ"][i].size, 51)
                    np.testing.assert_allclose(got.data["v_out_dB"][i], got.data["vdb_out"][i], atol=2e-5)


class TestPostVersion2013(Base):
    def test_python_rejects_and_waveview_reads(self):
        for name in POST_2013:
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "unsupported post_version '2013'"):
                    api.list_traces(WV / name)
                listing = api.list_traces(WV / name, engine="waveview")
                ts = waveview.read_traces(WV / name)
                self.assertEqual(ts.selected[1:], listing["traces"])
                self.assertTrue(all(a.size > 0 for a in ts.data[ts.selected[0]]))

    def test_same_values_as_the_ascii_twin(self):
        # rca.* is the same netlist with post=2 (ASCII, 7 digits), read by both engines.
        for binary, ascii_ in (("rc.tr0", "rca.tr0"), ("rc.sw0", "rca.sw0")):
            with self.subTest(binary):
                got = waveview.read_traces(WV / binary)
                want = read_traces(WV / ascii_)
                self.assertEqual((got.selected, got.sweep_values), (want.selected, want.sweep_values))
                self.assert_same(got, want, 1e-6, 1e-12)
        for binary, other in (("rc.ac0", "rca.ac0"), ("two.tr0", "twoa.tr0")):
            with self.subTest(binary):
                got = waveview.read_traces(WV / binary)
                want = waveview.read_traces(WV / other)
                self.assertEqual((got.selected, got.sweep_values), (want.selected, want.sweep_values))
                self.assert_same(got, want, 1e-5, 1e-9)

    def test_text_report(self):
        from spice_result_parser import mcp_server

        text = mcp_server.extract(str(WV / "rc.tr0"), ["v(out)"], engine="waveview")
        self.assertTrue(text.startswith("transient | x: TIME | 226 points in 3 sweeps"), text)
        self.assertIn("| engine: waveview", text.splitlines()[0])
        self.assertIn("-- sweep 2: rval=4000 --", text)


class TestMeasures(Base):
    def test_parity(self):
        for path in MEASURES:
            with self.subTest(path.name):
                got = measure.read_measures(path, engine="waveview")
                want = measure.read_measures(path)
                self.assertEqual(got.names, [n for n in want.names if n != "alter#"])
                self.assertEqual((got.title, got.params, got.rows, got.failed),
                                 (want.title, want.params, want.rows, want.failed))
                for name in got.names:
                    self.assert_values(got.values[name], want.values[name], 1e-15, msg=name)

    def test_param_count_includes_index(self):
        for name, params in (("two.mt0", ["rval", "cval"]), ("d1.mt0", ["rval"]), ("mc.mt0", [])):
            with self.subTest(name):
                self.assertEqual(measure.read_measures(WV / name, engine="waveview").params, params)
        # The Python reader misreads the $OPTION lines of a Monte Carlo file (a known error).
        mc = measure.read_measures(WV / "mc.mt0", engine="waveview")
        self.assertEqual((mc.title, mc.names, mc.rows), ("* monte carlo", ["index", "vmax", "tr", "temper"], 3))

    def test_srp_signals_of_the_caller_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "signals.txt").write_text("vmax\n")
            with mock.patch.dict(os.environ, {"SRP_SIGNALS": str(Path(tmp, "signals.txt"))}):
                ms = measure.read_measures(WV / "rc.mt0", engine="waveview")
        self.assertEqual((ms.names, ms.params), (["rval", "tdel", "vmax", "never", "temper"], ["rval"]))


class TestPaths(Base):
    def test_spaces_and_accents(self):
        # sx_open_sim_file_read splits its argument as a Tcl list, and it cannot open a
        # non-ASCII path; the scripts pass one list element and fall back to the 8.3 short
        # name. [list] escapes a lone brace with a backslash, which WaveView changes to "/".
        for folder in ("a b", "caf\u00e9", "two  spaces", "a{b", "x }y", "q{r}s"):
            with self.subTest(folder), tempfile.TemporaryDirectory() as tmp:
                where = Path(tmp) / folder
                where.mkdir()
                for name in ("rc9.tr0", "rc.mt0"):
                    shutil.copy(WV / name, where / name)
                got, want = waveview.read_traces(where / "rc9.tr0", ["v(out"]), read_traces(where / "rc9.tr0", ["v(out"])
                self.assertEqual((got.selected, got.sweep_values), (want.selected, want.sweep_values))
                self.assert_same(got, want, 0.0)
                self.assertEqual(measure.read_measures(where / "rc.mt0", engine="waveview").rows, 3)


    def test_path_of_260_characters_or_more(self):
        # WaveView cannot open such a Windows path; the scripts use the 8.3 short name.
        with tempfile.TemporaryDirectory() as tmp:
            where = Path(tmp, *(c * 60 for c in "defg"))
            where.mkdir(parents=True)
            shutil.copy(WV / "rc9.tr0", where / "rc9.tr0")
            self.assertGreater(len(str(where / "rc9.tr0")), 260)
            got = waveview.read_traces(where / "rc9.tr0", ["v(out"])
            self.assert_same(got, read_traces(where / "rc9.tr0", ["v(out"]), 0.0)


class TestFailures(Base):
    def test_file_that_waveview_cannot_open(self):
        with self.assertRaisesRegex(WaveViewError, "(?s)cannot open .*bogus.tr0.*unknown waveform format"):
            api.list_traces(WV / "bogus.tr0", engine="waveview")

    def test_unknown_signal(self):
        with self.assertRaisesRegex(WaveViewError, "unknown signal v\\(nope\\)"):
            waveview._run("extract.tcl", str(HERE / "test_9601.tr0"), ["v(vo)", "v(nope)"])

    def test_tcl_error_without_srp_done(self):
        # The exit code does not show success (0 after an error that the script catches):
        # only the missing srp_done shows the error.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(Path(waveview.TCL_DIR) / "list_traces.tcl", tmp)
            (Path(tmp) / "boom.tcl").write_text("set out [open $env(SRP_OUT) w]\nputs $out srp_listing\n"
                                                "sx_no_such_command\nputs $out srp_done\nclose $out\nexit\n")
            with mock.patch.object(waveview, "TCL_DIR", tmp):
                with self.assertRaisesRegex(WaveViewError, "(?s)no srp_done.*Tcl Error.*sx_no_such_command"):
                    waveview._run("boom.tcl", str(HERE / "test_9601.tr0"))
                self.assertIn("srp_listing", waveview._run("list_traces.tcl", str(HERE / "test_9601.tr0"))[0])

    def test_two_signals_of_one_name(self):
        with self.assertRaisesRegex(WaveViewError, "two signals named v\\(out\\)"):
            api.list_traces(WV / "dupnut.raw", engine="waveview")

    def test_file_without_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zeros.tr0"
            path.write_bytes(b"\0" * 64)
            with self.assertRaisesRegex(WaveViewError, "finds no signals in the file"):
                api.list_traces(path, engine="waveview")

    def test_truncated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            swept = Path(tmp) / "trunc_rc9.tr0"
            swept.write_bytes((WV / "rc9.tr0").read_bytes()[:3019])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ts = waveview.read_traces(swept, ["v(out"])
            self.assertEqual((ts.truncated, ts.sweep_values, ts.header.sweep_count_hint), (True, [[1000.0]], 3))
            self.assertTrue(any("probably truncated" in str(w.message) for w in caught))
            for name, source in (("cut_rc9.tr0", WV / "rc9.tr0"), ("cut_rc.tr0", WV / "rc.tr0")):
                cut = Path(tmp) / name                  # cut inside sweep 0: no complete sweep
                cut.write_bytes(source.read_bytes()[:1200])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = waveview.read_traces(cut, ["v(out"])
                self.assertEqual((ts.truncated, ts.sweep_values, ts.data["v_out"]), (True, [], []), name)
            flat = Path(tmp) / "trunc_9601.tr0"
            flat.write_bytes((HERE / "test_9601.tr0").read_bytes()[:26328])
            with self.assertRaisesRegex(WaveViewError, "points for .* x values in trace 0; the file is probably "
                                                       "truncated"):
                waveview.read_traces(flat)

    def test_nothing_is_written_beside_the_scripts(self):
        # WaveView writes ace_tcl.log beside the script it runs; the engine runs a copy.
        waveview.read_header(HERE / "test_9601.tr0")
        self.assertEqual(sorted(os.listdir(waveview.TCL_DIR)),
                         ["extract.tcl", "list_traces.tcl", "read_measures.tcl"])


class TestMcp(Base):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from spice_result_parser import mcp_server

        self.m = mcp_server

    def test_csv_is_byte_identical(self):
        for path in (HERE / "test_9601.tr0", HERE / "nutmeg_rc_bin.raw", WV / "rc9.tr0"):
            with self.subTest(path.name), tempfile.TemporaryDirectory() as tmp:
                self.m.extract(str(path), output="csv", dest=f"{tmp}/py.csv")
                out = self.m.extract(str(path), output="csv", dest=f"{tmp}/wv.csv", engine="waveview")
                self.assertEqual(out["engine"], "waveview")
                self.assertEqual(Path(f"{tmp}/wv.csv").read_bytes(), Path(f"{tmp}/py.csv").read_bytes())

    def test_text_and_measures(self):
        text = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], engine="waveview")
        head, _, rest = text.partition("\n")
        self.assertEqual(head.replace(" | engine: waveview", "") + "\n" + rest,
                         self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"]))
        report = self.m.read_measures(str(WV / "rc.mt0"), engine="waveview")
        self.assertTrue(report.startswith("measures | 3 rows | params: rval | engine: waveview"), report)
        self.assertIn("failed: never in 3 of 3 rows", report)


if __name__ == "__main__":
    unittest.main()
