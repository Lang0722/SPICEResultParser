"""Tests for the WaveView engine (waveview.py) and the engine switch, with a fake wv.

test/waveview/fake_wv.py stands in for WaveView on any OS: it serves real output of
the Tcl scripts, recorded on WaveView O-2018.09-SP2 (Windows 11, CRLF line ends), from
test/waveview/canned/. The HSPICE files in test/waveview/ come from HSPICE P-2019.06
running the .sp netlists beside them: rc/two/single/names2 are binary post_version 2013
(the Python reader rejects them), rc9/two9/odd/dc2/names2_9/big9 binary 9601, rca/twoa/deep
ASCII, post3 an HSPICE post=3 file, and d1.mt0/mc.mt0 measure files. ng_dc.raw comes from
ngspice-47; dims.raw, dupnut.raw, nutinf.raw, nutbig.raw and latin1.raw are written by hand.
The canned outputs of cut files (cut_*, trunc_*) come from copies cut as the tests cut
them. The tests against the real WaveView are in test_waveview_e2e.py.
"""
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).parent.resolve()
WV = HERE / "waveview"
FAKE = WV / "fake_wv.py"

from spice_result_parser import api, measure, nutmeg, waveview  # noqa: E402
from spice_result_parser.reader import read_header, read_traces  # noqa: E402
from spice_result_parser.waveview import WaveViewError  # noqa: E402

TR, AC, SW = HERE / "test_9601.tr0", HERE / "test_9601.ac0", HERE / "test_9601.sw0"
NUTMEG = HERE / "nutmeg_rc_bin.raw"


class FakeWaveView(unittest.TestCase):
    """Every WaveView run goes to fake_wv.py; SRP_* variables start unset."""

    def setUp(self):
        patcher = mock.patch.object(waveview, "wv_command", return_value=[sys.executable, str(FAKE)])
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("SRP_ENGINE", "SRP_WAVEVIEW", "SRP_WAVEVIEW_TIMEOUT", "SRP_FAKE_WV"):
            os.environ.pop(name, None)

    def fake(self, mode):
        os.environ["SRP_FAKE_WV"] = mode

    def assert_same_traces(self, ts, ref, rtol=0.0, atol=0.0):
        """Same names, sweeps and values; WaveView arrays are float64, 9601 arrays float32."""
        self.assertEqual(ts.selected, ref.selected)
        self.assertEqual(ts.sweep_indices, ref.sweep_indices)
        self.assertEqual(ts.truncated, ref.truncated)
        for name in ts.selected:
            self.assertEqual(len(ts.data[name]), len(ref.data[name]), name)
            for got, want in zip(ts.data[name], ref.data[name]):
                self.assertEqual(got.dtype, np.float64)
                if rtol or atol:
                    np.testing.assert_allclose(got, want.astype(np.float64), rtol=rtol, atol=atol, err_msg=name)
                else:
                    np.testing.assert_array_equal(got, want.astype(np.float64), err_msg=name)


class TestEngineSwitch(FakeWaveView):
    def test_resolve_engine(self):
        self.assertEqual(waveview.resolve_engine(), "python")
        self.assertEqual(waveview.resolve_engine("waveview"), "waveview")
        os.environ["SRP_ENGINE"] = "waveview"
        self.assertEqual(waveview.resolve_engine(), "waveview")
        self.assertEqual(waveview.resolve_engine("python"), "python")     # one call overrides the variable
        os.environ["SRP_ENGINE"] = " "
        self.assertEqual(waveview.resolve_engine(), "python")

    def test_unknown_engine_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "engine must be one of .*'hspice'"):
            waveview.resolve_engine("hspice")
        with self.assertRaisesRegex(ValueError, "engine must be one of"):
            api.extract(TR, engine="WaveView")
        os.environ["SRP_ENGINE"] = "wv"
        with self.assertRaisesRegex(ValueError, "SRP_ENGINE must be one of .*'wv'"):
            api.list_traces(TR)

    def test_default_engine_never_starts_waveview(self):
        with mock.patch.object(waveview, "run_script") as run:
            self.assertEqual(api.list_traces(TR)["format"], "9601")
            self.assertEqual(api.extract(TR).header.version, "9601")
            self.assertEqual(measure.read_measures(HERE / "test.mt0").names[-1], "alter#")
        run.assert_not_called()

    def test_engine_argument(self):
        self.assertEqual(api.list_traces(TR, engine="waveview")["format"], "waveview")
        self.assertEqual(api.extract(TR, ["v(vo"], engine="waveview").header.version, "waveview")
        self.assertNotIn("alter#", measure.read_measures(HERE / "test.mt0", engine="waveview").names)

    def test_srp_engine_switches_every_reader(self):
        os.environ["SRP_ENGINE"] = "waveview"
        self.assertEqual(api.list_traces(TR)["format"], "waveview")
        self.assertEqual(api.extract(TR, ["v(vo"]).header.version, "waveview")
        self.assertNotIn("alter#", measure.read_measures(HERE / "test.mt0").names)
        self.assertEqual(api.list_traces(TR, engine="python")["format"], "9601")


class TestFindWv(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("SRP_WAVEVIEW", "SRP_WAVEVIEW_TIMEOUT"):
            os.environ.pop(name, None)

    def test_srp_waveview_comes_first(self):
        os.environ["SRP_WAVEVIEW"] = "/opt/wv/bin/wv"
        with mock.patch("shutil.which", return_value="/usr/bin/wv"):
            self.assertEqual(waveview.find_wv(), "/opt/wv/bin/wv")
            del os.environ["SRP_WAVEVIEW"]
            self.assertEqual(waveview.find_wv(), "/usr/bin/wv")
            self.assertEqual(waveview.wv_command(), ["/usr/bin/wv"])

    def test_missing_binary_names_both_options(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(WaveViewError, "SRP_WAVEVIEW.*PATH"):
                waveview.find_wv()
            with self.assertRaisesRegex(WaveViewError, "WaveView not found"):
                api.list_traces(TR, engine="waveview")

    def test_binary_that_does_not_start(self):
        os.environ["SRP_WAVEVIEW"] = str(HERE / "no_such_dir" / "wv")
        with self.assertRaisesRegex(WaveViewError, "cannot start WaveView"):
            waveview.run_script("list_traces.tcl", {})

    def test_run_script_arguments(self):
        os.environ["SRP_WAVEVIEW"] = "wv"
        os.environ["SRP_SIGNALS"] = "mine.txt"              # the caller's own: it must not reach the script
        os.environ["SRP_OUT"] = "mine_out.txt"
        with mock.patch("subprocess.Popen") as popen:
            proc = popen.return_value
            proc.communicate.return_value = (b"console\r\n", None)
            self.assertEqual(waveview.run_script("extract.tcl", {"SRP_IN": "a.tr0"}), "console\r\n")
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["wv", "-ace_no_gui", os.path.join(waveview.TCL_DIR, "extract.tcl")])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["env"]["SRP_IN"], "a.tr0")
        self.assertNotIn("SRP_SIGNALS", kwargs["env"])
        self.assertNotIn("SRP_OUT", kwargs["env"])
        self.assertEqual(kwargs["env"]["SRP_WAVEVIEW"], "wv")             # the rest of the environment stays
        self.assertEqual(kwargs["start_new_session"], os.name != "nt")
        self.assertNotIn("shell", kwargs)
        proc.communicate.assert_called_once_with(timeout=600)

    def test_timeout_variable(self):
        self.assertEqual(waveview._timeout(), waveview.DEFAULT_TIMEOUT)
        os.environ["SRP_WAVEVIEW_TIMEOUT"] = "2.5"
        self.assertEqual(waveview._timeout(), 2.5)
        os.environ["SRP_WAVEVIEW_TIMEOUT"] = "1e6"
        self.assertEqual(waveview._timeout(), waveview.MAX_TIMEOUT)
        for bad in ("soon", "0", "-1", "nan", "inf", "1e9"):     # subprocess cannot wait 1e9 s
            os.environ["SRP_WAVEVIEW_TIMEOUT"] = bad
            with self.assertRaisesRegex(ValueError, "SRP_WAVEVIEW_TIMEOUT"):
                waveview._timeout()


class TestNames(unittest.TestCase):
    def test_hierarchical_names_become_the_file_names(self):
        self.assertEqual(waveview.raw_name("x1.v(mid)", False), "v(x1.mid")
        self.assertEqual(waveview.raw_name("xt.xm.xl.v(n1)", False), "v(xt.xm.xl.n1")
        self.assertEqual(waveview.raw_name("x1.i(r1)", False), "i(x1.r1")
        self.assertEqual(waveview.raw_name("x1.v(mid)", True), "v(x1.mid)")     # Nutmeg keeps the paren
        # WaveView takes out the path that all nodes share; v(x1.xl.n1,b) has no such path.
        self.assertEqual(waveview.raw_name("x1.xl.v(n1,b)", False), "v(x1.xl.n1,x1.xl.b")
        self.assertEqual(waveview.raw_name("x1.xl.vm(n1,b)", False), "vm(x1.xl.n1,x1.xl.b")
        self.assertEqual(waveview.raw_name("v(x1.xl.n1,b)", False), "v(x1.xl.n1,b")

    def test_flat_names(self):
        self.assertEqual(waveview.raw_name("v(0)", False), "v(0")
        self.assertEqual(waveview.raw_name("i(vs)", False), "i(vs")
        self.assertEqual(waveview.raw_name("vdb(out)", False), "vdb(out")
        self.assertEqual(waveview.raw_name("v(out)", True), "v(out)")
        self.assertEqual(waveview.raw_name("rval", False), "rval")
        self.assertEqual(waveview.raw_name("a.b", False), "a.b")

    def test_power_and_par_probes(self):
        # .probe p(r9) and vm=par('v(out)*2'): WaveView lists power(r9) and par(vm).
        self.assertEqual(waveview.raw_name("power(r9)", False), "p(r9")
        self.assertEqual(waveview.raw_name("x1.power(r1)", False), "p(x1.r1")
        self.assertEqual(waveview.raw_name("par(vm)", False), "vm")
        self.assertEqual(waveview.raw_name("par(x1.vm)", False), "x1.vm")
        self.assertEqual(waveview.raw_name("par(v(out)*2)", False), "par(v(out)*2")    # not a name: kept
        self.assertEqual(waveview.raw_name("power(r9)", True), "power(r9)")            # Nutmeg: as WaveView
        self.assertEqual(waveview.raw_name("par(vm)", True), "par(vm)")
        # dv=deriv('...'), ig=integ('...') and the .noise columns: the file has the bare name.
        for wv_name, raw in (("deriv(dv)", "dv"), ("integ(ig)", "ig"), ("noise(outnoise)", "outnoise")):
            self.assertEqual(waveview.raw_name(wv_name, False), raw)
        # Names with $ # [ ] < > (my$x=par(...)): HSPICE allows them; an operator is not a name.
        for wv_name, raw in (("par(my$x)", "my$x"), ("par(q[0])", "q[0]"), ("deriv(dv$)", "dv$"),
                             ("integ(ig#)", "ig#"), ("par(w<1>)", "w<1>"), ("par(v(in)*3)", "par(v(in)*3"),
                             ("par(rval*2)", "par(rval*2")):
            self.assertEqual(waveview.raw_name(wv_name, False), raw)

    def test_header_names(self):
        # The file's bare name of a named probe or expression, and an unnamed expression.
        for unfolded, own in (("v(vo)", "vo"), ("v(xt.vo)", "xt.vo"), ("power(xt.pw)", "xt.pw"),
                              ("par(my$x)", "my$x"), ("par(rv)", "par(rv"), ("v(out)", "v(out")):
            self.assertTrue(waveview._is_own_name(unfolded, own), (unfolded, own))
        for unfolded, own in (("v(x1.vo)", "vo"), ("v(a)", "v(b"), ("power(r9)", "p(r9"), ("v(xt.inner)", "1:inner")):
            self.assertFalse(waveview._is_own_name(unfolded, own), (unfolded, own))
        self.assertEqual(waveview._unfold("xt.x2.v(deep)"), "v(xt.x2.deep)")

    def test_sweep_conditions(self):
        # One value per parameter in parameter order, as %g: 'M' (mega) of the withname
        # form cannot occur, and a parameter name may hold ':' (.dc vin ... vdd ...: 0:vdd).
        self.assertEqual(waveview._condition_values(["2000", "2e-09"], ["rval", "cval"], "f"), [2000.0, 2e-9])
        self.assertEqual(waveview._condition_values(["1.5e+06", "-2.5e+06"], ["aa", "bb"], "f"), [1.5e6, -2.5e6])
        self.assertEqual(waveview._condition_values(["1.5"], ["0:vdd"], "f"), [1.5])
        with self.assertRaisesRegex(WaveViewError, "wrote 1 sweep values for the 2 sweep parameters"):
            waveview._condition_values(["1000"], ["rval", "cval"], "f")
        with self.assertRaisesRegex(WaveViewError, "cannot read sweep value 'fast'"):
            waveview._condition_values(["fast"], ["corner"], "f")

    def test_windows_nan_and_infinity(self):
        values = waveview._floats("1.0\n1.#QNAN\n-1.#IND\n1.#INF\n-1.#INF\nNAN\n1.0e-009\n", "f")
        np.testing.assert_array_equal(np.isnan(values), [False, True, True, False, False, True, False])
        self.assertEqual(values[3], np.inf)
        self.assertEqual(values[4], -np.inf)
        self.assertEqual(values[6], 1e-9)
        with self.assertRaisesRegex(WaveViewError, "not a number"):
            waveview._floats("1.0\nbogus\n", "f")

    def test_floats_of_plain_columns(self):
        np.testing.assert_array_equal(waveview._floats("\n1.0\n2.5e-009\n-3\n", "f"), [1.0, 2.5e-9, -3.0])
        np.testing.assert_array_equal(waveview._floats("\n1 2\n3 4\n", "f", 2), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(np.isnan(waveview._floats("\n1 1.#QNAN\n-1.#IND 4\n", "f", 2)),
                                      [False, True, True, False])
        with self.assertRaisesRegex(WaveViewError, "wrote 3 values where 4 were expected \\(2 per line\\)"):
            waveview._floats("\n1 2\n3\n", "f", 2)
        for empty in ("", "\n", " \n\n"):                     # np.fromstring alone gives [-1.0] here
            self.assertEqual(waveview._floats(empty, "f").size, 0, repr(empty))
        self.assertEqual(waveview._floats("1.2345678901234567e-05", "f")[0], 1.2345678901234567e-05)

    def test_floats_do_not_rely_on_the_warnings_filter(self):
        # NumPy 2.1 only warns about a value it cannot read, and returns the values before it.
        # The warnings filter is process-wide: another thread can set it to "ignore" and undo
        # a filter that this thread sets (simplefilter does nothing here).
        def warns(text, dtype, sep):
            warnings.warn("string or file could not be read to its end due to unmatched data", DeprecationWarning)
            return np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch.object(waveview.np, "fromstring", warns), \
                    mock.patch.object(waveview.warnings, "simplefilter"):
                with self.assertRaisesRegex(WaveViewError, "not a number"):
                    waveview._floats("\n1.0\nbogus\n3.0\n", "f")


class TestListTraces(FakeWaveView):
    def test_same_as_python_on_hspice_files(self):
        for path in (TR, AC, SW):
            want = api.list_traces(path)
            got = api.list_traces(path, engine="waveview")
            self.assertEqual(got.pop("format"), "waveview")
            want.pop("format")
            self.assertEqual(got, want, path.name)

    def test_ac_format_names(self):
        got = api.list_traces(AC, ac_format="realimag", engine="waveview")
        self.assertEqual(got["traces"], api.list_traces(AC, ac_format="realimag")["traces"])
        self.assertEqual(got["traces"][:2], ["v_0_Re", "v_0_Im"])

    def test_header_fields(self):
        h = waveview.read_header(TR)
        self.assertEqual((h.version, h.is_binary, h.type_codes, h.dtype), ("waveview", False, [], np.float64))
        self.assertEqual(h.col_raw_names, read_header(TR).col_raw_names)
        self.assertEqual(h.raw_names, read_header(TR).raw_names)

    def test_nutmeg_lists_plot_0_only(self):
        want = api.list_traces(NUTMEG)
        got = api.list_traces(NUTMEG, engine="waveview")
        self.assertEqual(got["plots"], ["Transient Analysis"])
        self.assertEqual(want["plots"][0], "Transient Analysis")
        for key in ("analysis", "x", "traces", "sweep_params", "sweep_count_hint", "plot"):
            self.assertEqual(got[key], want[key], key)

    def test_nutmeg_x_name_comes_from_the_header(self):
        # WaveView's xname is the type of variable 0 (voltage), not its name.
        for path, x in ((WV / "multi_bin.raw", "frequency"), (WV / "ng_dc.raw", "v_v-sweep")):
            with self.subTest(path.name):
                got = api.list_traces(path, engine="waveview")
                want = api.list_traces(path)
                self.assertEqual(want["x"], x)
                self.assertEqual(got.pop("plots"), want.pop("plots")[:1])
                self.assertEqual({**got, "format": ""}, {**want, "format": ""})

    def test_nutmeg_dimensions(self):
        got = api.list_traces(WV / "dims.raw", engine="waveview")
        self.assertEqual(got["sweep_count_hint"], 3)
        self.assertEqual({**got, "format": ""}, {**api.list_traces(WV / "dims.raw"), "format": ""})

    def test_power_and_par_names_match_the_python_engine(self):
        got = api.list_traces(WV / "odd.tr0", engine="waveview")
        want = api.list_traces(WV / "odd.tr0")
        self.assertEqual(got["traces"], want["traces"])
        self.assertIn("p_rbus", got["traces"])
        self.assertEqual(got["traces"][-1], "myexpr")
        self.assertEqual(waveview.read_header(WV / "odd.tr0").raw_names, read_header(WV / "odd.tr0").raw_names)

    def test_named_probes_match_the_python_engine(self):
        # .probe pw=p(r1) gmx=lx7(m1) vd=v(in,out) dv=deriv(...): the file has the bare names,
        # WaveView lists power(pw), lx(gmx), v(vd), deriv(dv).
        for name, traces in (("named3_9.tr0", ["pw", "gmx", "idm", "vdiff"]),
                             ("expr.tr0", ["par_v", "vd", "dv", "x1_loc", "v_out", "i_x1_r1", "v_in", "v_x1_n1"])):
            with self.subTest(name):
                got = api.list_traces(WV / name, engine="waveview")
                self.assertEqual(got["traces"], traces)
                self.assertEqual(got["traces"], api.list_traces(WV / name)["traces"])
                self.assertEqual(waveview.read_header(WV / name).raw_names, read_header(WV / name).raw_names)

    def test_hierarchical_named_probes_and_expressions(self):
        # xt.pw=p(xt.x2.r2): WaveView lists xt.power(pw). par('rval'): par(rval), as a named rval=par().
        want = ["top", "xt_x2_deep", "xt_pw", "xt_io", "my$x", "q[0]", "dv$", "par_rval", "par_temper", "par_v",
                "v_in"]
        self.assertEqual(api.list_traces(WV / "names2_9.tr0")["traces"], want)
        self.assertEqual(api.list_traces(WV / "names2_9.tr0", engine="waveview")["traces"], want)
        ts = waveview.read_traces(WV / "names2_9.tr0", ["xt.pw", "par(rval"])
        self.assertEqual(ts.selected, ["TIME", "xt_pw", "par_rval"])
        # Without the header (post_version 2013): a named probe keeps WaveView's name, and an
        # unnamed par('rval') cannot be told from rval=par(...).
        got = api.list_traces(WV / "names2.tr0", engine="waveview")["traces"]
        self.assertEqual(got, ["v_top", "v_xt_x2_deep", "p_xt_pw", "i_xt_io", "my$x", "q[0]", "dv$", "rval", "temper",
                               "par_v", "v_in"])

    def test_hierarchical_differential_probes(self):
        # WaveView lists v(x1.xl.n1,x1.xl.b) as x1.xl.v(n1,b), and v(x1.xl.n1,b) as it is.
        got = api.list_traces(WV / "diff2_9.tr0", engine="waveview")
        self.assertEqual(got["traces"][:2], ["v_x1_xl_n1,x1_xl_b", "v_x1_xl_n1,b"])
        self.assertEqual(got["traces"], api.list_traces(WV / "diff2_9.tr0")["traces"])

    def test_noise_columns(self):
        got = api.list_traces(WV / "noi.ac0", engine="waveview")
        self.assertEqual(got["traces"][-2:], ["outnoise", "innoise"])     # WaveView: noise(outnoise)
        self.assertEqual(read_header(WV / "noi.ac0").raw_names[-2:], ["outnoise", "innoise"])

    def test_analysis_comes_from_the_header(self):
        # An AC file with real signals only, without an .ac# extension: the header says ac.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acreal.bin"
            path.write_bytes((WV / "acreal_9.ac0").read_bytes())
            got = api.list_traces(path, engine="waveview")
            self.assertEqual((got["analysis"], got["x"]), ("ac", "HERTZ"))
            self.assertEqual(got["analysis"], api.list_traces(path)["analysis"])

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "zeros.tr0"
            empty.write_bytes(b"")
            with mock.patch.object(waveview, "run_script") as run:
                with self.assertRaisesRegex(ValueError, "empty file"):
                    api.list_traces(empty, engine="waveview")
            run.assert_not_called()
            # WaveView opens a file of zero bytes too, and finds no signals in it.
            empty.write_bytes(b"\0" * 64)
            with self.assertRaisesRegex(WaveViewError, "finds no signals in the file"):
                api.list_traces(empty, engine="waveview")

    def test_two_signals_of_one_name(self):
        # extract.tcl addresses a signal by its name: refuse, as both columns would get one signal.
        with self.assertRaisesRegex(WaveViewError, "two signals named v\\(out\\).*engine='python'"):
            api.list_traces(WV / "dupnut.raw", engine="waveview")

    def test_post3_file(self):
        # In an HSPICE post=3 file each signal has its own x points, and WaveView gives no x name.
        header, listing = waveview._listing(str(WV / "post3.tr0"), 0, "magphase")
        self.assertEqual((header.x_name, listing.shared_x), ("TIME", False))
        for path in (TR, WV / "rc.tr0", WV / "rca.tr0", WV / "twoa.tr0"):    # binary 9601 and 2013, ASCII
            self.assertTrue(waveview._shared_x(str(path)), path.name)
        self.assertTrue(waveview._listing(str(NUTMEG), 0, "magphase")[1].shared_x)

    def test_header_read_sets_no_warnings_filter(self):
        # A warnings filter is not safe in threads: an overlap leaves "ignore" for the process.
        with mock.patch.object(waveview.warnings, "catch_warnings", side_effect=AssertionError("a filter")), \
                mock.patch.object(waveview.warnings, "simplefilter", side_effect=AssertionError("a filter")):
            self.assertEqual(api.list_traces(WV / "named3_9.tr0", engine="waveview")["traces"],
                             api.list_traces(WV / "named3_9.tr0")["traces"])

    def test_post_version_2013_sweep(self):
        with self.assertRaisesRegex(ValueError, "unsupported post_version '2013'"):
            read_header(WV / "rc.tr0")
        got = api.list_traces(WV / "rc.tr0", engine="waveview")
        self.assertEqual(got["traces"], ["v_0", "v_in", "v_out", "i_vin"])
        self.assertEqual((got["analysis"], got["x"]), ("tr", "TIME"))
        self.assertEqual((got["sweep_params"], got["sweep_count_hint"]), (["rval"], 3))

    def test_real_signal_in_an_ac_file(self):
        got = api.list_traces(WV / "rc.ac0", engine="waveview")
        self.assertEqual(got["traces"][-3:], ["i_vin_Mag", "i_vin_Phase", "vdb_out"])    # vdb(out) is real
        self.assertEqual((got["analysis"], got["x"], got["sweep_params"]), ("ac", "HERTZ", ["cval"]))

    def test_hierarchical_names_match_the_python_engine(self):
        got = api.list_traces(WV / "two.tr0", engine="waveview")
        self.assertEqual(got["traces"], api.list_traces(WV / "twoa.tr0")["traces"])
        self.assertEqual(got["traces"][3], "v_x1_mid")
        self.assertEqual((got["sweep_params"], got["sweep_count_hint"]), (["rval", "cval"], 3))
        deep = api.list_traces(WV / "deep.tr0", engine="waveview")["traces"]
        self.assertEqual(deep[4:8], ["v_xt_xm_m", "v_xt_xm2_m", "v_xt_xm_xl_n1", "v_xt_xm_xl2_n1"])

    def test_file_that_waveview_cannot_open(self):
        with self.assertRaises(WaveViewError) as caught:
            api.list_traces(WV / "bogus.tr0", engine="waveview")
        text = str(caught.exception)
        self.assertIn("WaveView (list_traces.tcl): cannot open", text)
        self.assertIn("[ERROR] unknown waveform format", text)
        self.assertIn("[ error ] sx_open_sim_file_read aborted", text)
        self.assertNotIn("running Tcl script", text)                # only the error lines

    def test_checks_before_waveview_starts(self):
        with mock.patch.object(waveview, "run_script") as run:
            with self.assertRaises(FileNotFoundError):
                api.list_traces(HERE / "missing.tr0", engine="waveview")
            with self.assertRaisesRegex(ValueError, "reads only plot 0"):
                api.list_traces(NUTMEG, plot=1, engine="waveview")
            with self.assertRaisesRegex(ValueError, "plot must be an integer"):
                api.list_traces(NUTMEG, plot=1.5, engine="waveview")
            with self.assertRaisesRegex(ValueError, "read it with read_measures"):
                api.list_traces(HERE / "test.mt0", engine="waveview")
            with self.assertRaisesRegex(ValueError, "ac_format"):
                api.list_traces(AC, ac_format="polar", engine="waveview")
        run.assert_not_called()


class TestReadTraces(FakeWaveView):
    def test_9601_values_are_exact(self):
        # 2605 points, 1170 of them repeat the previous x value: the scan keeps them all.
        ts = waveview.read_traces(TR)
        self.assert_same_traces(ts, read_traces(TR))
        self.assertEqual(ts.data["TIME"][0].size, 2605)
        self.assertEqual(ts.sweep_values, [[]])

    def test_sw_and_nutmeg_values_are_exact(self):
        self.assert_same_traces(waveview.read_traces(SW), read_traces(SW))
        self.assert_same_traces(waveview.read_traces(NUTMEG), read_traces(NUTMEG))

    def test_python_header_failure_does_not_stop_the_backup(self):
        # The file header only adds facts (float32 x, the Nutmeg x name); any error there is skipped.
        want_tr, want_nut = waveview.read_traces(TR), waveview.read_traces(NUTMEG)
        boom = mock.Mock(side_effect=IndexError("parser bug"))
        with mock.patch.object(waveview.reader, "read_header", boom), \
                mock.patch.object(waveview.nutmeg, "_parse_plot_header", boom):
            got_tr, got_nut = waveview.read_traces(TR), waveview.read_traces(NUTMEG)
        self.assertEqual(boom.call_count, 2)
        self.assert_same_traces(got_tr, want_tr)
        self.assertEqual(got_nut.selected[1:], want_nut.selected[1:])

    def test_ac_formats(self):
        # WaveView's real and imaginary parts are about 1 ulp of the magnitude off (absolute),
        # not exact; they round to the stored float32 values. The Python engine computes magnitude,
        # phase and dB of a 9601 file in float32 (dB: up to 7e-6 off near 0 dB).
        for ac_format, rtol, atol in (("realimag", 1e-13, 0.0), ("magphase", 5e-7, 0.0), ("db", 5e-7, 1e-5)):
            got = waveview.read_traces(AC, ac_format=ac_format)
            want = read_traces(AC, ac_format=ac_format)
            self.assertEqual(got.header.ac_format, ac_format)
            self.assert_same_traces(got, want, rtol, atol)
        got = waveview.read_traces(AC, ["v(vo"], ac_format="realimag").data["v_vo_Re"][0]
        np.testing.assert_array_equal(got.astype(np.float32), read_traces(AC, ["v(vo"], ac_format="realimag")
                                      .data["v_vo_Re"][0])

    def test_selection(self):
        ref = read_traces(TR, ["v_vo"])
        for names in (["v_vo"], ["v(vo"], ["v(vo)"], "v(vo"):
            self.assert_same_traces(waveview.read_traces(TR, names), ref)
        self.assertEqual(waveview.read_traces(TR, ["v_v*"]).selected, ["TIME", "v_vo", "v_vs"])
        self.assertEqual(waveview.read_traces(AC, ["v(vo"]).selected, ["HERTZ", "v_vo_Mag", "v_vo_Phase"])
        self.assertEqual(waveview.read_traces(AC, ["v_vo_Phase"]).selected, ["HERTZ", "v_vo_Phase"])

    def test_x_only(self):
        ts = waveview.read_traces(TR, ["TIME"])
        self.assertEqual(ts.selected, ["TIME"])
        np.testing.assert_array_equal(ts.data["TIME"][0], read_traces(TR, ["TIME"]).data["TIME"][0])

    def test_unknown_trace_fails_before_the_export(self):
        with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
            with self.assertRaisesRegex(ValueError, "unknown trace 'nope'; available traces: TIME, v_0"):
                waveview.read_traces(TR, ["nope"])
        self.assertEqual([c.args[0] for c in run.call_args_list], ["list_traces.tcl"])

    def test_point_count_mismatch(self):
        # A complete file: the points must be WaveView's count of them (2605 for test_9601.tr0).
        short = [(np.arange(3.0), {"v(vo)": np.arange(3.0)})]
        with mock.patch.object(waveview, "_parse_points", return_value=short):
            with self.assertRaisesRegex(WaveViewError, "reads 3 of the 2605 points of v\\(vo\\): WaveView stops "
                                                       "early on some values of about 1e34"):
                waveview.read_traces(TR, ["v(vo"])
        short = [(np.arange(3.0), {"v(vo)": np.arange(2.0)})]
        with mock.patch.object(waveview, "_parse_points", return_value=short), \
                mock.patch.object(waveview, "_check_counts"):
            with self.assertRaisesRegex(WaveViewError, "wrote 2 points for v\\(vo\\) and 3 x values in trace 0: "
                                                       "the signals have their own x points"):
                waveview.read_traces(TR, ["v(vo"])

    def test_scan_that_stops_early(self):
        # WaveView ends the scan of v(g36) (up to 1e36) after 100 of its 122 points, without an error.
        with self.assertRaisesRegex(WaveViewError, "reads 100 of the 122 points of v\\(g36\\)"):
            waveview.read_traces(WV / "big9.tr0", ["v(g36"])
        with self.assertRaisesRegex(WaveViewError, "reads 100 of the 122 points of v\\(g34\\)"):
            waveview.read_traces(WV / "big9.tr0", ["v(g33", "v(g34"])
        self.assert_same_traces(waveview.read_traces(WV / "big9.tr0", ["v(g33", "v(in"]),
                                read_traces(WV / "big9.tr0", ["v(g33", "v(in"]))
        # Also when WaveView says that the file is not complete: no trim of the other signals.
        run = waveview.run_script
        cut = lambda *a: run(*a) + "   *** waveform file 'big9.tr0' not complete ! ***\r\n"
        with mock.patch.object(waveview, "run_script", side_effect=cut), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for names in (["v(g36", "v(in"], ["v(in", "v(g36"]):
                with self.subTest(names), self.assertRaisesRegex(WaveViewError, "reads 100 of the 122 points of "
                                                                                "v\\(g36\\)"):
                    waveview.read_traces(WV / "big9.tr0", names)

    def test_truncated_sweep(self):
        # rc9.tr0 cut inside sweep 1: WaveView reads sweep 0 only, the header declares 3.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trunc_rc9.tr0"
            path.write_bytes((WV / "rc9.tr0").read_bytes()[:3019])
            self.assertEqual(api.list_traces(path, engine="waveview")["sweep_count_hint"], 3)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ts = waveview.read_traces(path, ["v(out"])
            self.assertTrue(ts.truncated)
            self.assertEqual(ts.sweep_values, [[1000.0]])
            self.assertTrue(any("declares 3 sweeps and WaveView reads 1" in str(w.message) for w in caught))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(ValueError, "sweep index 2 out of range; the file holds 1 sweep "
                                                        "\\(0 to 0\\) \\(the file is truncated\\)"):
                    waveview.read_traces(path, sweeps=[2])
                want = read_traces(path, ["v(out"])
            self.assertEqual((want.truncated, want.header.sweep_count_hint), (True, 3))
            np.testing.assert_array_equal(ts.data["v_out"][0], want.data["v_out"][0])

    def test_no_complete_sweep(self):
        # A swept file cut inside sweep 0: WaveView lists the parameter, and no sweep.
        for name, source, warning in (("cut_rc9.tr0", "rc9.tr0", "declares 3 sweeps and WaveView reads 0: the file is "
                                                                 "probably truncated, and WaveView drops its partial"),
                                      ("cut_rc.tr0", "rc.tr0", ": WaveView reads no complete sweep: the file")):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes((WV / source).read_bytes()[:1200])
                with mock.patch.object(waveview, "_run", wraps=waveview._run) as run, \
                        warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ts = waveview.read_traces(path, ["v(out"])
                self.assertEqual([c.args[0] for c in run.call_args_list], ["list_traces.tcl"])
                self.assertTrue(any(warning in str(w.message) for w in caught), [str(w.message) for w in caught])
                self.assertEqual((ts.sweep_indices, ts.sweep_values, ts.truncated), ([], [], True))
                self.assertEqual(ts.data, {"TIME": [], "v_out": []})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self.assertRaisesRegex(ValueError, "holds 0 sweeps \\(none\\) \\(the file is truncated\\)"):
                        waveview.read_traces(path, sweeps=[0])

    def cut(self, tmp, name, source, size):
        """A copy of source cut to size bytes (size < 1: that part of the file), as the canned files were made."""
        data = Path(source).read_bytes()
        path = Path(tmp) / name
        path.write_bytes(data[:size if size >= 1 else int(len(data) * size)])
        return path

    def test_truncation_from_the_console(self):
        # WaveView prints "waveform file ... not complete" for a cut file: also for a 2013 file
        # (no header for the Python reader), an unswept file and an ASCII file.
        with tempfile.TemporaryDirectory() as tmp:
            for name, source, size, sweeps, lost in (("cut_t9601.tr0", TR, 0.6, [0], "its last sweep is partial"),
                                                     ("cut_rca.tr0", WV / "rca.tr0", 0.97, [0, 1, 2],
                                                      "its last sweep is partial")):
                with self.subTest(name):
                    path = self.cut(tmp, name, source, size)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        ts = waveview.read_traces(path)
                        want = read_traces(path)
                    messages = [str(w.message) for w in caught]
                    self.assertTrue(any("WaveView reports that the file is not complete: the file is probably "
                                        f"truncated, and {lost}" in m for m in messages), messages)
                    self.assertEqual((ts.truncated, ts.sweep_indices), (True, sweeps))
                    self.assertEqual(want.sweep_indices, sweeps)
                    for n in ts.selected:       # the points that WaveView reads are the file's points
                        for got, ref in zip(ts.data[n], want.data[n]):
                            self.assertGreater(got.size, 0)
                            np.testing.assert_array_equal(got, ref[:got.size].astype(np.float64), err_msg=n)
            # The partial sweep of an ASCII file: WaveView keeps it, as the Python engine does.
            self.assertEqual([a.size for a in ts.data["v_out"]], [a.size for a in want.data["v_out"]])
            # In cut_t9601.tr0, i(vs) has no point of the partial last row (-npoints 1228, the
            # others 1229): every selection keeps 1228 points.
            path = self.cut(tmp, "cut_t9601.tr0", TR, 0.6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for names in (None, ["v(vo"], ["i(vs"]):
                    ts = waveview.read_traces(path, names)
                    self.assertEqual({a[0].size for a in ts.data.values()}, {1228}, names)

    def test_cut_inside_the_only_data_block(self):
        # WaveView reads no point of a binary file cut inside its first data block ("error
        # reading waveform file"). The Python engine reads its complete points.
        with tempfile.TemporaryDirectory() as tmp:
            path = self.cut(tmp, "cut_big9.tr0", WV / "big9.tr0", 0.95)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ts = waveview.read_traces(path, ["v(in"])
            self.assertEqual((ts.truncated, ts.sweep_indices, [a.size for a in ts.data["v_in"]]), (True, [0], [0]))
            self.assertTrue(any("not complete" in str(w.message) for w in caught))

    def test_truncated_nutmeg_dimensions(self):
        # dims.raw (Dimensions: 3,2: 3 sweeps of 2 points) cut after 3 of its 6 points: the
        # sweeps that the file reaches, the last one partial, as nutmeg.py reads them.
        with tempfile.TemporaryDirectory() as tmp:
            data = (WV / "dims.raw").read_bytes()
            path = self.cut(tmp, "cut_dims2.raw", WV / "dims.raw", data.index(b" 3\t"))
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ts = waveview.read_traces(path, ["v(out)"])
                want = nutmeg.read_nutmeg_traces(path, ["v(out)"])
            self.assertEqual((ts.truncated, ts.sweep_indices), (True, [0, 1]))
            self.assertEqual([a.tolist() for a in ts.data["v_out"]], [[0.0, 0.5], [1.0]])
            self.assertEqual([a.tolist() for a in want.data["v_out"]], [[0.0, 0.5], [1.0]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaisesRegex(ValueError, "sweep index 2 out of range"):
                    waveview.read_traces(path, sweeps=[2])
            # Cut inside its last line: that value can be cut short, so the point is dropped.
            path = self.cut(tmp, "cut_dims.raw", WV / "dims.raw", data.index(b"\n 3\t"))
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                ts = waveview.read_traces(path, ["v(out)"])
                want = nutmeg.read_nutmeg_traces(path, ["v(out)"])
            self.assertEqual([a.tolist() for a in ts.data["v_out"]], [[0.0, 0.5]])
            self.assertEqual([a.tolist() for a in want.data["v_out"]], [[0.0, 0.5]])
            # Cut inside point 3: after its index, inside v(out) (WaveView reads 4 points of v(out)
            # and 3 of i(v1)), and inside i(v1) (WaveView reads 4 points of each, the last one
            # short). Every selection keeps the 3 complete points, as nutmeg.py.
            p3 = data.index(b"\n 3\t") + 1
            x_end = p3 + data[p3:].index(b"\n") + 1
            vo_end = x_end + data[x_end:].index(b"\n")
            for name, size in (("cutn_index.raw", p3 + 2), ("cutn_row.raw", x_end + 7), ("cutn_value.raw", vo_end + 8)):
                path = self.cut(tmp, name, WV / "dims.raw", size)
                for names in (None, ["v(out)"], ["i(v1)"]):
                    with self.subTest(name, names=names), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.assert_same_traces(waveview.read_traces(path, names), nutmeg.read_nutmeg_traces(path, names))

    def test_file_that_grows_between_the_two_runs(self):
        # The simulator ends one more sweep after list_traces.tcl: keep the sweeps of the listing.
        parse = waveview._parse_points
        with mock.patch.object(waveview, "_parse_points", side_effect=lambda *a: (lambda t: t + t[-1:])(parse(*a))):
            ts = waveview.read_traces(WV / "rc.tr0", ["v(out"])
        self.assertEqual(ts.sweep_indices, [0, 1, 2])
        with mock.patch.object(waveview, "_parse_points", side_effect=lambda *a: parse(*a)[:2]):
            with self.assertRaisesRegex(WaveViewError, "wrote 2 traces; its listing has 3 \\(the file can have "
                                                       "changed"):
                waveview.read_traces(WV / "rc.tr0", ["v(out"])

    def test_signals_with_their_own_x_points(self):
        # post=3: extract.tcl checks the x points (SRP_XCHECK) and stops at the first other x.
        with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
            with self.assertRaisesRegex(WaveViewError, "the signals have their own x points .*post=3.*v\\(a\\) has "
                                                       "other x points than v\\(0\\)"):
                waveview.read_traces(WV / "post3.tr0", ["v_0", "v_a"])
        self.assertEqual([c.kwargs.get("xcheck") for c in run.call_args_list], [None, True])
        with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
            waveview.read_traces(TR, ["v(vo"])
        self.assertEqual(run.call_args_list[1].kwargs["xcheck"], False)

    def test_nutmeg_values_beyond_the_limit(self):
        # WaveView reads 1e35 as 1e30, and inf changes the whole plot (x becomes 0, 0, 0, -1e30).
        for name in ("nutbig.raw", "nutinf.raw"):
            with self.subTest(name):
                with self.assertRaisesRegex(WaveViewError, "above 1e\\+30 .* engine='python'"):
                    waveview.read_traces(WV / name)
                self.assertEqual(len(nutmeg.read_nutmeg_traces(WV / name).data["v_s"][0]), 4 if "inf" in name else 3)

    def test_name_with_a_latin1_byte(self):
        # WaveView gives the bytes of the file: the scripts write each name as valid UTF-8.
        ts = waveview.read_traces(WV / "latin1.raw")
        self.assertEqual(ts.selected, ["time", "v_\u00e9", "v_x"])
        np.testing.assert_array_equal(ts.data["v_\u00e9"][0], [0.0, 0.5, 1.0])
        self.assertEqual(waveview.read_traces(WV / "latin1.raw", ["v(\u00e9)"]).selected, ["time", "v_\u00e9"])

    def test_named_and_differential_probes(self):
        for name in ("named3_9.tr0", "expr.tr0", "diff2_9.tr0"):
            with self.subTest(name):
                self.assert_same_traces(waveview.read_traces(WV / name), read_traces(WV / name))
        ts = waveview.read_traces(WV / "named3_9.tr0", ["vdiff", "pw"])
        self.assertEqual(ts.selected, ["TIME", "pw", "vdiff"])
        ts = waveview.read_traces(WV / "diff2_9.tr0", ["v(x1.xl.n1,x1.xl.b)"])
        self.assert_same_traces(ts, read_traces(WV / "diff2_9.tr0", ["v(x1.xl.n1,x1.xl.b)"]))

    def test_noise_values(self):
        ts = waveview.read_traces(WV / "noi.ac0", ["outnoise", "innoise"])
        self.assertEqual(ts.selected, ["HERTZ", "outnoise", "innoise"])
        np.testing.assert_allclose(ts.data["outnoise"][0][:3], 1.6465e-17, rtol=1e-4)     # 4kTR of 1 kOhm, V^2/Hz

    def test_unknown_signal_in_the_script(self):
        with self.assertRaisesRegex(WaveViewError, "extract.tcl\\): unknown signal nope"):
            waveview._run("extract.tcl", str(TR), ["v(vo)", "nope"])

    def test_sweeps_of_a_2013_file(self):
        ts = waveview.read_traces(WV / "rc.tr0")
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0], [4000.0]])
        self.assertEqual(ts.sweep_indices, [0, 1, 2])
        self.assertEqual([a.size for a in ts.data["TIME"]], [86, 74, 66])
        # rca.tr0 is the same netlist written as ASCII (7 digits), which the Python reader reads.
        self.assert_same_traces(ts, read_traces(WV / "rca.tr0"), rtol=1e-6, atol=1e-12)

    def test_sweeps_filter(self):
        ts = waveview.read_traces(WV / "rc.tr0", ["v(out"], sweeps=[2, 0])
        self.assertEqual(ts.sweep_indices, [0, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [4000.0]])
        self.assertEqual(waveview.read_traces(WV / "rc.tr0", sweeps=1).sweep_indices, [1])
        with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
            with self.assertRaisesRegex(ValueError, "sweep index 3 out of range; the file holds 3 sweeps"):
                waveview.read_traces(WV / "rc.tr0", sweeps=[3])
            with self.assertRaisesRegex(ValueError, "sweep index 1 out of range; the file holds 1 sweep "):
                waveview.read_traces(TR, sweeps=[1])
            with self.assertRaisesRegex(ValueError, "sweeps must be a list"):
                waveview.read_traces(TR, sweeps=[True])
        self.assertNotIn("extract.tcl", [c.args[0] for c in run.call_args_list])

    def test_two_parameter_sweep_and_subcircuit(self):
        ts = waveview.read_traces(WV / "two.tr0", ["v(x1.mid)", "v(out)"])
        self.assertEqual(ts.selected, ["TIME", "v_out", "v_x1_mid"])
        self.assertEqual(ts.header.sweep_params, ["rval", "cval"])
        self.assertEqual(ts.sweep_values, [[1000.0, 1e-9], [2000.0, 1e-9], [2000.0, 2e-9]])
        # No Python reference: the Python reader takes the .data table name (pts) for a third
        # sweep parameter, so its values of twoa.tr0 are shifted (reported, not fixed here).
        for x, mid in zip(ts.data["TIME"], ts.data["v_x1_mid"]):
            self.assertEqual(x.size, 46)
            self.assertEqual(x[0], 0.0)
            self.assertTrue(np.all(np.diff(x) > 0))
            self.assertTrue(0.0 <= mid.min() and mid.max() < 1.0)
        self.assertGreater(ts.data["v_out"][0].max(), ts.data["v_out"][2].max())    # cval 1n charges faster

    def test_complex_and_real_signals_in_one_ac_file(self):
        ts = waveview.read_traces(WV / "rc.ac0", ["v(out)", "vdb(out)"], ac_format="db")
        self.assertEqual(ts.selected, ["HERTZ", "v_out_dB", "v_out_Phase", "vdb_out"])
        for i in range(2):         # HSPICE's own vdb(out) agrees to about 3e-7 dB
            np.testing.assert_allclose(ts.data["v_out_dB"][i], ts.data["vdb_out"][i], atol=1e-6)
        # No Python reference: HSPICE writes vdb(out) as one real value per point, and the
        # Python reader takes every AC variable for a complex pair, so its columns of
        # rca.ac0 shift (reported, not fixed here).
        self.assertEqual([a.size for a in ts.data["HERTZ"]], [51, 51])

    def test_xrange_bounds_of_a_9601_file(self):
        # The Python engine compares float32 x with the bounds in float32: 0.006 rounds up
        # to 0.006000000052, and a point at float32(0.006) is inside.
        for window in ((0.003, 0.006), (0.005, 0.01)):
            with self.subTest(window):
                got = waveview.read_traces(TR, ["v(vo"], xrange=window)
                want = read_traces(TR, ["v(vo"], xrange=window)
                self.assert_same_traces(got, want)
        self.assertEqual(waveview.read_traces(TR, ["v(vo"], xrange=(0.003, 0.006)).data["TIME"][0].size, 786)
        rc = waveview.read_traces(WV / "rc.tr0", ["v(out"], xrange=(5e-6, 6e-6))     # 2013: not rounded
        self.assertTrue(all(a.max() <= 6e-6 for a in rc.data["TIME"]))

    def test_parameter_name_with_a_colon(self):
        # .dc vin 0 1 0.5 vdd 1 2 0.5: WaveView names the outer parameter 0:vdd.
        ts = waveview.read_traces(WV / "dc2.sw0")
        self.assertEqual(waveview._listing(str(WV / "dc2.sw0"), 0, "magphase")[1].params, ["0:vdd"])
        self.assertEqual(ts.sweep_values, [[1.0], [1.5], [2.0]])
        want = read_traces(WV / "dc2.sw0")
        self.assertEqual((ts.header.sweep_params, ts.sweep_values), (want.header.sweep_params, want.sweep_values))
        self.assert_same_traces(ts, want)

    def test_power_and_par_probes(self):
        for names in (["p(rbus)"], ["p_rbus"], ["myexpr"], ["p(rbus)", "myexpr"]):
            with self.subTest(names):
                self.assert_same_traces(waveview.read_traces(WV / "odd.tr0", names), read_traces(WV / "odd.tr0", names))

    def test_nutmeg_dimensions(self):
        got = waveview.read_traces(WV / "dims.raw")
        want = read_traces(WV / "dims.raw")
        self.assertEqual(len(got.data["v_v-sweep"]), 3)
        self.assertEqual(got.sweep_values, [[], [], []])
        self.assert_same_traces(got, want)
        self.assert_same_traces(waveview.read_traces(WV / "dims.raw", sweeps=[1]), read_traces(WV / "dims.raw",
                                                                                                 sweeps=[1]))

    def test_non_ascii_path_in_an_error(self):
        # The path can reach WaveView as "?" on Windows, so the engine names the likely cause.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "caf\u00e9" / "bogus.tr0"
            path.parent.mkdir()
            path.write_bytes((WV / "bogus.tr0").read_bytes())
            with self.assertRaisesRegex(WaveViewError, "cannot open .*the path has non-ASCII characters"):
                waveview.read_header(path)

    def test_xrange(self):
        x = read_traces(TR).data["TIME"][0]
        lo, hi = float(x[100]), float(x[900])
        ts = waveview.read_traces(TR, ["v(vo"], xrange=(lo, hi))
        self.assert_same_traces(ts, read_traces(TR, ["v(vo"], xrange=(lo, hi)))
        self.assertEqual(ts.data["TIME"][0].size, 802)
        swept = waveview.read_traces(WV / "rc.tr0", xrange=(5e-6, 6e-6))
        for a in swept.data["TIME"]:
            self.assertTrue(a.size and a.min() >= 5e-6 and a.max() <= 6e-6)
        empty = waveview.read_traces(WV / "rc.tr0", xrange=(1.0, 2.0))
        self.assertEqual([a.size for a in empty.data["TIME"]], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "xrange min must not exceed max"):
            waveview.read_traces(TR, xrange=(2.0, 1.0))

    def test_nutmeg_ac(self):
        got = waveview.read_traces(WV / "multi_bin.raw")
        want = read_traces(WV / "multi_bin.raw")
        self.assertEqual(got.selected[1:], want.selected[1:])
        np.testing.assert_allclose(got.data["frequency"][0], want.data["frequency"][0], rtol=1e-15)
        for name in got.selected[1:]:
            np.testing.assert_allclose(got.data[name][0], want.data[name][0], rtol=1e-14, atol=1e-14)

    def test_lf_line_ends(self):
        self.fake("lf")
        self.assert_same_traces(waveview.read_traces(TR, ["v(vo"]), read_traces(TR, ["v(vo"]))
        self.assertEqual(waveview.read_header(WV / "two.tr0").sweep_params, ["rval", "cval"])

    def test_script_that_stops_on_a_tcl_error(self):
        self.fake("no_done")
        with self.assertRaises(WaveViewError) as caught:
            waveview.read_traces(TR)
        text = str(caught.exception)
        self.assertIn("no srp_done", text)
        self.assertIn('[Tcl Error] [line 12]:invalid command name "sx_bogus"', text)

    def test_no_output_quotes_the_console_tail(self):
        self.fake("silent")
        with self.assertRaisesRegex(WaveViewError, "(?s)no srp_done.*running Tcl script"):
            waveview.read_header(TR)

    def test_timeout(self):
        self.fake("sleep")
        os.environ["SRP_WAVEVIEW_TIMEOUT"] = "0.5"
        with self.assertRaisesRegex(WaveViewError, "no result after 0.5 s; set SRP_WAVEVIEW_TIMEOUT"):
            waveview.read_header(TR)

    def test_clean_up_error_does_not_hide_the_error(self):
        # On Windows a WaveView that outlives a timeout holds its files in the temporary folder.
        self.fake("sleep")
        os.environ["SRP_WAVEVIEW_TIMEOUT"] = "0.5"
        made, mkdtemp = [], tempfile.mkdtemp
        with mock.patch.object(tempfile, "mkdtemp", side_effect=lambda **k: made.append(mkdtemp(**k)) or made[-1]), \
                mock.patch("os.unlink", side_effect=PermissionError(13, "in use")), \
                mock.patch("os.rmdir", side_effect=PermissionError(13, "in use")):
            with self.assertRaisesRegex(WaveViewError, "no result after 0.5 s"):
                waveview.read_header(TR)
        for folder in made:
            shutil.rmtree(folder)

    @unittest.skipIf(os.name == "nt", "POSIX process groups")
    def test_timeout_stops_what_a_launcher_started(self):
        # wv can be a launcher script: the timeout stops also the WaveView it started.
        self.fake("launcher")
        os.environ["SRP_WAVEVIEW_TIMEOUT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SRP_FAKE_PID"] = str(Path(tmp) / "pid")
            with self.assertRaisesRegex(WaveViewError, "no result after 1 s"):
                waveview.read_header(TR)
            pid = int((Path(tmp) / "pid").read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, 9)
            self.fail("the child of the launcher is still running")

    @unittest.skipIf(os.name == "nt", "POSIX process groups")
    def test_stop_all(self):
        # WaveView runs in its own session: a caller that a signal stops calls stop_all first.
        self.fake("launcher")
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pid"
            os.environ["SRP_FAKE_PID"] = str(pid_file)
            errors = []

            def read():
                try:
                    waveview.read_header(TR)
                except WaveViewError as exc:
                    errors.append(str(exc))

            thread = threading.Thread(target=read)
            thread.start()
            for _ in range(100):
                if pid_file.exists() and pid_file.read_text():
                    break
                time.sleep(0.1)
            self.assertEqual(len(waveview._running), 1)
            waveview.stop_all()
            thread.join(30)
            self.assertFalse(thread.is_alive())
            pid = int(pid_file.read_text())
        self.assertEqual(waveview._running, set())
        self.assertEqual(len(errors), 1)
        self.assertIn("no srp_done", errors[0])
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, 9)
            self.fail("the child of the launcher is still running")

    @unittest.skipIf(os.name == "nt", "POSIX applies a/link/.. after the link")
    def test_path_through_a_symbolic_link(self):
        # results/latest/../rc.tr0 with latest -> data/run42 is data/rc.tr0 for the kernel and for
        # the Python reads of the header; os.path.abspath would give results/rc.tr0.
        with tempfile.TemporaryDirectory() as tmp:
            run42 = Path(tmp) / "data" / "run42"
            run42.mkdir(parents=True)
            (run42.parent / "rc.tr0").write_bytes((WV / "rc.tr0").read_bytes())
            (Path(tmp) / "results").mkdir()
            (Path(tmp) / "results" / "latest").symlink_to(run42)
            (Path(tmp) / "results" / "rc.tr0").write_bytes(b"not this file")
            path = Path(tmp) / "results" / "latest" / ".." / "rc.tr0"
            with mock.patch.object(waveview, "run_script", wraps=waveview.run_script) as run:
                ts = waveview.read_traces(path, ["v(out"])
            self.assertEqual({c.args[1]["SRP_IN"] for c in run.call_args_list}, {os.path.realpath(path)})
            self.assertEqual(ts.sweep_indices, [0, 1, 2])

    def test_non_ascii_temporary_folder_in_an_error(self):
        # Windows can give WaveView "?" for such a folder: then no script runs at all.
        self.fake("silent")
        with tempfile.TemporaryDirectory(prefix="\u5f20") as tmp:
            with mock.patch.object(tempfile, "tempdir", tmp):
                with self.assertRaisesRegex(WaveViewError, "temporary folder .* has non-ASCII characters"):
                    waveview.read_header(TR)

    def test_extract_error_record(self):
        with self.assertRaisesRegex(WaveViewError, "(?s)cannot open .*unknown waveform format"):
            waveview._run("extract.tcl", str(WV / "bogus.tr0"), [])

    def test_temporary_files_are_removed(self):
        # WaveView writes ace_tcl.log beside the script: the engine runs a copy in its
        # temporary directory, so nothing is written to the package directory.
        before = sorted(os.listdir(waveview.TCL_DIR))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(tempfile, "tempdir", tmp):
                with mock.patch.object(waveview, "run_script", wraps=waveview.run_script) as run:
                    waveview.read_traces(TR, ["v(vo"])
                    with self.assertRaises(WaveViewError):
                        waveview.read_header(WV / "bogus.tr0")
                self.assertTrue(all(c.args[2].startswith(tmp) for c in run.call_args_list))
            self.assertEqual(os.listdir(tmp), [])
        self.assertEqual(sorted(os.listdir(waveview.TCL_DIR)), before)
        self.assertNotIn("ace_tcl.log", before)


class TestReadMeasures(FakeWaveView):
    def test_same_values_as_python_without_alter(self):
        for path in (HERE / "test.mt0", WV / "rc.mt0", WV / "two.mt0", WV / "d1.mt0"):
            got = measure.read_measures(path, engine="waveview")
            want = measure.read_measures(path)
            self.assertEqual(got.names, [n for n in want.names if n != "alter#"], path.name)
            self.assertEqual((got.title, got.params, got.rows, got.failed, got.truncated),
                             (want.title, want.params, want.rows, want.failed, want.truncated), path.name)
            for name in got.names:      # WaveView reads 1.388e-06 as 1.3879999999999999e-06: 1 ulp
                np.testing.assert_allclose(got.values[name], want.values[name], rtol=1e-15, err_msg=name)

    def test_param_count_rule(self):
        # PARAM_COUNT counts the leading columns, index included.
        cases = {"rc.mt0": ["rval"],            # PARAM_COUNT=1: rval tdel ...
                 "two.mt0": ["rval", "cval"],   # PARAM_COUNT=3: index rval cval tdel ...
                 "d1.mt0": ["rval"],            # PARAM_COUNT=2: index rval vmax ...
                 "mc.mt0": []}                  # PARAM_COUNT=1: index vmax tr ... (Monte Carlo)
        for name, params in cases.items():
            self.assertEqual(measure.read_measures(WV / name, engine="waveview").params, params, name)
        two = measure.read_measures(WV / "two.mt0", ["never"], engine="waveview")
        self.assertEqual(two.names, ["index", "rval", "cval", "never"])
        self.assertEqual(two.failed, {"never": 3})
        self.assertTrue(np.isnan(two.values["never"]).all())           # 1.#QNAN in the scan output
        mc = measure.read_measures(WV / "mc.mt0", ["tr"], engine="waveview")
        self.assertEqual((mc.names, mc.title, mc.rows), (["index", "tr"], "* monte carlo", 3))

    def test_measure_with_the_name_of_a_parameter(self):
        # .data tab rval vmax with .meas vmax, index and temper: two columns of each name. The
        # second one gets the suffix #<column>, in both engines, so that no value is lost.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = measure.read_measures(WV / "mdup.mt0", engine="waveview")
            want = measure.read_measures(WV / "mdup.mt0")
        self.assertEqual(want.names, ["index", "rval", "vmax", "vmax#3", "a_very_long_measure_name_number_one",
                                      "a_very_long_measure_name_number_two", "temper", "index#7", "d$x", "temper#9",
                                      "alter#"])
        self.assertEqual(got.names, want.names[:-1])
        self.assertEqual(got.params, ["rval", "vmax"])
        self.assertEqual(sum("repeated measure names disambiguated with '#<column>': index, temper, vmax"
                             in str(w.message) for w in caught), 2)
        for name in got.values:
            np.testing.assert_allclose(got.values[name], want.values[name], rtol=1e-15, err_msg=name)
        for name, values in (("vmax", [1.0, 2.0]), ("vmax#3", [0.9969, 0.9461]), ("index", [1.0, 2.0]),
                             ("index#7", [0.6, 0.6]), ("temper", [0.5982, 0.5832]), ("temper#9", [25.0, 25.0])):
            np.testing.assert_allclose(got.values[name], values, rtol=1e-15, err_msg=name)
        with self.assertWarnsRegex(RuntimeWarning, "repeated measure names"):
            ms = measure.read_measures(WV / "mdup.mt0", ["vmax#*"])
        self.assertEqual(ms.names, ["index", "rval", "vmax", "vmax#3"])

    def test_truncated_measure_file(self):
        # rc.mt0 cut inside row 3, and cut inside its last listed value (25.0000 -> 2): WaveView
        # reads the values of that row; the engine counts the values of the file text.
        data = (WV / "rc.mt0").read_bytes()
        row3 = data.index(b"  4.000e+03")
        with tempfile.TemporaryDirectory() as tmp:
            for name, size in (("cut_rc.mt0", data.index(b"2.781e-06", row3) + 9),
                               ("cut2_rc.mt0", data.index(b"25.0000", row3) + 1)):
                with self.subTest(name):
                    path = Path(tmp) / name
                    path.write_bytes(data[:size])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        got = measure.read_measures(path, engine="waveview")
                        want = measure.read_measures(path)
                    self.assertEqual((got.rows, got.truncated, got.failed), (2, True, {"never": 2}))
                    self.assertEqual((want.rows, want.truncated, want.failed), (2, True, {"never": 2}))
                    self.assertTrue(all(v.size == 2 for v in got.values.values()))
                    np.testing.assert_array_equal(got.values["temper"], [25.0, 25.0])
                    self.assertEqual(sum("file ends with an incomplete row; keeping 2 complete rows" in str(w.message)
                                         for w in caught), 2)

    def test_scan_that_stops_early(self):
        # rc.mt0 with vmax = 1e34 in row 1: WaveView reads 1 of its 3 values, without an error.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big34.mt0"
            path.write_bytes((WV / "rc.mt0").read_bytes().replace(b"0.9940", b"1.0e+34"))
            with self.assertRaisesRegex(WaveViewError, "reads 1 of the 3 values of vmax"):
                measure.read_measures(path, engine="waveview")
            self.assertEqual(measure.read_measures(path).values["vmax"].tolist(), [1e34, 0.9258, 0.7733])

    def test_long_title(self):
        # WaveView cuts a long title: the title comes from the .TITLE line of the file.
        data = (WV / "rc.mt0").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "title.mt0"
            path.write_bytes(data.replace(b"* rc sweep test", b"* " + b"a long title that goes on " * 5 + b"to the end"))
            got = measure.read_measures(path, engine="waveview")
            self.assertEqual(got.title, measure.read_measures(path).title)
            self.assertTrue(got.title.endswith("goes on to the end"), got.title)

    def test_srp_signals_of_the_caller_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "signals.txt").write_text("vmax\n")
            os.environ["SRP_SIGNALS"] = str(Path(tmp, "signals.txt"))
            ms = measure.read_measures(WV / "rc.mt0", engine="waveview")
        self.assertEqual(ms.names, ["rval", "tdel", "vmax", "never", "temper"])
        self.assertEqual(ms.params, ["rval"])

    def test_selection(self):
        ms = measure.read_measures(WV / "rc.mt0", ["vmax"], engine="waveview")
        self.assertEqual(ms.names, ["rval", "vmax"])
        self.assertEqual(ms.rows, 3)
        with self.assertRaisesRegex(ValueError, "unknown measure 'alter#'; available measures: rval, tdel"):
            measure.read_measures(WV / "rc.mt0", ["alter#"], engine="waveview")

    def test_not_a_measure_file(self):
        with mock.patch.object(waveview, "run_script") as run:
            with self.assertRaisesRegex(ValueError, "not an HSPICE measure file"):
                measure.read_measures(TR, engine="waveview")
            with self.assertRaises(FileNotFoundError):
                measure.read_measures(HERE / "missing.mt0", engine="waveview")
        run.assert_not_called()

    def test_lf_line_ends(self):
        self.fake("lf")
        self.assertEqual(measure.read_measures(WV / "two.mt0", engine="waveview").failed, {"never": 3})


class TestMcpOutputs(FakeWaveView):
    def setUp(self):
        super().setUp()
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from spice_result_parser import mcp_server

        self.m = mcp_server
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def tools(self):
        return {t.name: t for t in asyncio.run(self.m.server.list_tools())}

    def test_signal_stops_waveview_first(self):
        with mock.patch.object(waveview, "stop_all") as stop, mock.patch.object(self.m.os, "_exit") as end:
            self.m._stop(signal.SIGTERM, None)
        stop.assert_called_once_with()
        end.assert_called_once_with(128 + signal.SIGTERM)
        with mock.patch.object(self.m.server, "run"), mock.patch.object(self.m.signal, "signal") as handler:
            self.m.main()
        self.assertIn(mock.call(signal.SIGTERM, self.m._stop), handler.call_args_list)

    def test_every_tool_takes_engine(self):
        for name, tool in self.tools().items():
            schema = getattr(tool, "inputSchema", None) or tool.input_schema
            spec = schema["properties"]["engine"]
            self.assertIn("SRP_ENGINE", spec["description"], name)
            self.assertIn({"enum": ["python", "waveview"], "type": "string"}, spec["anyOf"], name)
            self.assertIsNone(spec["default"], name)

    def test_list_traces_names_the_engine(self):
        out = self.m.list_traces(str(TR), engine="waveview")
        self.assertEqual((out["engine"], out["format"]), ("waveview", "waveview"))
        self.assertEqual(self.m.list_traces(str(TR))["engine"], "python")
        os.environ["SRP_ENGINE"] = "waveview"
        self.assertEqual(self.m.list_traces(str(TR))["engine"], "waveview")
        json.dumps(out)

    def test_text_differs_only_by_the_engine_note(self):
        want = self.m.extract(str(TR), ["v(vo"])
        got = self.m.extract(str(TR), ["v(vo"], engine="waveview")
        self.assertNotIn("engine", want)
        head, _, rest = got.partition("\n")
        self.assertTrue(head.endswith(" | engine: waveview"), head)
        self.assertEqual(head[:-len(" | engine: waveview")] + "\n" + rest, want)

    def test_text_of_a_2013_sweep(self):
        text = self.m.extract(str(WV / "two.tr0"), ["v(x1.mid)"], engine="waveview")
        self.assertTrue(text.startswith("transient | x: TIME | 138 points in 3 sweeps"), text)
        self.assertIn("-- sweep 2: rval=2000, cval=2e-09 --", text)

    def test_summary(self):
        want = self.m.extract(str(TR), ["v(vo"], output="summary", downsample=10)
        got = self.m.extract(str(TR), ["v(vo"], output="summary", downsample=10, engine="waveview")
        self.assertEqual(got.pop("format"), "waveview")
        want.pop("format")
        self.assertEqual(got, want)
        json.dumps(got)

    def test_csv_is_byte_identical(self):
        want = self.m.extract(str(TR), output="csv", dest=str(self.dir / "py.csv"))
        got = self.m.extract(str(TR), output="csv", dest=str(self.dir / "wv.csv"), engine="waveview")
        self.assertEqual((want["engine"], got["engine"]), ("python", "waveview"))
        self.assertEqual((self.dir / "wv.csv").read_bytes(), (self.dir / "py.csv").read_bytes())
        self.assertEqual({k: v for k, v in got.items() if k not in ("path", "engine")},
                         {k: v for k, v in want.items() if k not in ("path", "engine")})

    def test_csv_of_a_sweep(self):
        out = self.m.extract(str(WV / "rc.tr0"), ["v(out"], output="csv", dest=str(self.dir / "rc.csv"),
                             sweeps=[1, 2], engine="waveview")
        self.assertEqual((out["sweeps"], out["points"], out["sweep_indices"]), (2, 140, [1, 2]))
        lines = (self.dir / "rc.csv").read_text().splitlines()
        self.assertEqual(lines[0], "sweep,rval,TIME,v_out")
        self.assertTrue(lines[1].startswith("1,2000,0,"))

    def test_npz(self):
        self.m.extract(str(WV / "rc.tr0"), ["v(out"], output="npz", dest=str(self.dir / "rc.npz"),
                       engine="waveview")
        with np.load(self.dir / "rc.npz") as z:
            self.assertEqual(sorted(z.files), ["TIME@0", "TIME@1", "TIME@2", "__sweep_params__",
                                               "__sweep_values__", "v_out@0", "v_out@1", "v_out@2"])
            np.testing.assert_array_equal(z["__sweep_values__"], [[1000.0], [2000.0], [4000.0]])
            self.assertEqual(z["v_out@2"].size, 66)

    def test_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        out = self.m.extract(str(WV / "rc.ac0"), ["v(out)"], output="png", dest=str(self.dir / "ac.png"),
                             engine="waveview")
        self.assertEqual((out["output"], out["engine"], out["sweeps"]), ("png", "waveview", 2))
        self.assertTrue(all(v > 0 for v in out["size"]))
        self.assertEqual((self.dir / "ac.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_read_measures_tool(self):
        got = self.m.read_measures(str(WV / "two.mt0"), engine="waveview")
        want = self.m.read_measures(str(WV / "two.mt0"))
        self.assertTrue(got.startswith("measures | 3 rows | params: rval, cval | engine: waveview\n"))
        self.assertNotIn("engine", want)
        self.assertIn("failed: never in 3 of 3 rows", got)
        self.assertNotIn("alter#", got)
        self.assertIn("alter#", want)

    def test_waveview_errors_reach_the_client(self):
        with self.assertRaisesRegex(self.m.ToolError, "(?s)WaveViewError: WaveView \\(list_traces.tcl\\): "
                                                      "cannot open .*unknown waveform format"):
            self.m.list_traces(str(WV / "bogus.tr0"), engine="waveview")
        with self.assertRaisesRegex(self.m.ToolError, "engine must be one of"):
            self.m.extract(str(TR), engine="spice")
        result = asyncio.run(self.m.server.call_tool("extract", {"path": str(WV / "rc.tr0"), "names": "v(out",
                                                                  "engine": "waveview"}))
        content = getattr(result, "content", None) or (result[0] if isinstance(result, tuple) else result)
        self.assertIn("| engine: waveview", content[0].text)


class TestScriptsAndDocs(unittest.TestCase):
    SCRIPTS = ("list_traces.tcl", "extract.tcl", "read_measures.tcl")

    def test_scripts_ship_in_the_package(self):
        package = Path(waveview.__file__).parent
        self.assertEqual(Path(waveview.TCL_DIR), package / "tcl")
        for name in self.SCRIPTS:
            self.assertTrue((package / "tcl" / name).is_file(), name)

    def test_scripts_are_tcl_8_4(self):
        newer = (r"\bdict\b|\blassign\b|\{\*\}|\blmap\b|\bapply\b|string cat|\b(in|ni)\b\s*\{"
                 r"|\{[^{}\n]*\s(in|ni)\s[^{}\n]*\}|\b(lreverse|lrepeat|try|throw|chan)\b")
        for name in self.SCRIPTS:
            text = (Path(waveview.TCL_DIR) / name).read_text()
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            self.assertNotRegex(code, newer, name)
            self.assertIn("$env(SRP_IN)", code, name)              # paths come from the environment
            self.assertIn("set arg [list $path]", code, name)                 # one argument, also with spaces
            self.assertIn("sx_open_sim_file_read $arg", code, name)
            self.assertIn("srp_done", code, name)
            self.assertTrue(code.rstrip().endswith(("exit", "finish")), name)
            self.assertIn("wv -ace_no_gui " + name, text, name)       # how to run it by hand

    def test_docs_describe_the_switch(self):
        for doc in ("README.md", "Usage.md"):
            text = (HERE.parent / doc).read_text()
            for token in ("SRP_ENGINE", "SRP_WAVEVIEW", "waveview"):
                self.assertIn(token, text, doc)
        self.assertIn("SRP_WAVEVIEW_TIMEOUT", (HERE.parent / "Usage.md").read_text())


if __name__ == "__main__":
    unittest.main()
