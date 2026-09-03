"""
Regression tests: the repackaged pipeline must reproduce the original CLI scripts.

This is the guard rail for the whole exercise - the tool was a repackaging, not a
re-derivation, so any drift from ``test_analysis.py`` is a bug. Two levels are
checked:

  1. ``first_contact_auto`` returns bit-identical values (all 8 outputs, both
     the reflection and transmission branches).
  2. The assembled ``_threshold.csv`` table is bit-identical when using
     ``StrainCalibration.from_hardware()``, which applies the voltage->strain
     arithmetic in the scripts' exact operation order.

     With a single ``multiplier`` instead, results differ by ~1e-19 absolute
     (~1e-15 relative) purely from float64 non-associativity. That is asserted
     as a bounded difference, not equality.

Run:  python tests/test_parity.py       (or: pytest tests/test_parity.py)

Skips cleanly when the original scripts or the PicoScope sample data are absent,
so the suite is still usable by anyone who clones this tool on its own.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
REPO = HERE.parents[2]                       # .../StrainVisor
ORIG_SCRIPT = REPO / "src/split-hopkinson_bar/Oscilloscope_Analysis/test_analysis.py"
DATA_DIR = REPO / "src/split-hopkinson_bar/Oscilloscope_Analysis/picoscope_csv"

SAMPLES = [
    "1d9bar_confined_Pastic_FineSand-0002.csv",
    "1d1bar.csv",
    "1d9bar_confined_Alu_Coarse.csv",
]

# Hardware constants as hardcoded in the CLI scripts.
V_EX, GAUGE_FACTOR, GAIN_C, GAIN_D = 10.0, 2.04, 100.0, 100.0

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(BACKEND))

import pipeline  # noqa: E402
import pulse_detection  # noqa: E402
import scope_io  # noqa: E402


def _load_original():
    spec = importlib.util.spec_from_file_location("orig_test_analysis", ORIG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["orig_test_analysis"] = module
    spec.loader.exec_module(module)
    return module


def _available() -> list[Path]:
    if not ORIG_SCRIPT.is_file() or not DATA_DIR.is_dir():
        return []
    return [DATA_DIR / s for s in SAMPLES if (DATA_DIR / s).is_file()]


def _read_naive(path: Path) -> pd.DataFrame:
    """The scripts' own read: two-row header, flattened."""
    data = pd.read_csv(path, header=[0, 1])
    data.columns = [f"{c[0]} {c[1]}" if c[1] != "" else c[0] for c in data.columns]
    return data


def _original_threshold_frame(orig, path: Path) -> pd.DataFrame:
    """test_analysis.process_single_file's assembly, inlined verbatim."""
    data = _read_naive(path)
    col_c = "Channel C (mV)" if "Channel C (mV)" in data.columns else "Channel C (V)"
    scale_c = 1000.0 if "(mV)" in col_c else 1.0
    col_d = "Channel D (mV)" if "Channel D (mV)" in data.columns else "Channel D (V)"
    scale_d = 1000.0 if "(mV)" in col_d else 1.0

    (ref_idx_L, ref_idx_R, ref_t_L, ref_t_R, reflect_mu, _rp, _ysc, _lc) = orig.first_contact_auto(
        data["Time (ms)"], data[col_c], 5, "prominences", "reflection"
    )
    (trans_idx_L, trans_idx_R, trans_t_L, trans_t_R, trans_mu, trans_peaks, _ysd, _ld) = orig.first_contact_auto(
        data["Time (ms)"], data[col_d], 3, "prominences", "transmission", sg_win=51
    )

    if len(ref_t_L) > 0 and len(trans_t_L) > 0:
        t_incident = ref_t_L[0]
        fL, fR, ftL, ftR, fp = [], [], [], [], []
        for i in range(len(trans_t_L)):
            if trans_t_L[i] <= (t_incident + 0.25):
                fL.append(trans_idx_L[i]); fR.append(trans_idx_R[i])
                ftL.append(trans_t_L[i]); ftR.append(trans_t_R[i])
                if i < len(trans_peaks):
                    fp.append(trans_peaks[i])
        trans_idx_L, trans_idx_R = fL, fR

    ref_idx_L, ref_idx_R = ref_idx_L[:2], ref_idx_R[:2]
    trans_idx_L, trans_idx_R = trans_idx_L[:1], trans_idx_R[:1]

    columns, max_len, shared = {}, 0, None
    for i, (s, e) in enumerate(zip(ref_idx_L, ref_idx_R)):
        tp = data.loc[s:e, "Time (ms)"]
        tn = tp - tp.iloc[0]
        raw_v = abs(data.loc[s:e, col_c] - reflect_mu) / scale_c
        strain = ((raw_v / GAIN_C) * (2 / GAUGE_FACTOR)) / V_EX
        label = "Incident Pulse" if i == 0 else f"Reflected Pulse {i}"
        columns[f"{label} Time (ms)"] = tp.values
        columns[f"{label} Strain"] = strain.values
        if len(tn) > max_len:
            max_len, shared = len(tn), tn.values
    for i, (s, e) in enumerate(zip(trans_idx_L, trans_idx_R)):
        tp = data.loc[s:e, "Time (ms)"]
        tn = tp - tp.iloc[0]
        raw_v = abs(data.loc[s:e, col_d] - trans_mu) / scale_d
        strain = ((raw_v / GAIN_D) * (2 / GAUGE_FACTOR)) / V_EX
        columns[f"Transmission {i + 1} Time (ms)"] = tp.values
        columns[f"Transmission {i + 1} Strain"] = strain.values
        if len(tn) > max_len:
            max_len, shared = len(tn), tn.values

    frame = pd.DataFrame({"Shared Time (ms)": shared})
    for name, values in columns.items():
        frame[name] = pd.Series(values).reindex(range(max_len))
    return frame


def _new_result(path: Path):
    info = scope_io.describe(str(path), name=path.name)
    col_c = scope_io.pick_channel(info["columns"], "C")
    col_d = scope_io.pick_channel(info["columns"], "D")
    scope = scope_io.load(str(path), name=path.name, signal_cols=[col_c, col_d])
    return pipeline.analyse(scope, col_c, col_d)


def test_detection_bit_identical() -> None:
    paths = _available()
    if not paths:
        print("SKIP test_detection_bit_identical (reference script or data missing)")
        return
    orig = _load_original()
    names = ["index_lefts", "index_rights", "t_lefts", "t_rights", "mu",
             "peak_indices", "y_s", "low_thr"]

    for path in paths:
        data = _read_naive(path)
        col_c = scope_io.pick_channel(list(data.columns), "C")
        col_d = scope_io.pick_channel(list(data.columns), "D")
        for mode, col, pulse_num, kwargs in (
            ("reflection", col_c, 5, {}),
            ("transmission", col_d, 3, {"sg_win": 51}),
        ):
            expected = orig.first_contact_auto(
                data["Time (ms)"], data[col], pulse_num, "prominences", mode, **kwargs
            )
            actual = pulse_detection.first_contact_auto(
                data["Time (ms)"], data[col], pulse_num, mode, **kwargs
            )
            for name, want, got in zip(names, expected, actual):
                want_a = np.asarray(want, dtype=float)
                got_a = np.asarray(got, dtype=float)
                assert want_a.shape == got_a.shape, f"{path.name}/{mode}/{name}: shape"
                assert np.array_equal(want_a, got_a, equal_nan=True), \
                    f"{path.name}/{mode}/{name}: values differ"
        print(f"  OK detection bit-identical: {path.name}")


def test_threshold_frame_bit_identical_hardware_mode() -> None:
    paths = _available()
    if not paths:
        print("SKIP test_threshold_frame_bit_identical_hardware_mode (data missing)")
        return
    orig = _load_original()
    calibration = pipeline.StrainCalibration.from_hardware(V_EX, GAUGE_FACTOR, GAIN_C)

    for path in paths:
        expected = _original_threshold_frame(orig, path)
        actual = pipeline.to_threshold_frame(_new_result(path), calibration)

        assert list(expected.columns) == list(actual.columns), \
            f"{path.name}: column mismatch\n{list(expected.columns)}\n{list(actual.columns)}"
        assert expected.shape == actual.shape, f"{path.name}: shape {expected.shape} vs {actual.shape}"
        for col in expected.columns:
            want = expected[col].to_numpy(dtype=float)
            got = actual[col].to_numpy(dtype=float)
            assert np.array_equal(want, got, equal_nan=True), f"{path.name}/{col}: values differ"
        print(f"  OK threshold CSV bit-identical: {path.name} {actual.shape}")


def test_multiplier_mode_within_float_tolerance() -> None:
    paths = _available()
    if not paths:
        print("SKIP test_multiplier_mode_within_float_tolerance (data missing)")
        return
    orig = _load_original()
    for path in paths:
        expected = _original_threshold_frame(orig, path)
        actual = pipeline.to_threshold_frame(_new_result(path), pipeline.DEFAULT_STRAIN_MULTIPLIER)
        for col in expected.columns:
            if "Strain" not in col:
                continue
            want = expected[col].to_numpy(dtype=float)
            got = actual[col].to_numpy(dtype=float)
            finite = np.isfinite(want) & np.isfinite(got)
            if not finite.any():
                continue
            rel = np.abs(got[finite] - want[finite]) / np.maximum(np.abs(want[finite]), 1e-30)
            assert rel.max() < 1e-12, f"{path.name}/{col}: relative drift {rel.max():.3e} too large"
        print(f"  OK multiplier mode within 1e-12 relative: {path.name}")


def test_pruned_read_matches_naive_read() -> None:
    paths = _available()
    if not paths:
        print("SKIP test_pruned_read_matches_naive_read (data missing)")
        return
    for path in paths:
        naive = _read_naive(path)
        col_c = scope_io.pick_channel(list(naive.columns), "C")
        col_d = scope_io.pick_channel(list(naive.columns), "D")
        pruned = scope_io.load(str(path), name=path.name, signal_cols=[col_c, col_d])
        for col in ("Time (ms)", col_c, col_d):
            want = naive[col].to_numpy(dtype=float)
            got = pruned.frame[col].to_numpy(dtype=float)
            assert want.shape == got.shape, f"{path.name}/{col}: shape"
            assert np.array_equal(want, got), f"{path.name}/{col}: values differ"
        print(f"  OK pruned read == naive read: {path.name}")


def test_generic_csv_roundtrip() -> None:
    """Generic single-header CSVs work without any PicoScope conventions."""
    import io as _io

    raw = b"t_s,inc_v,trans_v\n0.0,0.1,0.02\n1e-6,0.2,0.03\n2e-6,0.15,0.01\n"
    info = scope_io.describe(_io.BytesIO(raw), name="generic.csv")
    assert info["layout"] == "generic", info
    assert info["columns"] == ["t_s", "inc_v", "trans_v"], info

    scope = scope_io.load(
        _io.BytesIO(raw), name="generic.csv", time_col="t_s",
        signal_cols=["inc_v", "trans_v"], time_unit="s",
    )
    assert scope.n_samples == 3
    assert scope.time_unit == "s"
    assert list(scope.frame.columns) == ["t_s", "inc_v", "trans_v"]
    print("  OK generic CSV round trip")


def test_calibration_equivalence() -> None:
    raw = np.array([0.0, 1.0, 2.5, 10.0])
    hw = pipeline.StrainCalibration.from_hardware()
    mult = pipeline.StrainCalibration(multiplier=pipeline.DEFAULT_STRAIN_MULTIPLIER)
    assert np.allclose(hw.apply(raw), mult.apply(raw), rtol=1e-12, atol=0)
    assert hw.effective_multiplier == mult.effective_multiplier
    try:
        pipeline.StrainCalibration(v_ex=10.0)  # incomplete hardware set
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for incomplete calibration")
    print("  OK calibration modes agree and validate")


def test_missing_pulse_columns_present_but_empty() -> None:
    """batch_analysis.py injected empty columns; the schema must stay stable."""
    result = pipeline.AnalysisResult(name="synthetic.csv", time_unit="ms", status="ok")
    result.windows.append(
        pipeline.PulseWindow(
            label="Incident Pulse", kind=pipeline.INCIDENT, channel="Channel C (V)",
            idx_start=0, idx_end=2,
            time_abs=np.array([0.0, 1.0, 2.0]), raw_v=np.array([0.0, 1.0, 0.5]),
        )
    )
    frame = pipeline.to_threshold_frame(result)
    for col in (
        "Shared Time (ms)", "Incident Pulse Time (ms)", "Incident Pulse Strain",
        "Reflected Pulse 1 Time (ms)", "Reflected Pulse 1 Strain",
        "Transmission 1 Time (ms)", "Transmission 1 Strain",
    ):
        assert col in frame.columns, f"missing {col}"
    assert frame["Reflected Pulse 1 Strain"].isna().all()
    print("  OK missing pulses yield present-but-empty columns")


def test_duplicate_reflection_flagged_but_csv_unchanged() -> None:
    """Bar-on-bar traces duplicate the incident window; flag it, don't drop it.

    With no specimen between the bars there is no impedance mismatch and so no
    reflected pulse, but find_peaks still returns a second Channel C peak inside
    the same excursion. The duplicate must be marked for display purposes while
    the emitted CSV stays exactly as the CLI scripts produced it.
    """
    paths = [p for p in _available() if p.name == "1d1bar.csv"]
    if not paths:
        print("SKIP test_duplicate_reflection_flagged_but_csv_unchanged (1d1bar.csv missing)")
        return

    result = _new_result(paths[0])
    incident = result.window(pipeline.INCIDENT)
    reflected = next((w for w in result.windows if w.kind == pipeline.REFLECTED), None)
    assert incident is not None and reflected is not None, "expected both windows"
    assert (reflected.idx_start, reflected.idx_end) == (incident.idx_start, incident.idx_end),         "1d1bar.csv is expected to duplicate the incident window"
    assert reflected.duplicate_of == incident.label, f"not flagged: {reflected.duplicate_of}"
    assert result.has_reflection is True
    assert result.has_distinct_reflection is False
    assert any("bar-on-bar" in m for m in result.messages), result.messages

    # The CSV must still carry the duplicated columns, unchanged.
    frame = pipeline.to_threshold_frame(result, pipeline.StrainCalibration.from_hardware())
    inc = frame["Incident Pulse Strain"].to_numpy(dtype=float)
    ref = frame["Reflected Pulse 1 Strain"].to_numpy(dtype=float)
    assert np.array_equal(inc, ref, equal_nan=True), "duplicate columns must remain in the CSV"
    print("  OK duplicate reflection flagged, CSV columns preserved")


def test_distinct_reflection_not_flagged() -> None:
    """A test with material between the bars must not be marked as duplicated."""
    paths = [p for p in _available() if p.name != "1d1bar.csv"]
    if not paths:
        print("SKIP test_distinct_reflection_not_flagged (data missing)")
        return
    for path in paths:
        result = _new_result(path)
        if not result.has_reflection:
            continue
        assert result.has_distinct_reflection, f"{path.name}: unexpectedly flagged as duplicate"
        for w in result.windows:
            assert w.duplicate_of is None, f"{path.name}: {w.label} wrongly flagged"
        print(f"  OK distinct reflection kept: {path.name}")


TESTS = [
    test_generic_csv_roundtrip,
    test_calibration_equivalence,
    test_missing_pulse_columns_present_but_empty,
    test_pruned_read_matches_naive_read,
    test_detection_bit_identical,
    test_threshold_frame_bit_identical_hardware_mode,
    test_multiplier_mode_within_float_tolerance,
    test_duplicate_reflection_flagged_but_csv_unchanged,
    test_distinct_reflection_not_flagged,
]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        print(f"\n{test.__name__}")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
