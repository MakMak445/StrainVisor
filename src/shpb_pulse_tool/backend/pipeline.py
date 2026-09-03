"""
End-to-end SHPB pulse pipeline: detect -> gate -> truncate -> convert -> assemble.

This is the orchestration that ``test_analysis.process_single_file`` and
``batch_analysis.process_and_overlay`` performed inline, lifted out of the CLI
so it can be driven from an API or a UI. The sequence and every constant match
those scripts:

  1. Channel C -> ``first_contact_auto(..., pulse_num=5, "reflection")``
     Channel D -> ``first_contact_auto(..., pulse_num=3, "transmission", sg_win=51)``
  2. max-transmission-delay gate: drop transmitted pulses starting later than
     ``t_incident + max_trans_delay`` (0.25 ms in the scripts).
  3. truncate to the first 2 Channel C pulses (incident + 1st reflection) and
     the first 1 Channel D pulse (1st transmission).
  4. per pulse: ``raw_v = abs(raw - mu) / scale`` then ``strain = raw_v * k``.
  5. assemble the padded CSV with the existing ``_threshold.csv`` schema.

The one deliberate structural change: **strain conversion is separated from
detection.** Detection is the expensive part (~0.4 s/file, dominated by the CSV
read) and does not depend on the voltage->strain multiplier, so the multiplier
can be changed and plots re-rendered without re-detecting anything.

Strain multiplier
-----------------
The scripts computed ``strain = ((raw_v / GAIN) * (2 / GAUGE_FACTOR)) / V_EX``
with V_EX=10.0, GAUGE_FACTOR=2.04, GAIN_C=GAIN_D=100.0. That is algebraically
``raw_v * 2 / (GAIN * GAUGE_FACTOR * V_EX)``, i.e. a single multiplier of
``0.000980392156862745`` - the value exposed as ``DEFAULT_STRAIN_MULTIPLIER``
and overridable per channel by the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulse_detection import first_contact_auto
from scope_io import ScopeFile

# Hardware constants from the CLI scripts, kept only to derive the default.
V_EX = 10.0
GAUGE_FACTOR = 2.04
GAIN = 100.0
DEFAULT_STRAIN_MULTIPLIER = 2.0 / (GAIN * GAUGE_FACTOR * V_EX)  # 0.000980392156862745

# Gate from the scripts: a transmitted pulse must start within this much time
# (in the trace's own time unit, ms for PicoScope exports) of the incident one.
DEFAULT_MAX_TRANS_DELAY = 0.25

REFL_PULSE_NUM = 5   # Channel C peaks requested before truncation
TRANS_PULSE_NUM = 3  # Channel D peaks requested before truncation
MAX_REFL_KEPT = 2    # incident + first reflection
MAX_TRANS_KEPT = 1   # first transmission

INCIDENT = "incident"
REFLECTED = "reflected"
TRANSMITTED = "transmitted"


@dataclass
class StrainCalibration:
    """Voltage -> strain conversion.

    Two equivalent-but-not-bit-identical forms:

      * ``multiplier`` - a single number the user computes themselves, applied
        as ``strain = raw_v * multiplier``. This is the documented input.
      * ``v_ex`` / ``gauge_factor`` / ``gain`` - the hardware constants, applied
        in the CLI scripts' exact operation order,
        ``((raw_v / gain) * (2 / gauge_factor)) / v_ex``.

    Both give the same answer to ~1e-15 relative, but float64 is not
    associative, so only the hardware form reproduces the old ``_threshold.csv``
    files bit-for-bit. Use it when you need to diff against previous outputs.
    """

    multiplier: float | None = None
    v_ex: float | None = None
    gauge_factor: float | None = None
    gain: float | None = None

    def __post_init__(self) -> None:
        if self.multiplier is None and None in (self.v_ex, self.gauge_factor, self.gain):
            raise ValueError(
                "StrainCalibration needs either multiplier, or all of "
                "v_ex/gauge_factor/gain"
            )

    @classmethod
    def coerce(cls, value: "float | StrainCalibration | None") -> "StrainCalibration":
        if value is None:
            return cls(multiplier=DEFAULT_STRAIN_MULTIPLIER)
        if isinstance(value, StrainCalibration):
            return value
        return cls(multiplier=float(value))

    @classmethod
    def from_hardware(cls, v_ex: float = V_EX, gauge_factor: float = GAUGE_FACTOR,
                      gain: float = GAIN) -> "StrainCalibration":
        return cls(v_ex=v_ex, gauge_factor=gauge_factor, gain=gain)

    def apply(self, raw_v: np.ndarray) -> np.ndarray:
        if self.multiplier is not None:
            return raw_v * self.multiplier
        return ((raw_v / self.gain) * (2 / self.gauge_factor)) / self.v_ex

    @property
    def effective_multiplier(self) -> float:
        if self.multiplier is not None:
            return self.multiplier
        return 2.0 / (self.gain * self.gauge_factor * self.v_ex)


@dataclass
class PulseWindow:
    """One detected pulse, in volts. Strain is applied later by ``strain()``."""

    label: str            # "Incident Pulse" | "Reflected Pulse 1" | "Transmission 1"
    kind: str             # INCIDENT | REFLECTED | TRANSMITTED
    channel: str
    idx_start: int
    idx_end: int
    time_abs: np.ndarray
    raw_v: np.ndarray     # abs(raw - mu) / scale, i.e. volts
    # Set when this window bounds exactly the same samples as an earlier one.
    # Happens on bar-on-bar tests: with no specimen between the bars there is no
    # impedance mismatch and so no reflected pulse, but find_peaks still returns
    # a second Channel C peak inside the same above-threshold excursion, and both
    # peaks' scans terminate at the same low-threshold crossings. Presentation
    # only - the pulse is still emitted, so the CSV schema and contents are
    # unchanged.
    duplicate_of: str | None = None

    @property
    def time_norm(self) -> np.ndarray:
        """Time since this pulse's own start, as the scripts plotted it."""
        return self.time_abs - self.time_abs[0] if len(self.time_abs) else self.time_abs

    def strain(self, calibration: "float | StrainCalibration | None" = None) -> np.ndarray:
        return StrainCalibration.coerce(calibration).apply(self.raw_v)

    @property
    def n(self) -> int:
        return len(self.time_abs)


@dataclass
class ChannelDiagnostics:
    """The traces test_analysis.py drew on its top diagnostic panel."""

    channel: str
    mu: float
    low_thr: float
    y_s: np.ndarray
    peak_indices: list[int]
    start_indices: list[int]


@dataclass
class AnalysisResult:
    name: str
    time_unit: str
    status: str                       # "ok" | "skipped"
    reason: str = ""
    messages: list[str] = field(default_factory=list)
    windows: list[PulseWindow] = field(default_factory=list)
    diagnostics: dict[str, ChannelDiagnostics] = field(default_factory=dict)
    time_axis: np.ndarray | None = None
    n_samples: int = 0

    @property
    def has_incident(self) -> bool:
        return any(w.kind == INCIDENT for w in self.windows)

    @property
    def has_reflection(self) -> bool:
        return any(w.kind == REFLECTED for w in self.windows)

    @property
    def has_transmission(self) -> bool:
        return any(w.kind == TRANSMITTED for w in self.windows)

    @property
    def has_distinct_reflection(self) -> bool:
        """True only if a reflected pulse was found that is not a duplicate."""
        return any(w.kind == REFLECTED and w.duplicate_of is None for w in self.windows)

    def window(self, kind: str) -> PulseWindow | None:
        return next((w for w in self.windows if w.kind == kind), None)


@dataclass
class DetectionSettings:
    """Everything the UI can retune when a detection comes out wrong.

    Defaults reproduce test_analysis.py exactly.
    """

    sg_win: int = 51
    sg_poly: int = 2
    baseline_frac: float = 0.1
    k_lo_margin: float = 2.0
    refl_prominence_min: float = 0.1
    refl_distance: int = 8000
    refl_height_mult: float = 5.0
    trans_width_min: int = 500
    trans_sg_win: int = 51
    max_trans_delay: float = DEFAULT_MAX_TRANS_DELAY
    refl_pulse_num: int = REFL_PULSE_NUM
    trans_pulse_num: int = TRANS_PULSE_NUM

    def kwargs_common(self) -> dict:
        return {
            "sg_poly": self.sg_poly,
            "baseline_frac": self.baseline_frac,
            "k_lo_margin": self.k_lo_margin,
            "refl_prominence_min": self.refl_prominence_min,
            "refl_distance": self.refl_distance,
            "refl_height_mult": self.refl_height_mult,
            "trans_width_min": self.trans_width_min,
        }


def analyse(
    scope: ScopeFile,
    incident_col: str,
    transmitted_col: str,
    settings: DetectionSettings | None = None,
) -> AnalysisResult:
    """Run detection + gating + truncation for one trace.

    ``incident_col`` is the bar carrying incident and reflected pulses
    (Channel C in the CLI scripts); ``transmitted_col`` is the far bar
    (Channel D).
    """
    settings = settings or DetectionSettings()
    t = scope.frame[scope.time_col].to_numpy(dtype=np.float64)
    unit = scope.time_unit

    result = AnalysisResult(
        name=scope.name, time_unit=unit, status="ok", time_axis=t, n_samples=len(t)
    )

    # --- 1. detection, one call per channel (same args as the CLI scripts) ---
    ref = first_contact_auto(
        t, scope.frame[incident_col], settings.refl_pulse_num, "reflection",
        sg_win=settings.sg_win, **settings.kwargs_common(),
    )
    ref_idx_L, ref_idx_R, ref_t_L, _ref_t_R, reflect_mu, ref_peaks, ys_c, low_c = ref

    trans = first_contact_auto(
        t, scope.frame[transmitted_col], settings.trans_pulse_num, "transmission",
        sg_win=settings.trans_sg_win, **settings.kwargs_common(),
    )
    trans_idx_L, trans_idx_R, trans_t_L, _trans_t_R, trans_mu, trans_peaks, ys_d, low_d = trans

    ref_idx_L, ref_idx_R = list(ref_idx_L), list(ref_idx_R)
    trans_idx_L, trans_idx_R = list(trans_idx_L), list(trans_idx_R)
    ref_peaks, trans_peaks = list(ref_peaks), list(trans_peaks)
    ref_t_L, trans_t_L = list(ref_t_L), list(trans_t_L)

    # --- 2. max transmission delay gate ---
    if len(ref_t_L) > 0 and len(trans_t_L) > 0:
        t_incident = ref_t_L[0]
        keep = [
            i for i in range(len(trans_t_L))
            if trans_t_L[i] <= (t_incident + settings.max_trans_delay)
        ]
        dropped = len(trans_t_L) - len(keep)
        if dropped:
            result.messages.append(
                f"{dropped} transmitted pulse(s) past the "
                f"{settings.max_trans_delay} {unit} gate were dropped."
            )
        trans_idx_L = [trans_idx_L[i] for i in keep]
        trans_idx_R = [trans_idx_R[i] for i in keep]
        trans_peaks = [trans_peaks[i] for i in keep if i < len(trans_peaks)]

    # --- 3. fail-safes (from batch_analysis.py) ---
    if len(ref_idx_L) == 0:
        result.status = "skipped"
        result.reason = "Could not detect the initial incident pulse."
        result.diagnostics = _diagnostics(
            incident_col, reflect_mu, low_c, ys_c, ref_peaks, ref_idx_L,
            transmitted_col, trans_mu, low_d, ys_d, trans_peaks, trans_idx_L,
        )
        return result

    # --- 4. truncate to the first pulses only ---
    ref_idx_L, ref_idx_R = ref_idx_L[:MAX_REFL_KEPT], ref_idx_R[:MAX_REFL_KEPT]
    ref_peaks = ref_peaks[:MAX_REFL_KEPT]
    trans_idx_L, trans_idx_R = trans_idx_L[:MAX_TRANS_KEPT], trans_idx_R[:MAX_TRANS_KEPT]
    trans_peaks = trans_peaks[:MAX_TRANS_KEPT]

    if len(ref_idx_L) <= 1 and len(trans_idx_L) == 0:
        result.status = "skipped"
        result.reason = "Found incident pulse, but missing BOTH reflection and transmission."

    # --- 5. build pulse windows (volts; strain applied on demand) ---
    for i, (a, b) in enumerate(zip(ref_idx_L, ref_idx_R)):
        result.windows.append(
            _window(
                scope, incident_col, reflect_mu, a, b,
                label="Incident Pulse" if i == 0 else f"Reflected Pulse {i}",
                kind=INCIDENT if i == 0 else REFLECTED,
            )
        )
    for i, (a, b) in enumerate(zip(trans_idx_L, trans_idx_R)):
        result.windows.append(
            _window(
                scope, transmitted_col, trans_mu, a, b,
                label=f"Transmission {i + 1}", kind=TRANSMITTED,
            )
        )

    _flag_duplicate_windows(result)

    if not result.has_reflection:
        result.messages.append("No reflection pulse detected; its CSV columns will be empty.")
    elif not result.has_distinct_reflection:
        result.messages.append(
            "No distinct reflection: the reflected window covers exactly the same "
            "samples as the incident pulse. Expected for a bar-on-bar test - with "
            "no specimen between the bars there is no impedance mismatch, so there "
            "is no reflected pulse. The CSV still contains the duplicated columns."
        )
    if not result.has_transmission:
        result.messages.append("No transmission pulse detected; its CSV columns will be empty.")

    result.diagnostics = _diagnostics(
        incident_col, reflect_mu, low_c, ys_c, ref_peaks, ref_idx_L,
        transmitted_col, trans_mu, low_d, ys_d, trans_peaks, trans_idx_L,
    )
    return result


def _flag_duplicate_windows(result: AnalysisResult) -> None:
    """Mark windows that bound exactly the same samples as an earlier window."""
    seen: dict[tuple[str, int, int], str] = {}
    for w in result.windows:
        key = (w.channel, w.idx_start, w.idx_end)
        if key in seen:
            w.duplicate_of = seen[key]
        else:
            seen[key] = w.label


def _window(scope, col, mu, idx_start, idx_end, label, kind) -> PulseWindow:
    """Slice one pulse. ``.loc[a:b]`` in the scripts was inclusive of b."""
    a, b = int(idx_start), int(idx_end)
    time_abs = scope.frame[scope.time_col].iloc[a:b + 1].to_numpy(dtype=np.float64)
    raw = scope.frame[col].iloc[a:b + 1].to_numpy(dtype=np.float64)
    raw_v = np.abs(raw - mu) / scope.scale_for(col)
    return PulseWindow(
        label=label, kind=kind, channel=col,
        idx_start=a, idx_end=b, time_abs=time_abs, raw_v=raw_v,
    )


def _diagnostics(col_c, mu_c, low_c, ys_c, peaks_c, starts_c,
                 col_d, mu_d, low_d, ys_d, peaks_d, starts_d) -> dict:
    out = {}
    if ys_c is not None and np.size(ys_c):
        out[col_c] = ChannelDiagnostics(
            channel=col_c, mu=float(mu_c), low_thr=float(low_c),
            y_s=np.asarray(ys_c), peak_indices=[int(i) for i in peaks_c],
            start_indices=[int(i) for i in starts_c],
        )
    if ys_d is not None and np.size(ys_d):
        out[col_d] = ChannelDiagnostics(
            channel=col_d, mu=float(mu_d), low_thr=float(low_d),
            y_s=np.asarray(ys_d), peak_indices=[int(i) for i in peaks_d],
            start_indices=[int(i) for i in starts_d],
        )
    return out


def to_threshold_frame(
    result: AnalysisResult,
    calibration_incident: "float | StrainCalibration | None" = None,
    calibration_transmitted: "float | StrainCalibration | None" = None,
) -> pd.DataFrame:
    """Build the padded ``_threshold.csv`` table.

    Schema matches the existing outputs exactly, so files written here stay
    readable by interface.py and delay_analysis.py:

        Shared Time (ms), Incident Pulse Time (ms), Incident Pulse Strain,
        Reflected Pulse 1 Time (ms), Reflected Pulse 1 Strain,
        Transmission 1 Time (ms), Transmission 1 Strain

    Missing pulses get present-but-empty columns, as batch_analysis.py did.
    """
    cal_inc = StrainCalibration.coerce(calibration_incident)
    cal_trn = (
        cal_inc if calibration_transmitted is None
        else StrainCalibration.coerce(calibration_transmitted)
    )
    unit = result.time_unit
    unit_sfx = f" ({unit})" if unit else ""

    columns: dict[str, np.ndarray] = {}
    max_len = 0
    shared_time: np.ndarray | None = None

    for w in result.windows:
        cal = cal_trn if w.kind == TRANSMITTED else cal_inc
        columns[f"{w.label} Time{unit_sfx}"] = w.time_abs
        columns[f"{w.label} Strain"] = w.strain(cal)
        if w.n > max_len:
            max_len = w.n
            shared_time = w.time_norm

    for label in ("Reflected Pulse 1", "Transmission 1"):
        if f"{label} Time{unit_sfx}" not in columns:
            columns[f"{label} Time{unit_sfx}"] = np.array([])
            columns[f"{label} Strain"] = np.array([])

    frame = pd.DataFrame(
        {f"Shared Time{unit_sfx}": shared_time if shared_time is not None else np.array([])}
    )
    for name, values in columns.items():
        frame[name] = pd.Series(values).reindex(range(max_len))
    return frame
