"""
Pulse detection for SHPB oscilloscope traces.

This is a faithful port of ``first_contact_auto`` from
``src/split-hopkinson_bar/Oscilloscope_Analysis/test_analysis.py`` (the newer of
the two CLI scripts, with the backward-scan fallbacks and diagnostic returns).
The numerical behaviour is intentionally identical; the only changes are:

  * ``find_peaks`` tuning values are function parameters instead of literals,
    so a bad detection can be corrected from the UI rather than by editing the
    script. Every default equals the value hardcoded in test_analysis.py.
  * the vestigial ``property`` argument is dropped - both CLI scripts accepted
    it but never used it for anything (batch_analysis.py assigned
    ``proms = np.argsort(properties[property])`` and then discarded it).
  * inputs are coerced to float64 numpy arrays up front (pandas Series in,
    identical numerics out) to avoid repeated Series overhead on 600k+ samples.

Subtleties preserved deliberately - do not "clean these up", they change results:

  * ``base`` is sliced from the smoothed signal BEFORE rectification. The line
    ``y_s = abs(y_s - mu) + mu`` rebinds ``y_s`` to a new array, so ``base``
    keeps referring to the pre-rectification data. mu/sigma are therefore
    baseline statistics of the *signed* signal.
  * sigma applies a 1.4826 factor on top of ``statsmodels.robust.mad``, which
    already normalises by c=0.6745. This double-scales the MAD relative to a
    textbook robust sigma. It is the existing behaviour and the thresholds are
    tuned around it.
  * the transmission branch computes a slope-refined index ``j`` and then
    overwrites it with ``peaks[0]``, keeping the slope logic only as a fallback
    for when no peak is found.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from statsmodels import robust

# Defaults lifted verbatim from test_analysis.py's first_contact_auto.
BASELINE_FRAC = 0.1
MIN_CONSEC = 1
K_HI_BOUNDS = (3.0, 10.0)
K_STEP = 0.5
K_LO_MARGIN = 2.0
SLOPE_MULT = 4.0
SG_WIN = 51
SG_POLY = 2

# Channel C (incident + reflected) peak finding.
REFL_PROMINENCE_MIN = 0.1
REFL_DISTANCE = 8000
REFL_HEIGHT_MULT = 5.0     # height=(REFL_HEIGHT_MULT * high_thr, None)

# Channel D (transmitted) peak finding.
TRANS_WIDTH_MIN = 500      # width=(TRANS_WIDTH_MIN, None), height=(high_thr, None)


class DetectionError(ValueError):
    """Raised when inputs are structurally unusable (not merely 'no pulse found')."""


def first_contact_auto(
    t,
    y,
    pulse_num: int,
    ref_or_trans: str,
    baseline_frac: float = BASELINE_FRAC,
    min_consec: int = MIN_CONSEC,
    k_hi_bounds: tuple[float, float] = K_HI_BOUNDS,
    k_step: float = K_STEP,
    k_lo_margin: float = K_LO_MARGIN,
    slope_mult: float = SLOPE_MULT,
    sg_win: int = SG_WIN,
    sg_poly: int = SG_POLY,
    refl_prominence_min: float = REFL_PROMINENCE_MIN,
    refl_distance: int = REFL_DISTANCE,
    refl_height_mult: float = REFL_HEIGHT_MULT,
    trans_width_min: int = TRANS_WIDTH_MIN,
):
    """Locate pulse windows in a single oscilloscope channel.

    Returns ``(index_lefts, index_rights, t_lefts, t_rights, mu, peak_indices,
    y_s, low_thr)`` exactly as test_analysis.py did.

    ``ref_or_trans`` selects the peak-finding strategy:
      * ``"reflection"`` - takes up to ``pulse_num`` peaks above
        ``refl_height_mult * high_thr``, separated by ``refl_distance`` samples.
        Used for Channel C, where peak 0 is the incident pulse and peak 1 the
        first reflection.
      * ``"transmission"`` - takes the first peak above ``high_thr`` that is at
        least ``trans_width_min`` samples wide. Used for Channel D.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(y)
    if n < 10:
        return [], [], [], [], 0, [], [], 0

    if ref_or_trans not in ("reflection", "transmission"):
        raise DetectionError(
            "Invalid entry for parameter ref_or_trans, must enter either "
            "reflection or transmission"
        )

    sg_win = int(sg_win) | 1
    y_s = savgol_filter(y, sg_win, sg_poly, mode="interp")

    n0 = max(50, int(baseline_frac * n))
    base = y_s[:n0]                      # pre-rectification view, see module docstring
    mu = np.median(base)
    sigma = 1.4826 * robust.mad(base) + 1e-12
    y_s = abs(y_s - mu) + mu             # rebinds y_s; base stays signed

    dt = np.median(np.diff(t))
    dy = savgol_filter(y_s, sg_win, sg_poly, deriv=1, delta=dt, mode="interp")
    slope_sigma = 1.4826 * robust.mad(dy[:n0]) + 1e-12
    slope_thr = slope_mult * slope_sigma

    def has_false_run(k_hi: float) -> bool:
        high = mu + k_hi * sigma
        ah = (base > high).astype(np.int8)
        run = np.convolve(ah, np.ones(min_consec, int), mode="same")
        return np.any(run >= min_consec)

    k_hi_candidates = np.arange(k_hi_bounds[0], k_hi_bounds[1] + 1e-9, k_step)
    chosen_k_hi = None
    for k in k_hi_candidates:
        if not has_false_run(k):
            chosen_k_hi = k
            break

    if chosen_k_hi is None:
        high = float(np.quantile(base, 0.999))
        chosen_k_hi = (high - mu) / sigma

    high_thr = mu + chosen_k_hi * sigma
    low_thr = max(mu + 1.5 * sigma, high_thr - k_lo_margin * sigma)
    low_thr = min(low_thr, high_thr - 0.5 * sigma)

    t_lefts: list[float] = []
    t_rights: list[float] = []
    index_lefts: list[int] = []
    index_rights: list[int] = []
    peak_indices: list[int] = []

    def scan_window(j: int) -> None:
        """Walk out from peak j to the low-threshold crossings on either side."""
        # Backward scan, with the argmin fallback added in test_analysis.py.
        i = j
        while i > 0 and y_s[i] > low_thr:
            i -= 1
        if i <= 0:
            i = int(np.argmin(y_s[:j])) if j > 0 else 0
            index_lefts.append(i)
            t_lefts.append(float(t[i]))
        else:
            y0, y1 = y_s[i], y_s[i + 1]
            t_cross = (
                float(t[i])
                if y1 == y0
                else float(t[i] + ((low_thr - y0) / (y1 - y0)) * (t[i + 1] - t[i]))
            )
            index_lefts.append(i)
            t_lefts.append(t_cross)

        # Forward scan.
        k = j
        while k < n and y_s[k] > low_thr:
            k += 1
        index_rights.append(min(k, n - 1))
        t_rights.append(float(t[min(k, n - 1)]))

    if ref_or_trans == "transmission":
        peaks, _ = find_peaks(y_s, height=(high_thr, None), width=(trans_width_min, None))
        ah_full = (y_s > high_thr).astype(np.int8)
        run_full = np.convolve(ah_full, np.ones(min_consec, int), mode="same")
        idxs = np.where((run_full >= min_consec) & (ah_full == 1))[0]
        if idxs.size == 0:
            return [], [], [], [], mu, [], y_s, low_thr

        j = int(idxs[0])
        if np.abs(dy[j]) < slope_thr:
            j2 = j + np.argmax(np.abs(dy[j:min(j + 6, n)]))
            if np.abs(dy[j2]) >= slope_thr:
                j = j2

        j = peaks[0] if len(peaks) > 0 else j
        peak_indices.append(int(j))
        scan_window(int(j))

    else:  # reflection
        strain_peaks, _properties = find_peaks(
            y_s,
            prominence=(refl_prominence_min, None),
            plateau_size=True,
            height=(refl_height_mult * high_thr, None),
            distance=refl_distance,
        )
        for peak in strain_peaks[:pulse_num]:
            peak_indices.append(int(peak))
            scan_window(int(peak))

    return index_lefts, index_rights, t_lefts, t_rights, mu, peak_indices, y_s, low_thr
