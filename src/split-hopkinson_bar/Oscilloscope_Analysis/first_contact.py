import numpy as np
from scipy.signal import savgol_filter
from statsmodels import robust
import pandas as pd
import matplotlib.pyplot as plt

def first_contact_auto(
    t, y,
    baseline_frac=0.1,          # fraction of the start used as baseline
    min_consec=1,               # required consecutive samples above High
    k_hi_bounds=(3.0, 10.0),     # search range for High in sigma units
    k_step=0.5,                  # search step for k_hi
    k_lo_margin=2.0,             # Low = (k_hi - k_lo_margin)*sigma above median
    slope_mult=4.0,              # derivative threshold = slope_mult * MAD(dy_baseline)
    sg_win=51, sg_poly=2         # Savitzky–Golay smoothing (odd window)
):
    """
    Auto-tune hysteresis thresholds from baseline and return first-contact time.
    Returns: t_contact, info (dict with thresholds/diagnostics)
    """
    y=abs(y)
    n = len(y)
    if n < 10:
        return t[0], {"reason": "too_short"}

    # --- zero-phase smoothing (small window)
    sg_win = int(sg_win) | 1
    y_s = savgol_filter(y, sg_win, sg_poly, mode="interp")

    # --- baseline window
    n0 = max(50, int(baseline_frac * n))
    base = y_s[:n0]
    mu = np.median(base)
    sigma = 1.4826 * robust.mad(base) + 1e-12

    # --- derivative & slope threshold from baseline dynamics
    dt = np.median(np.diff(t))
    dy = savgol_filter(y_s, sg_win, sg_poly, deriv=1, delta=dt, mode="interp")
    slope_sigma = 1.4826 * robust.mad(dy[:n0]) + 1e-12
    slope_thr = slope_mult * slope_sigma

    # --- choose the smallest k_hi with ZERO false runs in baseline
    def has_false_run(k_hi):
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

    # Fallback: if even huge k_hi still has runs (super noisy baseline),
    # switch to empirical-quantile High at 99.9% of baseline
    if chosen_k_hi is None:
        high = float(np.quantile(base, 0.999))
        chosen_k_hi = (high - mu) / sigma
    high_thr = mu + chosen_k_hi * sigma

    # --- choose Low below High but still well above baseline
    # Low = max(median + 1.5σ, High - k_lo_margin*σ) but < High - 0.5σ
    low_thr = max(mu + 1.5 * sigma, high_thr - k_lo_margin * sigma)
    low_thr = min(low_thr, high_thr - 0.5 * sigma)

    # --- find first confirmed event in full series
    ah_full = (y_s > high_thr).astype(np.int8)
    run_full = np.convolve(ah_full, np.ones(min_consec, int), mode="same")
    idxs = np.where((run_full >= min_consec) & (ah_full == 1))[0]
    if idxs.size == 0:
        return t[0], {
            "reason": "no_event",
            "mu": mu, "sigma": sigma,
            "k_hi": chosen_k_hi, "low_thr": low_thr, "high_thr": high_thr
        }

    j = int(idxs[0])

    # slope check: if too flat, nudge forward to a steeper point nearby
    if np.abs(dy[j]) < slope_thr:
        j2 = j + np.argmax(np.abs(dy[j:min(j+6, n)]))
        if np.abs(dy[j2]) >= slope_thr:
            j = j2

    # --- walk back to Low and interpolate precise crossing
    i = j
    while i > 0 and y_s[i] > low_thr:
        i -= 1
    if i <= 0:
        t_cross = float(t[0])
    else:
        y0, y1 = y_s[i], y_s[i+1]
        if y1 == y0:
            t_cross = float(t[i])
        else:
            a = (low_thr - y0) / (y1 - y0)
            t_cross = float(t[i] + a * (t[i+1] - t[i]))

    while i < n and y_s[i] > low_thr:
        i += 1
    if i >= n:
        t_cross_back = float(t[n-1])
    else:
        y0, y1 = y_s[i], y_s[i-1]
        if y1 == y0:
            t_cross_back = float(t[i])
        else:
            a = (low_thr - y1) / (y0 - y1)
            t_cross_back = float(t[i] + a * (t[i]) - t[i-1])

    info = {
        "mu": mu, "sigma": sigma,
        "slope_sigma": slope_sigma, "slope_thr": slope_thr,
        "k_hi": float(chosen_k_hi), "k_lo_eff": float((low_thr - mu) / sigma),
        "low_thr": float(low_thr), "high_thr": float(high_thr),
        "baseline_len": int(n0), "min_consec": int(min_consec), 
        "first_index": i
    }
    return t_cross, t_cross_back, mu


data = pd.read_csv("/home/makmak/Projects/cv2/Images/Picoscope/picoscope csv/1d9bar_confined_Alu_Fine.csv", header=[0, 1])
#print(data)
#print(data.iloc[1, 0])
data.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in data.columns]
#print(data)
cross_time, cross_back_time, mu = first_contact_auto(data.loc[:, "Time (ms)"], data.loc[:, "Channel D (V)"])
plt.figure()
plt.plot(data.loc[:, "Time (ms)"], data.loc[:, "Channel D (V)"])
plt.plot(data.loc[:, "Time (ms)"], data.loc[:, "Channel C (V)"])
plt.axvline(cross_time, 0, 1)
plt.axvline(cross_back_time, 0, 1)
print(cross_time, cross_back_time)
plt.show()