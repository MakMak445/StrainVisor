import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from statsmodels import robust
import sys
plt.ion() # Turns on Interactive Mode
# ==========================================
# HARDWARE CALIBRATION CONSTANTS
# ==========================================
V_EX = 10.0
GAUGE_FACTOR = 2.04
GAIN_C = 100.0
GAIN_D = 100.0

# ==========================================
# 1. FIXED PEAK DETECTION (WITH DIAGNOSTIC EXPORTS)
# ==========================================
def first_contact_auto(
    t, y, pulse_num: int, property: str, ref_or_trans: str,
    baseline_frac=0.1, min_consec=1, k_hi_bounds=(3.0, 10.0), 
    k_step=0.5, k_lo_margin=2.0, slope_mult=4.0, 
    sg_win=51, sg_poly=2, prominence=(0.1, None)
):
    n = len(y)
    if n < 10: return [], [], [], [], 0, [], [], 0

    sg_win = int(sg_win) | 1
    y_s = savgol_filter(y, sg_win, sg_poly, mode="interp")

    n0 = max(50, int(baseline_frac * n))
    base = y_s[:n0]
    mu = np.median(base)
    sigma = 1.4826 * robust.mad(base) + 1e-12
    y_s = abs(y_s - mu) + mu
    
    dt = np.median(np.diff(t))
    dy = savgol_filter(y_s, sg_win, sg_poly, deriv=1, delta=dt, mode="interp")
    slope_sigma = 1.4826 * robust.mad(dy[:n0]) + 1e-12
    slope_thr = slope_mult * slope_sigma

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

    if chosen_k_hi is None:
        high = float(np.quantile(base, 0.999))
        chosen_k_hi = (high - mu) / sigma
        
    high_thr = mu + chosen_k_hi * sigma
    low_thr = max(mu + 1.5 * sigma, high_thr - k_lo_margin * sigma)
    low_thr = min(low_thr, high_thr - 0.5 * sigma)
    
    t_lefts, t_rights, index_lefts, index_rights = [], [], [], []
    peak_indices = []
    
    if ref_or_trans == 'transmission':
        peaks, _ = find_peaks(y_s, height=(high_thr, None), width=(500, None))
        ah_full = (y_s > high_thr).astype(np.int8)
        run_full = np.convolve(ah_full, np.ones(min_consec, int), mode="same")
        idxs = np.where((run_full >= min_consec) & (ah_full == 1))[0]
        if idxs.size == 0: return [], [], [], [], mu, [], y_s, low_thr

        j = int(idxs[0])
        if np.abs(dy[j]) < slope_thr:
            j2 = j + np.argmax(np.abs(dy[j:min(j+6, n)]))
            if np.abs(dy[j2]) >= slope_thr: j = j2
            
        j = peaks[0] if len(peaks) > 0 else j
        peak_indices.append(j)
        
        # --- FIXED BACKWARD SCAN WITH FALLBACK ---
        i = j
        while i > 0 and y_s[i] > low_thr: i -= 1
        if i <= 0:
            i = int(np.argmin(y_s[:j])) if j > 0 else 0
            t_cross = float(t[i])
            index_lefts.append(i)
            t_lefts.append(t_cross)
        else:
            y0, y1 = y_s[i], y_s[i+1]
            t_cross = float(t[i]) if y1 == y0 else float(t[i] + ((low_thr - y0) / (y1 - y0)) * (t[i+1] - t[i]))
            index_lefts.append(i)
            t_lefts.append(t_cross)
            
        k = j
        while k < n and y_s[k] > low_thr: k += 1
        index_rights.append(min(k, n-1))
        t_rights.append(float(t[min(k, n-1)]))

    elif ref_or_trans == 'reflection':
        # Your updated parameters
        strain_peaks, properties = find_peaks(y_s, prominence=(0.1, None), plateau_size=True, height=(5*high_thr, None), distance=8000)
        for peak in strain_peaks[:pulse_num]:
            j = peak
            peak_indices.append(j)
            
            # --- FIXED BACKWARD SCAN WITH FALLBACK ---
            i = j
            while i > 0 and y_s[i] > low_thr: i -= 1
            if i <= 0:
                i = int(np.argmin(y_s[:j])) if j > 0 else 0
                t_cross = float(t[i])
                index_lefts.append(i)
                t_lefts.append(t_cross)
            else:
                y0, y1 = y_s[i], y_s[i+1]
                t_cross = float(t[i]) if y1 == y0 else float(t[i] + ((low_thr - y0) / (y1 - y0)) * (t[i+1] - t[i]))
                index_lefts.append(i)
                t_lefts.append(t_cross)
                
            k = j
            while k < n and y_s[k] > low_thr: k += 1
            index_rights.append(min(k, n-1))
            t_rights.append(float(t[min(k, n-1)]))

    return index_lefts, index_rights, t_lefts, t_rights, mu, peak_indices, y_s, low_thr

# ==========================================
# 2. SINGLE FILE PROCESSING
# ==========================================
def process_single_file(filepath, output_directory):
    print(f"\nProcessing: {os.path.basename(filepath)}")
    data = pd.read_csv(filepath, header=[0, 1])
    data.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in data.columns]
    
    col_c = "Channel C (mV)" if "Channel C (mV)" in data.columns else "Channel C (V)"
    scale_c = 1000.0 if "(mV)" in col_c else 1.0
    
    col_d = "Channel D (mV)" if "Channel D (mV)" in data.columns else "Channel D (V)"
    scale_d = 1000.0 if "(mV)" in col_d else 1.0

    # Incident & Reflection
    ref_idx_L, ref_idx_R, ref_t_L, ref_t_R, reflect_mu, ref_peaks, ys_c, low_c = first_contact_auto(
        data["Time (ms)"], data[col_c], 5, 'prominences', 'reflection'
    )
    
    # Transmission 
    trans_idx_L, trans_idx_R, trans_t_L, trans_t_R, trans_mu, trans_peaks, ys_d, low_d = first_contact_auto(
        data["Time (ms)"], data[col_d], 3, 'prominences', 'transmission', sg_win=51
    )

    # Max Transmission Time Gate
    if len(ref_t_L) > 0 and len(trans_t_L) > 0:
        t_incident = ref_t_L[0]
        filtered_trans_idx_L, filtered_trans_idx_R, filtered_trans_t_L, filtered_trans_t_R, filtered_trans_peaks = [], [], [], [], []
        
        for i in range(len(trans_t_L)):
            if trans_t_L[i] <= (t_incident + 0.25):
                filtered_trans_idx_L.append(trans_idx_L[i])
                filtered_trans_idx_R.append(trans_idx_R[i])
                filtered_trans_t_L.append(trans_t_L[i])
                filtered_trans_t_R.append(trans_t_R[i])
                if i < len(trans_peaks): filtered_trans_peaks.append(trans_peaks[i])
                
        trans_idx_L, trans_idx_R = filtered_trans_idx_L, filtered_trans_idx_R
        trans_t_L, trans_t_R = filtered_trans_t_L, filtered_trans_t_R
        trans_peaks = filtered_trans_peaks

    # Truncate to first pulses
    ref_idx_L, ref_idx_R, ref_peaks = ref_idx_L[:2], ref_idx_R[:2], ref_peaks[:2]
    trans_idx_L, trans_idx_R, trans_peaks = trans_idx_L[:1], trans_idx_R[:1], trans_peaks[:1]

    # --- Setup 2-Panel Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    time_arr = data["Time (ms)"].values

    # ==========================================
    # AXIS 1: DIAGNOSTIC (Smoothed Data, Thresholds, Peaks & Starts)
    # ==========================================
    ax1.plot(time_arr, ys_c, label='Channel C (Smoothed)', color='steelblue', alpha=0.8)
    ax1.axhline(low_c, color='steelblue', linestyle='--', alpha=0.5, label='Ch C Low Threshold')
    
    ax1.plot(time_arr, ys_d, label='Channel D (Smoothed)', color='lightcoral', alpha=0.8)
    ax1.axhline(low_d, color='lightcoral', linestyle='--', alpha=0.5, label='Ch D Low Threshold')
    
    # Plot the detected peaks (X)
    if len(ref_peaks) > 0:
        ax1.scatter(time_arr[ref_peaks], ys_c[ref_peaks], color='blue', s=100, marker='x', linewidth=2, zorder=5, label='Chosen Peaks (C)')
    if len(trans_peaks) > 0:
        ax1.scatter(time_arr[trans_peaks], ys_d[trans_peaks], color='red', s=100, marker='x', linewidth=2, zorder=5, label='Chosen Peaks (D)')
        
    # Plot the detected start times (O)
    if len(ref_idx_L) > 0:
        ax1.scatter(time_arr[ref_idx_L], ys_c[ref_idx_L], color='black', s=60, marker='o', zorder=6, label='Detected Pulse Starts')
    if len(trans_idx_L) > 0:
        ax1.scatter(time_arr[trans_idx_L], ys_d[trans_idx_L], color='black', s=60, marker='o', zorder=6)

    ax1.set_title(f"Peak & Start Diagnostic: {os.path.basename(filepath)}")
    ax1.set_xlabel('Absolute Time (ms)')
    ax1.set_ylabel('Processed Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ==========================================
    # AXIS 2: ALIGNED STRAIN
    # ==========================================
    pulse_data_for_csv = {}
    max_pulse_length = 0
    shared_time_axis = None

    for i, (idx_start, idx_end) in enumerate(zip(ref_idx_L, ref_idx_R)):
        time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
        time_normalized = time_pulse - time_pulse.iloc[0]
        
        raw_v = abs(data.loc[idx_start:idx_end, col_c] - reflect_mu) / scale_c
        strain_pulse = ((raw_v / GAIN_C) * (2 / GAUGE_FACTOR)) / V_EX
        
        label_name = 'Incident Pulse' if i == 0 else f'Reflected Pulse {i}'
        ax2.plot(time_normalized, strain_pulse, label=label_name)
        
        pulse_data_for_csv[f'{label_name} Time (ms)'] = time_pulse.values
        pulse_data_for_csv[f'{label_name} Strain'] = strain_pulse.values
        
        if len(time_normalized) > max_pulse_length:
            max_pulse_length = len(time_normalized)
            shared_time_axis = time_normalized.values

    for i, (idx_start, idx_end) in enumerate(zip(trans_idx_L, trans_idx_R)):
        time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
        time_normalized = time_pulse - time_pulse.iloc[0]
        
        raw_v = abs(data.loc[idx_start:idx_end, col_d] - trans_mu) / scale_d
        strain_pulse = ((raw_v / GAIN_D) * (2 / GAUGE_FACTOR)) / V_EX
        
        ax2.plot(time_normalized, strain_pulse, label=f'Transmission Pulse {i+1}', linestyle='--')
        
        pulse_data_for_csv[f'Transmission {i+1} Time (ms)'] = time_pulse.values
        pulse_data_for_csv[f'Transmission {i+1} Strain'] = strain_pulse.values
        
        if len(time_normalized) > max_pulse_length:
            max_pulse_length = len(time_normalized)
            shared_time_axis = time_normalized.values

    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.set_title("Aligned Strain Output")
    ax2.set_xlabel('Time Since Pulse Start (ms)')
    ax2.set_ylabel('Strain (\u03BC\u03B5)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()

    # ==========================================
    # INTERACTIVE PROMPT (FIXED GUI RENDERING)
    # ==========================================
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5) # Force the script to wait half a second so the GUI can paint
    
    cont = input('\nSave this data? (Y/N/Quit): ').strip().lower()
    
    if cont == 'q' or cont == 'quit':
        print("Processing aborted by user.")
        plt.close('all')
        sys.exit(0)
    elif cont == 'y':
        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Save Plot
        plt.savefig(os.path.join(output_directory, f"{base_name}_threshold.svg"), dpi=300)
        
        # Save CSV
        processed_df = pd.DataFrame({'Shared Time (ms)': shared_time_axis})
        for col_name, data_values in pulse_data_for_csv.items():
            processed_df[col_name] = pd.Series(data_values).reindex(range(max_pulse_length))
            
        csv_path = os.path.join(output_directory, f"{base_name}_threshold.csv")
        processed_df.to_csv(csv_path, index=False)
        
        print(f"Plot saved to: {os.path.join(output_directory, f'{base_name}_threshold.svg')}")
        print(f"CSV saved to: {csv_path}")
    else:
        print("Discarded.")
    
    plt.close('all')

if __name__ == "__main__":
    # --- CONFIGURE PATHS HERE ---
    FILE_TO_FIX = "/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Oscilloscope_Analysis/picoscope_csv/1d9bar_confined_Pastic_FineSand-0002.csv"
    
    # Save it directly into your baseline folder so delay_analysis.py finds it immediately
    OUTPUT_FOLDER = "/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Oscilloscope_Analysis/overlay_results3/tests"
    
    process_single_file(FILE_TO_FIX, OUTPUT_FOLDER)