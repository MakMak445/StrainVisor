import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from statsmodels import robust

# ==========================================
# HARDWARE CALIBRATION CONSTANTS
# ==========================================
V_EX = 10.0          # Excitation Voltage (V)
GAUGE_FACTOR = 2.0   # Strain Gauge Factor
GAIN_C = 100.0       # Amplifier Gain for Channel C (Incident/Reflected)
GAIN_D = 100.0       # Amplifier Gain for Channel D (Transmitted)

# ==========================================
# 1. PEAK DETECTION & BASELINE (ORIGINAL)
# ==========================================
def first_contact_auto(
    t, y, pulse_num: int, property: str, ref_or_trans: str,
    baseline_frac=0.1,          
    min_consec=1,               
    k_hi_bounds=(3.0, 10.0),    
    k_step=0.5,                  
    k_lo_margin=2.0,             
    slope_mult=4.0,              
    sg_win=51, sg_poly=2,        
    prominence=(0.1, None), width=(None, None), plateau_size=True 
):
    n = len(y)
    if n < 10:
        return t[0], {"reason": "too_short"}

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
    
    t_lefts = []
    t_rights = []
    index_lefts = []
    index_rights = []
    
    if ref_or_trans == 'transmission':
        peaks, _ = find_peaks(y_s, prominence=(0.01, None), height=(None, None))
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
        if np.abs(dy[j]) < slope_thr:
            j2 = j + np.argmax(np.abs(dy[j:min(j+6, n)]))
            if np.abs(dy[j2]) >= slope_thr:
                j = j2
        j=peaks[0]
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
                index_lefts.append(i)
                t_lefts.append(t_cross)
        k=j
        while k < n and y_s[k] > low_thr:
            k += 1
        if i >= n:
            t_cross_back = float(t[n-1])
        else:
            y0, y1 = y_s[k], y_s[k-1]
            if y1 == y0:
                t_cross_back = float(t[k])
            else:
                a = (low_thr - y1) / (y0 - y1)
                t_cross_back = float(t[k-1] + a * (t[k] - t[k-1]))
                index_rights.append(k)
                t_rights.append(t_cross_back)

    elif ref_or_trans == 'reflection':
        strain_peaks, properties = find_peaks(y_s, prominence=(0.1, None), width=(None, None), plateau_size=True, height=(5*high_thr, None), distance = 5000)
        proms = np.argsort(properties[property])
        for peak in strain_peaks[:pulse_num]:
            prom1_strain_idx = peak
            j = peak
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
                    index_lefts.append(i)
                    t_lefts.append(t_cross)
            k=j
            while k < n and y_s[k] > low_thr:
                k += 1
            if i >= n:
                t_cross_back = float(t[n-1])
            else:
                y0, y1 = y_s[k], y_s[k-1]
                if y1 == y0:
                    t_cross_back = float(t[k])
                else:
                    a = (low_thr - y1) / (y0 - y1)
                    t_cross_back = float(t[k-1] + a * (t[k] - t[k-1]))
                    index_rights.append(k)
                    t_rights.append(t_cross_back)
    else: 
        raise RuntimeError("Invalid entry for parameter ref_or_trans, must enter either reflection or transmission")
        
    index_lefts, index_rights, t_lefts, t_rights = np.sort(index_lefts), np.sort(index_rights), np.sort(t_lefts), np.sort(t_rights)
    return index_lefts, index_rights, t_lefts, t_rights, mu

# ==========================================
# 2. PROCESSING AND OVERLAY FUNCTION
# ==========================================
def process_and_overlay(filepath, output_directory):
    print(f"\nProcessing: {os.path.basename(filepath)}")
    data = pd.read_csv(filepath, header=[0, 1])
    data.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in data.columns]
    
    # --- AUTO-DETECT UNITS AND SCALING ---
    if "Channel C (V)" in data.columns:
        col_c, scale_c = "Channel C (V)", 1.0
    elif "Channel C (mV)" in data.columns:
        col_c, scale_c = "Channel C (mV)", 1000.0
    else:
        print(f"Skipping {filepath}: Could not find Channel C in V or mV.")
        return

    if "Channel D (V)" in data.columns:
        col_d, scale_d = "Channel D (V)", 1.0
    elif "Channel D (mV)" in data.columns:
        col_d, scale_d = "Channel D (mV)", 1000.0
    else:
        print(f"Skipping {filepath}: Could not find Channel D in V or mV.")
        return

    # 1. Get Indices and Baselines using dynamically found column names
    ref_idx_L, ref_idx_R, ref_t_L, ref_t_R, reflect_mu = first_contact_auto(
        data["Time (ms)"], data[col_c], 2, 'prominences', 'reflection'
    )
    
    trans_idx_L, trans_idx_R, trans_t_L, trans_t_R, trans_mu = first_contact_auto(
        data["Time (ms)"], data[col_d], 1, 'prominences', 'transmission'
    )

    if len(ref_idx_L) < 2 or len(trans_idx_L) < 1:
        print(f"Skipping {filepath}: Could not detect all 3 required pulses.")
        return

    # --- Setup Plot and Dictionary for CSV (Original Logic) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    pulse_data_for_csv = {}
    max_pulse_length = 0
    shared_time_axis = None

    # --- Process Reflection Pulses (Channel C) ---
    for i, (idx_start, idx_end) in enumerate(zip(ref_idx_L, ref_idx_R)):
        time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
        time_normalized = time_pulse - time_pulse.iloc[0]
        
        # Extract raw, subtract baseline, apply scale (V vs mV), convert to Strain
        raw_v = abs(data.loc[idx_start:idx_end, col_c] - reflect_mu) / scale_c
        strain_pulse = ((raw_v / GAIN_C) * (2 / GAUGE_FACTOR)) / V_EX
        
        label_name = 'Incident Pulse' if i == 0 else f'Reflected Pulse {i}'
        ax.plot(time_normalized, strain_pulse, label=label_name)
        
        pulse_data_for_csv[f'{label_name} Time (ms)'] = time_pulse.values
        pulse_data_for_csv[f'{label_name} Strain'] = strain_pulse.values
        
        if len(time_normalized) > max_pulse_length:
            max_pulse_length = len(time_normalized)
            shared_time_axis = time_normalized.values

    # --- Process Transmission Pulses (Channel D) ---
    for i, (idx_start, idx_end) in enumerate(zip(trans_idx_L, trans_idx_R)):
        time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
        time_normalized = time_pulse - time_pulse.iloc[0]
        
        # Extract raw, subtract baseline, apply scale (V vs mV), convert to Strain
        raw_v = abs(data.loc[idx_start:idx_end, col_d] - trans_mu) / scale_d
        strain_pulse = ((raw_v / GAIN_D) * (2 / GAUGE_FACTOR)) / V_EX
        
        ax.plot(time_normalized, strain_pulse, label=f'Transmission Pulse {i+1}', linestyle='--')
        
        pulse_data_for_csv[f'Transmission {i+1} Time (ms)'] = time_pulse.values
        pulse_data_for_csv[f'Transmission {i+1} Strain'] = strain_pulse.values
        
        if len(time_normalized) > max_pulse_length:
            max_pulse_length = len(time_normalized)
            shared_time_axis = time_normalized.values

    # --- Final Plot Touches ---
    ax.axhline(0, color='black', linestyle=':', linewidth=1, label='Common Baseline')
    ax.set_title(f"Threshold Alignment: {os.path.basename(filepath)}")
    ax.set_xlabel('Time Since Pulse Start (ms)')
    ax.set_ylabel('Strain (\u03BC\u03B5)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Non-blocking user verification (Comment out the input() lines below for full automation)
    plt.draw()
    plt.pause(0.1) 
    
    cont = input('Save this data? (Y/N/Quit): ').strip().lower()
    
    if cont == 'q' or cont == 'quit':
        print("Batch processing aborted by user.")
        plt.close()
        sys.exit(0)
    elif cont == 'y':
        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Save Plot
        plt.savefig(os.path.join(output_directory, f"{base_name}_threshold.svg"), dpi=300)
        
        # Save CSV (Original NaN padding method)
        processed_df = pd.DataFrame({'Shared Time (ms)': shared_time_axis})
        for col_name, data_values in pulse_data_for_csv.items():
            padded_data = pd.Series(data_values).reindex(range(max_pulse_length))
            processed_df[col_name] = padded_data
            
        csv_path = os.path.join(output_directory, f"{base_name}_threshold.csv")
        processed_df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
    else:
        print("Discarded.")
    
    plt.close()

# ==========================================
# 3. BATCH ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # 1. Define your exact Input and Output folders here
    input_folder = "/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Oscilloscope_Analysis/picoscope_csv"
    output_folder = "/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Oscilloscope_Analysis/overlay_results2"
    
    # 2. Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
    else:
        print(f"Found {len(csv_files)} files. Starting batch processing...")
        for file in sorted(csv_files):
            process_and_overlay(file, output_folder)
        
        print("\nBatch processing complete.")