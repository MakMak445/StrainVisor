import numpy as np
from scipy.signal import savgol_filter
from statsmodels import robust
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def first_contact_auto(
    t, y, pulse_num: int, property: str, ref_or_trans: str,
    baseline_frac=0.1,          # fraction of the start used as baseline
    min_consec=1,               # required consecutive samples above High
    k_hi_bounds=(3.0, 10.0),     # search range for High in sigma units
    k_step=0.5,                  # search step for k_hi
    k_lo_margin=2.0,             # Low = (k_hi - k_lo_margin)*sigma above median
    slope_mult=4.0,              # derivative threshold = slope_mult * MAD(dy_baseline)
    sg_win=51, sg_poly=2,        # Savitzky–Golay smoothing (odd window)
    prominence=(0.1, None), width=(None, None), plateau_size=True #find_peaks parameters
):
    """
    Auto-tune hysteresis thresholds from baseline and return first-contact time.
    Returns: t_contact, info (dict with thresholds/diagnostics)
    """
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
    y_s = abs(y_s - mu) + mu
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
    t_lefts = []
    t_rights = []
    index_lefts = []
    index_rights = []
    if ref_or_trans == 'transmission':
        # --- find first confirmed event in full series
        peaks, _ = find_peaks(y_s, prominence=(0.01, None), height=(None, None))
        plt.plot(t, y)
        for peak in peaks: plt.plot(t[peak], y[peak], 'X', markersize=10, color='red', label='peak')
        plt.show()
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
            #print(k)
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
        plt.plot(t, y)
        plt.axvline(t[i], color='red')
        plt.axvline(t[k], color='red')
        plt.show()

    elif ref_or_trans == 'reflection':
        strain_peaks, properties = find_peaks(y_s, prominence=(0.1, None), width=(None, None), plateau_size=True, height=(5*high_thr, None), distance = 5000)
        plt.plot(t,y)
        for peak in strain_peaks: 
            plt.plot(t[peak], y[peak], 'X', markersize=10, color='red', label='peak')
        plt.show()
        proms = np.argsort(properties[property])
        for peak in strain_peaks[:pulse_num]:
            prom1_strain_idx = peak
            j = peak#round((properties["left_edges"][prom1_strain_idx] + properties["right_edges"][prom1_strain_idx])/2)
            #plt.plot(t, abs(y), '.')
            #plt.plot(t[j], y_s[j], 'X', markersize=10, color='red', label='peak')
            #plt.show()
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
                    index_lefts.append(i)
                    t_lefts.append(t_cross)
            k=j
            while k < n and y_s[k] > low_thr:
                k += 1
                #print(k)
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
                    plt.plot(t, y)
                    plt.axvline(t[i], color='red')
                    plt.axvline(t[k], color='red')
                    plt.show()
    else: raise RuntimeError("Invalid entry for parameter ref_or_trans, must enter either reflection or transmission")
    info = {
        "mu": mu, "sigma": sigma,
        "slope_sigma": slope_sigma, "slope_thr": slope_thr,
        "k_hi": float(chosen_k_hi), "k_lo_eff": float((low_thr - mu) / sigma),
        "low_thr": float(low_thr), "high_thr": float(high_thr),
        "baseline_len": int(n0), "min_consec": int(min_consec), 
        "first_index": i
    }
    index_lefts, index_rights, t_lefts, t_rights = np.sort(index_lefts), np.sort(index_rights), np.sort(t_lefts), np.sort(t_rights)
    return index_lefts, index_rights, t_lefts, t_rights, mu
2
'''
data = pd.read_csv("/home/makmak/Projects/cv2/Images/Picoscope/picoscope csv/1d9bar_confined_Alu_Fine.csv", header=[0, 1])
#print(data)
#print(data.iloc[1, 0])
data.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in data.columns]
#print(data)
cross_time, cross_back_time, low_thresh, high_thresh = first_contact_auto(data.loc[:, "Time (ms)"], data.loc[:, "Channel C (V)"])
plt.figure()
plt.plot(data.loc[:, "Time (ms)"], data.loc[:, "Channel D (V)"])
#plt.plot(data.loc[:, "Time (ms)"], data.loc[:, "Channel A (V)"])
plt.axvline(cross_time, 0, 1)
plt.axvline(cross_back_time, 0, 1)
plt.axhline(-high_thresh, 0, 1)
plt.axhline(high_thresh, 0, 1)
plt.axhline(-low_thresh, 0, 1)
plt.axhline(low_thresh, 0, 1)
print(cross_time, cross_back_time)
plt.show()
'''

def overlay(filepath):
    data = pd.read_csv(filepath, header=[0, 1])
    data.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in data.columns]
    reflect_index_lefts, reflect_index_rights, reflect_t_lefts, reflect_t_rights, reflect_mu = first_contact_auto(data.loc[:, "Time (ms)"], 
                                                                                                       data.loc[:, "Channel C (V)"], 
                                                                                                       2, 'prominences', 'reflection'
                                                                                                       )
    data.replace()
    pulse_index_range = reflect_index_rights[0] - reflect_index_lefts[0]
    trans_index_lefts, trans_index_rights, trans_t_lefts, trans_t_rights, trans_mu = first_contact_auto(data.loc[:, "Time (ms)"], 
                                                                                                                          data.loc[:, "Channel D (V)"], 
                                                                                                                          1, 'prominences', 'transmission'
                                                                                                        )
    if np.isclose(np.mean(reflect_index_rights), reflect_index_rights[0], atol = 100):
        reflect_index_lefts, reflect_index_rights, reflect_t_lefts, reflect_t_rights, reflect_mu = first_contact_auto(data.loc[:, "Time (ms)"], 
                                                                                                       data.loc[:, "Channel C (V)"], 
                                                                                                       2, 'prominences', 'reflection'
                                                                                                       )
    '''                                                                                                                         
    reflection_signal = data.loc[:, "Channel C (V)"].copy()
    time = data.loc[:, "Time (ms)"].copy()
    #reflection_signal[pulse_start_index:pulse_end_index] = reflect_base
    strain_peaks, properties = find_peaks(abs(reflection_signal), prominence=(0.1, None), width=(None, None), plateau_size=True)
    proms = np.argsort(properties["prominences"])
    prom1_strain_idx = proms[-1]
    prom2_strain_idx = proms[-2]
    '''
    #fig = plt.figure(figsize=(12, 6))
    #plt.plot(time, reflection_signal, '.', label = 'Data')
    #for peak in strain_peaks: plt.plot(time[peak], reflection_signal[peak], 'X', markersize=10, color='red', label='peak')

        # --- Create the figure and axes for the overlay plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

   # This dictionary will hold the absolute time/signal pairs for the CSV.
    pulse_data_for_csv = {}
    # These will track the longest pulse to create the shared time axis.
    max_pulse_length = 0
    shared_time_axis = None

    # --- Process and Plot Reflection Pulses (Channel C) ---
    # Loop through each detected reflection pulse using its start and end indices
    for i, (idx_start, idx_end) in enumerate(zip(reflect_index_lefts, reflect_index_rights)):
        # Slice the time segment for this specific pulse
        time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
        # Normalize the time segment to start at 0
        time_normalized = time_pulse - time_pulse.iloc[0]

        # Slice the signal segment and correct its baseline
        signal_pulse = abs(data.loc[idx_start:idx_end, "Channel C (V)"] - reflect_mu)
        
        # Plot the isolated, corrected pulse
        ax.plot(time_normalized, signal_pulse, label=f'Reflection Pulse {i+1}')
        # Define column names for this pulse's absolute time and signal.
        time_col = f'Reflected Time {i+1} (ms)'
        signal_col = f'Reflected Signal {i+1} (V)'

        # Store the absolute time and signal values.
        pulse_data_for_csv[time_col] = time_pulse.values
        pulse_data_for_csv[signal_col] = signal_pulse.values

        # Use the NORMALIZED time to find the longest pulse, which sets the 
        # length for the "Shared Time" column.
        if len(time_normalized) > max_pulse_length:
            max_pulse_length = len(time_normalized)
            shared_time_axis = time_normalized.values


    # --- Process and Plot Transmission Pulses (Channel D) ---
    # Loop through each detected transmission pulse
    if trans_t_lefts[0]<=reflect_t_rights[1]:
        transmission = True
        for i, (idx_start, idx_end) in enumerate(zip(trans_index_lefts, trans_index_rights)):
            # Slice and normalize the time segment
            time_pulse = data.loc[idx_start:idx_end, "Time (ms)"]
            time_normalized = time_pulse - time_pulse.iloc[0]

            # Slice the signal and correct its baseline
            signal_pulse = abs(data.loc[idx_start:idx_end, "Channel D (V)"] - trans_mu)
            
            # Plot the isolated, corrected pulse
            ax.plot(time_normalized, signal_pulse, label=f'Transmission Pulse {i+1}', linestyle='--')
           # Define column names for this pulse's absolute time and signal.
            time_col = f'Transmission Time {i+1} (ms)'
            signal_col = f'Transmission Signal {i+1} (V)'

            # Store the absolute time and signal values.
            pulse_data_for_csv[time_col] = time_pulse.values
            pulse_data_for_csv[signal_col] = signal_pulse.values

            # Update the max length if this pulse is longer.
            if len(time_normalized) > max_pulse_length:
                max_pulse_length = len(time_normalized)
                shared_time_axis = time_normalized.values
    else: 
        print('No valid transmission pulse detected')
        transmission = False

    # --- Final Touches ---
    ax.axhline(0, color='black', linestyle=':', linewidth=1, label='Common Baseline')
    ax.set_title('Overlay of Detected Pulses')
    ax.set_xlabel('Time Since Pulse Start (ms)')
    ax.set_ylabel('Baseline-Corrected Voltage (V)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    cont = input('Are these pulses acceptable (Do not worry about order of pulses)? (Y/N)').lower().strip()
    if cont == 'y':
        output_directory = "/home/makmak/Projects/cv2/src/split-hopkinson_bar/Oscilloscope_Analysis/overlay_results"
        new_filename = os.path.splitext(os.path.basename(filepath))[0] + ".svg"
        output_path = os.path.join(output_directory, new_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # 1. Create the final DataFrame, starting with the zero-based "Shared Time" axis.
        processed_df = pd.DataFrame({'Shared Time (ms)': shared_time_axis})

        # 2. Add the pairs of absolute time and signal columns to the DataFrame.
        # Shorter pulses will be padded with 'NaN' to match the longest pulse.
        for col_name, data_values in pulse_data_for_csv.items():
            padded_data = pd.Series(data_values).reindex(range(max_pulse_length))
            processed_df[col_name] = padded_data

        # 3. Construct the output path for the new CSV file.
        csv_filename = os.path.splitext(os.path.basename(filepath))[0] + "_processed_pulses.csv"
        csv_output_path = os.path.join(output_directory, csv_filename)

        # 4. Save the DataFrame to the CSV file.
        processed_df.to_csv(csv_output_path, index=False)
        print(f"\nProcessed pulse data saved to: {csv_output_path}")
        plt.show()
    else:
        plt.show()
        return 