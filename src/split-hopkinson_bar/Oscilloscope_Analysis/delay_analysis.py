import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_pulse_delay(filepath):
    """Extracts the absolute time delay between Incident and Transmission pulses."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return None
        
    inc_time_cols = [col for col in df.columns if 'Incident' in col and 'Time' in col]
    trans_time_cols = [col for col in df.columns if 'Transmission' in col and 'Time' in col]
    
    if not inc_time_cols or not trans_time_cols:
        print(f"  -> Missing required time columns in {os.path.basename(filepath)}")
        return None
        
    clean_inc = df[inc_time_cols[0]].dropna()
    clean_trans = df[trans_time_cols[0]].dropna()
    
    # If there is no incident wave, the file is invalid
    if clean_inc.empty:
        print(f"  -> Skipping {os.path.basename(filepath)}: No incident pulse detected.")
        return None
        
    # If incident exists but transmission is empty, return the text flag
    if clean_trans.empty:
        return "No Transmission"
        
    t_incident_start = clean_inc.iloc[0]
    t_transmission_start = clean_trans.iloc[0]
    
    return t_transmission_start - t_incident_start

def process_velocity_group(base_folder, particulate_folder, velocity_label):
    """Calculates material delays for a specific velocity group."""
    print(f"\n--- Processing {velocity_label} ---")
    
    # 1. Find and calculate baseline
    base_files = glob.glob(os.path.join(base_folder, "*.csv"))
    if not base_files:
        print(f"Error: No baseline CSV found in {base_folder}")
        return []
    if len(base_files) > 1:
        print(f"Warning: Multiple files in {base_folder}. Using {os.path.basename(base_files[0])}")
        
    baseline_delay = get_pulse_delay(base_files[0])
    if baseline_delay is None or baseline_delay == "No Transmission":
        print("Error: Baseline file lacks a valid transmission pulse.")
        return []
        
    print(f"Calibration Baseline: {baseline_delay:.5f} ms")
    
    # 2. Process all particulates for this velocity
    results = []
    specimen_files = glob.glob(os.path.join(particulate_folder, "*.csv"))
    
    if not specimen_files:
        print(f"No particulate files found in {particulate_folder}")
        return results
        
    for file in sorted(specimen_files):
        specimen_delay = get_pulse_delay(file)
        
        if specimen_delay is not None:
            # Handle the string flag vs numeric calculation
            if specimen_delay == "No Transmission":
                material_delay = "No Transmission"
            else:
                material_delay = specimen_delay - baseline_delay
                
            filename = os.path.basename(file).replace('_threshold.csv', '').replace('.csv', '')
            
            confinement = "Unknown"
            if "al" in filename.lower():
                confinement = "Aluminium"
            elif "plas" in filename.lower():
                confinement = "Plastic"
            
            results.append({
                "Specimen": filename,
                "Velocity": velocity_label,
                "Confinement": confinement,
                "Material Delay (ms)": material_delay
            })
            
    return results

def main():
    # ==========================================
    # SET YOUR MAIN DIRECTORY HERE
    # ==========================================
    ROOT_DIR = "/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Oscilloscope_Analysis/overlay_results3"
    
    folders = {
        "1d1": {
            "base": os.path.join(ROOT_DIR, "1d1_base"),
            "parts": os.path.join(ROOT_DIR, "1d1_particulates")
        },
        "1d9": {
            "base": os.path.join(ROOT_DIR, "1d9_base"),
            "parts": os.path.join(ROOT_DIR, "1d9_particulates")
        }
    }
    
    all_results = []
    
    for vel_label, paths in folders.items():
        if os.path.exists(paths["base"]) and os.path.exists(paths["parts"]):
            group_results = process_velocity_group(paths["base"], paths["parts"], vel_label)
            all_results.extend(group_results)
        else:
            print(f"\nWarning: Missing folders for {vel_label}. Checked:")
            print(f" - {paths['base']}")
            print(f" - {paths['parts']}")

    if not all_results:
        print("\nNo data processed. Exiting.")
        return
        
    # --- Data Organization & Table Output ---
    df = pd.DataFrame(all_results)
    print("\nFinal Delay Results:")
    print(df.to_string(index=False))
    
    # --- Plotting ---
    # Filter out "No Transmission" rows so matplotlib doesn't crash on text data
    plot_df = df[df["Material Delay (ms)"] != "No Transmission"].copy()
    
    if plot_df.empty:
        print("\nNote: No valid transmission data available to generate a plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = np.arange(len(plot_df))
    
    colors = []
    for _, row in plot_df.iterrows():
        if row['Velocity'] == '1d1':
            colors.append('skyblue' if row['Confinement'] == 'Plastic' else 'steelblue')
        else:
            colors.append('lightcoral' if row['Confinement'] == 'Plastic' else 'firebrick')

    bars = ax.bar(x_positions, plot_df['Material Delay (ms)'], color=colors, edgecolor='black')
    
    ax.set_yscale('log')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Particulate Material Transit Delay (Normalized to Bar-on-Bar)")
    ax.set_ylabel(r"Additional Time Delay $\Delta t$ (ms) [Log Scale]")
    ax.set_xticks(x_positions)
    
    labels = [f"{row['Specimen']}\n({row['Velocity']})" for _, row in plot_df.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='1d1 - Aluminium'),
        Patch(facecolor='skyblue', edgecolor='black', label='1d1 - Plastic'),
        Patch(facecolor='firebrick', edgecolor='black', label='1d9 - Aluminium'),
        Patch(facecolor='lightcoral', edgecolor='black', label='1d9 - Plastic')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()