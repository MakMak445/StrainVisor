import os
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Strictly headless backend for fast, crash-free rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_grid_dimensions(aoi_coords, target_arrows):
    """
    Mathematically calculates the ideal number of rows and columns to 
    evenly distribute a specific number of arrows across the AOI's dimensions.
    """
    width = aoi_coords[2] - aoi_coords[0]
    height = aoi_coords[3] - aoi_coords[1]
    
    area = width * height
    if area <= 0:
        return 1, 1
        
    density = target_arrows / area
    step = 1 / np.sqrt(density)
    
    cols = max(1, int(round(width / step)))
    rows = max(1, int(round(height / step)))
    
    return cols, rows

def process_vectors(dx_path, dy_path, aoi, target_arrows):
    """
    Loads raw DIC text files, resizes them to the target arrow count,
    maps them to the true AOI physical coordinates, and calculates magnitude.
    """
    if not os.path.exists(dx_path) or not os.path.exists(dy_path):
        return None
        
    u = pd.read_csv(dx_path, sep=r'\s+', header=None).values
    v = pd.read_csv(dy_path, sep=r'\s+', header=None).values
    
    cols, rows = get_grid_dimensions(aoi, target_arrows)
    
    # Resize the raw DIC data to match our mathematically perfect grid
    u_resized = cv2.resize(u.astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v.astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    
    # Map the coordinates directly to the exact pixel bounds of the AOI
    x_lin = np.linspace(aoi[0], aoi[2], cols)
    y_lin = np.linspace(aoi[1], aoi[3], rows)
    X, Y = np.meshgrid(x_lin, y_lin)
    
    # Calculate magnitude and normalize vectors to length 1
    mag = np.sqrt(u_resized**2 + v_resized**2)
    valid = mag > 0
    
    u_norm = np.zeros_like(u_resized)
    v_norm = np.zeros_like(v_resized)
    
    u_norm[valid] = u_resized[valid] / mag[valid]
    v_norm[valid] = -v_resized[valid] / mag[valid] # Flip Y for correct matplotlib visual
    
    return X, Y, u_norm, v_norm, mag

def render_dual_aoi_flows(dic_folder_1, dic_folder_2, image_folder, output_folder, 
                          aoi_1, aoi_2, arrows_1=400, arrows_2=150, arrow_scale=35):
                              
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Find all available frames in Folder 1
    files_1 = [f for f in os.listdir(dic_folder_1) if "_dx_" in f and f.endswith(".txt")]
    frame_indices = []
    
    for f in files_1:
        match = re.search(r'_dx_\d+_(\d+)\.txt', f)
        if match:
            frame_indices.append(int(match.group(1)))
            
    frame_indices = sorted(frame_indices)
    all_images = sorted([f for f in os.listdir(image_folder) if f.endswith(".tiff")])
    
    print(f"Found {len(frame_indices)} frames to process.")

    for frame_idx in frame_indices:
        if frame_idx >= len(all_images):
            continue
            
        # Reconstruct file names
        # Adjust the prefix matching logic if your files have a different naming structure
        prefix_1 = files_1[0].split('_dx_')[0] 
        
        dx_1 = os.path.join(dic_folder_1, f"{prefix_1}_dx_0_{frame_idx}.txt")
        dy_1 = os.path.join(dic_folder_1, f"{prefix_1}_dy_0_{frame_idx}.txt")
        
        # Assume Folder 2 uses the exact same frame index numbering
        # Find the prefix for Folder 2 dynamically
        try:
            sample_file_2 = [f for f in os.listdir(dic_folder_2) if "_dx_" in f][0]
            prefix_2 = sample_file_2.split('_dx_')[0]
        except IndexError:
            print("Error: Could not find valid files in DIC Folder 2.")
            return

        dx_2 = os.path.join(dic_folder_2, f"{prefix_2}_dx_0_{frame_idx}.txt")
        dy_2 = os.path.join(dic_folder_2, f"{prefix_2}_dy_0_{frame_idx}.txt")
        
        data_1 = process_vectors(dx_1, dy_1, aoi_1, arrows_1)
        data_2 = process_vectors(dx_2, dy_2, aoi_2, arrows_2)
        
        if not data_1 or not data_2:
            print(f"Skipping frame {frame_idx}: Missing text files in one or both folders.")
            continue
            
        X1, Y1, u1, v1, mag1 = data_1
        X2, Y2, u2, v2, mag2 = data_2
        
        # Calculate the absolute max displacement across BOTH regions for unified coloring
        global_max = max(np.max(mag1), np.max(mag2))
        if global_max == 0: global_max = 0.1 # Prevent divide by zero
        
        # Set up a unified color normalizer
        norm = mcolors.Normalize(vmin=0, vmax=global_max)
        
        img_name = all_images[frame_idx]
        img = cv2.imread(os.path.join(image_folder, img_name), cv2.IMREAD_GRAYSCALE)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img, cmap='gray')
        
        # Plot Region 1
        quiv1 = plt.quiver(X1, Y1, u1, v1, mag1, 
                           cmap='jet', norm=norm, scale=arrow_scale, pivot='mid', headwidth=4)
        
        # Plot Region 2 (Uses the exact same colormap and normalizer)
        quiv2 = plt.quiver(X2, Y2, u2, v2, mag2, 
                           cmap='jet', norm=norm, scale=arrow_scale, pivot='mid', headwidth=4)
        
        # Because both use the same norm, one colorbar represents both regions perfectly
        cbar = plt.colorbar(quiv1, fraction=0.046, pad=0.04)
        cbar.set_label('Displacement Magnitude (pixels)', rotation=270, labelpad=15)
        
        plt.title(f"Dual-AOI Flow Field: {img_name}")
        plt.axis('off')
        
        save_path = os.path.join(output_folder, f"dual_overlay_{frame_idx:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Processed and saved: dual_overlay_{frame_idx:03d}.png")

if __name__ == "__main__":
    
    # --- 1. DIRECTORY SETUP ---
    # Point these to your two separate DIC output folders
    DIC_FOLDER_REGION_1 = '/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Video_Analysis/OutputPlotsA' 
    DIC_FOLDER_REGION_2 = '/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Video_Analysis/OutputPlotsB'
    
    # Point this to your raw TIFF images
    IMAGE_FOLDER = '/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Video_Analysis/Camera Njord/Njord_10_17_35'
    
    # Point this to where you want the final PNG sequence saved
    OUTPUT_FOLDER = '/home/MakMak445/projects/StrainVisor/src/split-hopkinson_bar/Video_Analysis/CombinedPlots'
    
    # --- 2. AOI BOUNDING BOXES ---
    # Format: [X_start, Y_start, X_end, Y_end]
    AOI_REGION_1 = [85, 0, 206, 203] # Example for vertical bar
    AOI_REGION_2 = [2, 203, 396, 228]  # Your 25-pixel tall horizontal strip
    
    # --- 3. ARROW COUNTS ---
    # Exact number of arrows to generate inside the boxes above
    TARGET_ARROWS_REGION_1 = 40 
    TARGET_ARROWS_REGION_2 = 20
    
    # --- RUN THE VISUALIZER ---
    render_dual_aoi_flows(
        dic_folder_1=DIC_FOLDER_REGION_1,
        dic_folder_2=DIC_FOLDER_REGION_2,
        image_folder=IMAGE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        aoi_1=AOI_REGION_1,
        aoi_2=AOI_REGION_2,
        arrows_1=TARGET_ARROWS_REGION_1,
        arrows_2=TARGET_ARROWS_REGION_2,
        arrow_scale=35
    )