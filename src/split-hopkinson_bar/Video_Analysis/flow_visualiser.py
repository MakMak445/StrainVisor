import matplotlib
matplotlib.use('TkAgg') # Bypasses Qt entirely, preventing all conflicts
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pandas as pd
import numpy as np
import os
import re
import cv2

def get_aoi_visually(image_path):
    print(f"Loading reference image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Failed to load image at {image_path}")
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title("Draw AOI (Click & Drag). CLOSE window when done.")
    
    roi_coords = []
    
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        x_start, x_end = min(x1, x2), max(x1, x2)
        y_start, y_end = min(y1, y2), max(y1, y2)
        
        roi_coords.clear()
        roi_coords.extend([x_start, y_start, x_end, y_end])
        print(f"Selected -> X: {x_start}-{x_end}, Y: {y_start}-{y_end}")

    rs = RectangleSelector(ax, onselect, useblit=True,
                           button=[1], minspanx=5, minspany=5,
                           spancoords='pixels', interactive=True)
    
    print("Window opened: Draw your box, then CLOSE the window to continue.")
    plt.show() # Blocks execution until you close the window
    
    if not roi_coords:
        print("Warning: No AOI drawn. Defaulting to full image.")
        return [0, 0, img.shape[1], img.shape[0]]
        
    return roi_coords

def overlay_flows(plot_folder, image_folder, step=10, arrow_scale=35):
    dx_files = sorted(
        [f for f in os.listdir(plot_folder) if "_dx_" in f and f.endswith(".txt")],
        key=lambda x: [int(c) for c in re.findall(r'\d+', x.split('_dx_')[1])]
    )
    all_images = sorted([f for f in os.listdir(image_folder) if f.endswith(".tiff")])
    
    if not dx_files or not all_images:
        print("Error: Missing displacement text files or .tiff images.")
        return

    first_img_path = os.path.join(image_folder, all_images[0])
    aoi_coords = get_aoi_visually(first_img_path)
    x_start, y_start, x_end, y_end = aoi_coords
    
    for dx_file in dx_files:
        match = re.search(r'_dx_\d+_(\d+)\.txt', dx_file)
        if not match: continue
            
        current_frame_idx = int(match.group(1))
        
        if current_frame_idx >= len(all_images):
            continue
            
        img_name = all_images[current_frame_idx]
        img = cv2.imread(os.path.join(image_folder, img_name), cv2.IMREAD_GRAYSCALE)
        
        dy_file = dx_file.replace('_dx_', '_dy_')
        u = pd.read_csv(os.path.join(plot_folder, dx_file), sep=r'\s+', header=None).values
        v = pd.read_csv(os.path.join(plot_folder, dy_file), sep=r'\s+', header=None).values
        
        rows, cols = u.shape
        
        # Stretch data accurately across the chosen box
        x_lin = np.linspace(x_start, x_end, cols)
        y_lin = np.linspace(y_start, y_end, rows)
        x_full, y_full = np.meshgrid(x_lin, y_lin)
        
        x_sub, y_sub = x_full[::step, ::step], y_full[::step, ::step]
        u_sub, v_sub = u[::step, ::step], v[::step, ::step]
        
        magnitude = np.sqrt(u_sub**2 + v_sub**2)
        u_norm, v_norm = np.zeros_like(u_sub), np.zeros_like(v_sub)
        
        valid = magnitude > 0
        u_norm[valid] = u_sub[valid] / magnitude[valid]
        v_norm[valid] = -v_sub[valid] / magnitude[valid] 
        
        # Switch to Agg backend for fast, headless saving of the output plots
        plt.switch_backend('Agg')
        plt.figure(figsize=(12, 8))
        plt.imshow(img, cmap='gray')
        
        quiv = plt.quiver(x_sub, y_sub, 
                          u_norm, v_norm, magnitude, 
                          cmap='jet', scale=arrow_scale, pivot='mid', headwidth=4)
        
        cbar = plt.colorbar(quiv, fraction=0.046, pad=0.04)
        cbar.set_label('Displacement Magnitude (pixels)', rotation=270, labelpad=15)
        
        plt.title(f"Flow Overlay Field: {img_name}")
        plt.axis('off')
        
        save_path = os.path.join(plot_folder, f"overlay_{current_frame_idx:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Processed and saved: overlay_{current_frame_idx:03d}.png")

if __name__ == "__main__":
    PLOT_DIR = 'OutputPlots'
    IMG_DIR = ''
    
    overlay_flows(plot_folder=PLOT_DIR, image_folder=IMG_DIR, step=10, arrow_scale=35)