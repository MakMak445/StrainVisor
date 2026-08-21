"""
Builds a conference-presentation-quality animation of granular material flow
for SHPB tests, by advecting a sparse grid of seed points through the py2DIC
frame-to-frame displacement field (see DIC Results/<test>/*_dx_*.txt and
*_dy_*.txt) and overlaying their paths on the source video frames.

Each dx/dy file pair is the displacement between two *consecutive* sampled
frames (see py2DIC-master/sources/Main.py: start_index = stop_index at the
end of every level), not cumulative from a fixed reference frame. Seed point
positions are therefore updated by interpolating the local displacement field
at each point's *current* (already-moved) position every step and adding it
on - i.e. advection through a time-varying velocity field, not just reading
a fixed grid cell's cumulative offset.

Sign convention: dx and dy are both added as-is (no flip) when advecting pixel
positions here. py2DIC's dy is TP_search_y - TP_temp_y, i.e. positive dy means
the matched patch moved to a LARGER row index = further down in image/pixel
coordinates - and since positions here are plain data coordinates on an axis
where larger y is lower on screen (matching image rows), adding dy directly
already moves points the correct visual direction (confirmed against
run_dual_aoi_dic.py's "under_bar" AOI, which sits *below* the bar, i.e. the
expected direction for material expelled from the gap).

flow_visualiser.py negates dy for ITS quiver display (v_norm = -v_sub), which
looks like a contradiction but isn't: matplotlib's quiver() defaults to
angles='uv', which rotates arrows using standard "+v = up on screen" math
convention regardless of axis orientation - it does NOT account for the
image-style inverted y-axis the way a plain coordinate offset (as used here
via scatter/LineCollection) does. So flow_visualiser's negation is a quiver-
specific correction for that default, not evidence dy's sign needs flipping
in general. Verified empirically by rendering both mechanisms side by side
with known-sign test vectors - do not "fix" this again without doing the same.

AOI: run this locally (needs a display). It loops through every test below,
opening each test's first frame in a window in turn - click and drag to draw
a box around the granular material region, then CLOSE the window to move on
to the next test. The grid is mapped across that box instead of the full
frame, same as flow_visualiser.py's get_aoi_visually().
"""
import os
import re

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator

matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

VIDEO_ANALYSIS_DIR = "src/split-hopkinson_bar/Video_Analysis"

# (DIC Results subfolder, Camera Njord source subfolder) for every test to render.
TESTS = [
    ("Njord_10_27", "Njord_10_27_19"),
    ("Njord_10_39", "Njord_10_39_15"),
    ("Njord_10_48", "Njord_10_48_18"),
    ("Njord_10_59", "Njord_10_59_53"),
    ("Njord_11_09", "Njord_11_09_09"),
    ("Njord_11_17", "Njord_11_17_04"),
    ("Njord_9_37", "Njord_09_37_36"),
    ("Njord 9_58", "Njord_09_58_25"),
]

N_POINTS_ROWS = 14
N_POINTS_COLS = 10
TRAIL_LENGTH = 8
FPS = 6
COL_INSET_FRAC = 0.15  # keep seed columns within [15%, 85%] of the AOI, away from the noisy bar edge


def get_aoi_visually(image_path):
    print(f"Loading reference image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at {image_path}")

    plt.switch_backend('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title("Draw AOI around the granular material (Click & Drag). CLOSE window when done.")

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
    plt.show()

    if not roi_coords:
        print("Warning: No AOI drawn. Defaulting to full image.")
        return [0, 0, img.shape[1], img.shape[0]]

    return roi_coords


def load_grid_sequence(dic_dir):
    dx_files = sorted(
        (f for f in os.listdir(dic_dir) if "_dx_" in f and f.endswith(".txt")),
        key=lambda x: int(re.search(r'_dx_\d+_(\d+)\.txt', x).group(1)),
    )
    frame_indices, dx_stack, dy_stack = [], [], []
    for dx_file in dx_files:
        frame_idx = int(re.search(r'_dx_\d+_(\d+)\.txt', dx_file).group(1))
        dy_file = dx_file.replace('_dx_', '_dy_')
        u = pd.read_csv(os.path.join(dic_dir, dx_file), sep=r'\s+', header=None).values
        v = pd.read_csv(os.path.join(dic_dir, dy_file), sep=r'\s+', header=None).values
        dx_stack.append(u)
        dy_stack.append(v)
        frame_indices.append(frame_idx)
    return frame_indices, np.array(dx_stack), np.array(dy_stack)


def process_test(test_name, img_subdir):
    dic_dir = f"DIC Results/{test_name}"
    img_dir = f"{VIDEO_ANALYSIS_DIR}/Camera Njord/{img_subdir}"
    out_path = f"{VIDEO_ANALYSIS_DIR}/{test_name.replace(' ', '_')}_flow_animation.mp4"

    frame_indices, dx_all, dy_all = load_grid_sequence(dic_dir)
    n_steps, rows, cols = dx_all.shape

    all_images = sorted(f for f in os.listdir(img_dir) if f.endswith(".tiff"))
    ref_img = cv2.imread(os.path.join(img_dir, all_images[0]), cv2.IMREAD_GRAYSCALE)

    x_start, y_start, x_end, y_end = get_aoi_visually(os.path.join(img_dir, all_images[0]))
    plt.switch_backend('Agg')  # headless from here on, for fast frame rendering

    # Grid points map linearly across the AOI you drew, not the full frame.
    x_lin = np.linspace(x_start, x_end, cols)
    y_lin = np.linspace(y_start, y_end, rows)

    # Inset columns only - the AOI's left/right edges sit on the low-texture bar
    # surface where DIC decorrelates and gives erratic, not real, motion. Rows span
    # the full AOI height since the granular material fills it vertically.
    col_lo, col_hi = COL_INSET_FRAC * (cols - 1), (1 - COL_INSET_FRAC) * (cols - 1)
    seed_row_idx = np.linspace(0, rows - 1, N_POINTS_ROWS).round().astype(int)
    seed_col_idx = np.linspace(col_lo, col_hi, N_POINTS_COLS).round().astype(int)
    seed_rows, seed_cols = np.meshgrid(seed_row_idx, seed_col_idx, indexing='ij')
    seed_rows, seed_cols = seed_rows.ravel(), seed_cols.ravel()
    n_points = seed_rows.size

    positions = np.stack([x_lin[seed_cols], y_lin[seed_rows]], axis=1).astype(float)
    trajectories = np.zeros((n_points, n_steps + 1, 2))
    trajectories[:, 0, :] = positions

    for t in range(n_steps):
        interp_dx = RegularGridInterpolator((y_lin, x_lin), dx_all[t], bounds_error=False, fill_value=0.0)
        interp_dy = RegularGridInterpolator((y_lin, x_lin), dy_all[t], bounds_error=False, fill_value=0.0)
        query = positions[:, [1, 0]]  # (y, x) order
        local_dx = interp_dx(query)
        local_dy = interp_dy(query)
        positions[:, 0] += local_dx
        positions[:, 1] += local_dy  # unflipped: positive dy = further down in image coords
        trajectories[:, t + 1, :] = positions

    display_frame_indices = [0] + frame_indices

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, n_points))

    im_artist = ax.imshow(ref_img, cmap='gray', vmin=0, vmax=255)
    scat = ax.scatter(
        trajectories[:, 0, 0], trajectories[:, 0, 1],
        c=colors, s=22, edgecolors='white', linewidths=0.4, zorder=3,
    )
    trail_collections = [
        LineCollection([], linewidths=1.2, zorder=2) for _ in range(n_points)
    ]
    for lc in trail_collections:
        ax.add_collection(lc)

    # Zoom to the AOI (plus a small margin) instead of the full frame, since the
    # real displacement here is only a handful of pixels - invisible at full-frame scale.
    pad_x = 0.05 * (x_end - x_start)
    pad_y = 0.05 * (y_end - y_start)
    ax.set_xlim(x_start - pad_x, x_end + pad_x)
    ax.set_ylim(y_end + pad_y, y_start - pad_y)
    ax.axis('off')
    title = ax.set_title("", fontsize=13, color='white', backgroundcolor='black')
    fig.patch.set_facecolor('black')

    def update(frame_num):
        img_idx = display_frame_indices[frame_num]
        img = cv2.imread(os.path.join(img_dir, all_images[img_idx]), cv2.IMREAD_GRAYSCALE)
        im_artist.set_data(img)

        pts = trajectories[:, frame_num, :]
        scat.set_offsets(pts)

        trail_start = max(0, frame_num - TRAIL_LENGTH)
        for i, lc in enumerate(trail_collections):
            seg_pts = trajectories[i, trail_start:frame_num + 1, :]
            if len(seg_pts) >= 2:
                segments = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
                n_seg = segments.shape[0]
                alphas = np.linspace(0.05, 0.9, n_seg)
                rgba = np.tile(colors[i], (n_seg, 1))
                rgba[:, 3] = alphas
                lc.set_segments(segments)
                lc.set_color(rgba)
            else:
                lc.set_segments([])

        title.set_text(f"Granular Flow — {test_name}   |   {all_images[img_idx]}")
        return [im_artist, scat, title, *trail_collections]

    n_frames = len(display_frame_indices)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = animation.FFMpegWriter(fps=FPS, bitrate=6000)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Saved animation to {out_path}")


def main():
    for test_name, img_subdir in TESTS:
        print(f"\n=== {test_name} ===")
        process_test(test_name, img_subdir)


if __name__ == "__main__":
    main()
