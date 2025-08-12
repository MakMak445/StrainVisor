import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import savgol_filter, find_peaks, welch, peak_widths
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import heapq
from statsmodels import robust

vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/Footage_00065.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/Footage_00132.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/Footage_00065.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-015/Data/Actions/DropWeight/Task_0059/Footage_00084.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-019/Data/Actions/DropWeight/Task_0066/Footage_00082.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-021/Data/Actions/DropWeight/Task_0043/Footage_00098.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-029/Data/Actions/DropWeight/Task_0038/Footage_00173.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-031/Data/Actions/DropWeight/Task_0029/Footage_00139.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/Footage_00051.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-051/Data/Actions/DropWeight/Task_0002/Footage_00076.cine"
#vidpath = ""
#vidpath = ""

file = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/ForceData_00065.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/ForceData_00132.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/ForceData_00065.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-015/Data/Actions/DropWeight/Task_0059/ForceData_00084.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-019/Data/Actions/DropWeight/Task_0066/ForceData_00082.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-021/Data/Actions/DropWeight/Task_0043/ForceData_00098.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-029/Data/Actions/DropWeight/Task_0038/ForceData_00173.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-031/Data/Actions/DropWeight/Task_0029/ForceData_00139.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/ForceData_00051.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-051/Data/Actions/DropWeight/Task_0002/ForceData_00076.csv"
#file = ""
#file = ""

def video_properties(path: str):
    cap=cv.VideoCapture(path)
    if cap.isOpened() == False: raise TypeError('Not correct path to video')
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    cap.release()
    cv.destroyAllWindows()
    total_time = frame_count / fps
    return fps, frame_count, total_time

def crop_video_with_roi(input_path: str, output_path: str, codec: str = 'mp4v') -> None:
    """
    Crop all frames of a video using an interactively selected ROI from the first frame.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the cropped video will be saved.
        codec (str): FourCC codec string for the output video (default 'mp4v').

    Usage:
        crop_video_with_roi('input.mp4', 'cropped_output.mp4')
    """
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame from video.")

    roi = cv.selectROI('Select ROI', frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, roi)
    cv.destroyWindow('Select ROI')

    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*codec)
    out = cv.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot open video writer for file: {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y:y+h, x:x+w]
        out.write(cropped)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Cropped video saved to {output_path}, processed {frame_count} frames.")

def force_tranducer_graph(time, volt):
    time_np=np.array(time).astype(float)
    time=time_np*10e-4
    if time[0] != 0.0: time = time - time[0]
    volt_np=np.array(volt).astype(float)
    volt=volt_np*10e-4
    force_time_total = time[-1]
    return time, volt, force_time_total


def find_impact_frame(time, volt, vidpath):
    time, volt, total_force_time = force_tranducer_graph(time, volt)
    sg_volt=savgol_filter(volt, 100, 2)
    base = sg_volt[:int(0.1*len(sg_volt))]
    mu_std = np.mean(base)
    sigma_std = robust.mad(base)
    noise_thresh = mu_std + 2*sigma_std

    imp_time = []  
    imp_volt = []
    noise_time = []
    noise_volt = []
    for t, v in zip(time, sg_volt):
        if v<noise_thresh:
            noise_time.append(t)
            noise_volt.append(v)
        else:
            imp_time.append(t)
            imp_volt.append(v)
    #plt.plot(noise_time, noise_volt, color='red')
    #plt.plot(imp_time, imp_volt, color='green')
    #plt.show()
    vid_fps, vid_frame_count, vid_total_time = video_properties(vidpath)
    frame_guess = round((imp_time[0]*vid_fps)*(vid_frame_count/(total_force_time*vid_fps)))

    cap = cv.VideoCapture(vidpath)
    if cap.isOpened() == False: raise TypeError('Not correct path to video')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, init_frame = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 538)
    ret = cap.read()
    blur = cv.GaussianBlur(cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY), (11, 11), 0)

    _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    init_contour, hierarchy = cv.findContours(frame_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)                       
    if len(init_contour) == 1:
        init_points = init_contour[0].reshape(-1, 2)
        init_height = init_points[:, 1].max() - init_points[:, 1].min()
    found = False
    while not found:
        #print(frame_guess)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_guess-1)
        ret, frame_prev = cap.read()
        ret, frame_current = cap.read()
        ret, frame_next = cap.read()
        frames = [frame_prev, frame_current, frame_next]
        contour_counts = []
        contour_dist = []
        toofar = [False, False, False]
        toosmall = [False, False, False]
        collision_frame = [False, False, False]
        close_to_collision = False
        for i, frame in enumerate(frames):
            if ret == True:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                norm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
                blur = cv.GaussianBlur(gray, (5, 5), 0)
                _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                contours, _ = cv.findContours(frame_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # Minimum area threshold — tweak this value!
                min_area = 200  
                # Filtered list of contours
                contours = [cnt for cnt in contours if cv.contourArea(cnt) >= min_area]
                contour_counts.append(len(contours))
                if len(contours) == 2:
                    drop_weight = contours[1]
                    sample = contours[0]
                    weight_points = drop_weight.reshape(-1, 2)
                    sample_points = sample.reshape(-1, 2)
                    lowest_point_weight = weight_points[:, 1].max()
                    highest_point_sample = sample_points[:, 1].min()
                    #print(highest_point_sample, lowest_point_weight)
                    vert_dist = highest_point_sample-lowest_point_weight
                    seed_contours = contours
                    if vert_dist<0:
                        toofar[i] = True
                        contour_dist.append(-3)
                    else: contour_dist.append(vert_dist)
                    if 0<vert_dist<30: close_to_collision = True
                    #print(contour_dist)
                elif len(contours)>2:
                    toofar[i] = True
                    contour_dist.append(-3)
                    #print(contour_dist)
                elif len(contours) == 1:
                    points = contours[0].reshape(-1, 2)
                    if points[:, 1].min()<11:
                        collision_frame[i] = True
                        contour_dist.append(-2)
                        #print(contour_dist)
                    else:
                        toosmall[i] = True
                        contour_dist.append(-1)
                        #print(contour_dist)
                '''
                output = frame.copy()
                cv.drawContours(output, contours, -1, (0, 255, 0), 1)
                cv.imshow('contours', output)
                cv.waitKey(0)
                cv.destroyAllWindows()
                '''
        #print(f"initial height is {init_height}")
        if collision_frame[2] and collision_frame[1] and not (collision_frame[0]):
            found = True
            impact_frame_index = frame_guess
            seed_frame = frames[0]
            print(f"first impact frame found. Frame {impact_frame_index}")
            impact_frame = frames[1]
            continue

        if np.any(toosmall):
            frame_guess+=30
            continue
        if np.any(toofar):
            frame_guess-=30
        if collision_frame[0] == True:
            frame_guess-=5
            continue
        if close_to_collision:
            frame_guess+=1
            continue
        if np.all(np.array(contour_dist)>0):
            frame_guess_extrap=frame_guess+round(((contour_dist[-1] / np.mean([contour_dist[1]-contour_dist[2], contour_dist[0]-contour_dist[1]]))-1))
            if frame_guess_extrap==frame_guess:
                frame_guess+=1
            else: frame_guess = frame_guess_extrap
    return seed_contours, seed_frame, impact_frame_index, init_height, vid_fps

"""
    Given:
      - seed_frame: a BGR image where your objects (sample, weight, drop) are touching
      - seed_contours: list of contours (np.ndarray of points) detected on seed_frame

    Returns:
      - markers: the labeled marker image after watershed (–1 = boundary)
      - segmented: color‐coded BGR image showing each segment
"""
def obtain_base_index(init_markers):
    obtained = False
    index = 2
    h, w = init_markers.shape
    col_num = w//2
    while not obtained:
        if init_markers[h-index, col_num]!=init_markers[h-index-1, col_num]:
            obtained = True
            return (h-index), index
        else: index+=1
    
def obtain_height_from_markers(markers, base_index, init_height, prev_height):
    #prev_index = 0
    bad_point=False
    obtained = False
    index = 0
    h, w = markers.shape
    prev_index = base_index - prev_height - 5
    #print(prev_height, prev_index)
    col_num = w//2
    while not obtained:
        if (markers[prev_index+index ,col_num] == -1):
            obtained = True
            height = height = base_index - index - prev_index
            if abs((height-prev_height)/prev_height)>=0.2:
                bad_point=True
                return None, bad_point
            else: bad_point = False
            if (base_index - index - prev_index)<=init_height:
                return height, bad_point
            else: return init_height, bad_point
        else: 
            index += 1

def obtain_crop_dimensions(seed_frame, seed_contours):
    width, height =seed_frame.shape[:2]
    sample_contour = seed_contours[0]
    sample_points = np.array(sample_contour.reshape(-1, 2))
    weight_contour = seed_contours[1]
    weight_points = weight_contour.reshape(-1, 2)
    highest_point_sample = sample_points[:, 1].min()
    lowest_point_weight = weight_points[:,1].max()
    vert_border = highest_point_sample
    hori_cent_disp = sample_points - width//2
    horizontal_max = hori_cent_disp.min()+width
    horizontal_max = sample_points[:, 0].max()
    horizontal_min = sample_points[:, 0].min()
    return horizontal_min, horizontal_max, round(lowest_point_weight*0.95)

def obtain_markers(frame, horiz_min, horiz_max, vert_min):
    cropped = frame[vert_min:, horiz_min:horiz_max]
    grey = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (7,7), 0)
    norm = cv.normalize(blur, None, 0, 255, cv.NORM_MINMAX)
    thresh = cv.adaptiveThreshold(norm, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10)
    #cv.imshow('Threshold OTSU', thresh)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 5)
    
    sure_bg = cv.dilate(opening,kernel,iterations=5)
    
    dist_transform, labels = cv.distanceTransformWithLabels(opening,cv.DIST_L2,cv.DIST_MASK_PRECISE, cv.DIST_LABEL_CCOMP)
    ret, sure_fg = cv.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    disp = cv.convertScaleAbs(markers, alpha=255.0/markers.max())
    #cv.imshow('markers', disp)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv.watershed(cropped,markers)
    cropped[markers == -1] = [255,0,0]
    cropped[markers == 1] = [0, 255, 0]
    cropped[markers == 2] = [0, 0, 255]
    cropped[markers == 3] = [0, 255, 255]
    cropped[markers == 4] = [255, 255, 0]
    cropped[markers == 5] = [255, 0, 255]
    cropped[markers == 6] = [255, 102, 102]
    cropped[markers == 7] = [102, 102, 255]
    #cv.imshow('thresh', cropped)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return markers

def generate_strain_graph(time, volt, vidpath):
    seed_contours, seed_frame, impact_index, init_height, fps = find_impact_frame(time, volt, vidpath)
    hormin, hormax, vermin = obtain_crop_dimensions(seed_frame, seed_contours)
    cap = cv.VideoCapture(vidpath)
    tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, impact_index-1)
    heights = []
    base_initialisation = False
    frames = []
    consect_bad = 0
    for i in range(impact_index, tot_frames):
        ret, frame = cap.read()
        if ret:
            markers = obtain_markers(frame, hormin, hormax, vermin)
            if base_initialisation == False:
                base_index, base_height = obtain_base_index(markers) 
                base_initialisation = True 
                init_height = init_height-base_height
            if heights:
                #print(f"init height is {init_height}")
                height, bad_point = obtain_height_from_markers(markers, base_index, init_height, heights[-1])
            else: height, bad_point = obtain_height_from_markers(markers, base_index, init_height, init_height)
            if height and not bad_point: 
                heights.append(height)
                frames.append(i)
                consect_bad = 0
            elif bad_point:
                consect_bad += 1
                if consect_bad==10:
                    break
        else: continue
    time = np.array(frames)/fps
    return time, (heights-init_height)/init_height

def first_contact_auto(
    t, y,
    baseline_frac=0.10,          # fraction of the start used as baseline
    min_consec=20,               # required consecutive samples above High
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
    sigma = 1.4826 * robust.mad(base) + 1e-12  # robust σ

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

    # Fallback: if even huge k_hi still has runs (super no-bisy baseline),
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

    info = {
        "mu": mu, "sigma": sigma,
        "slope_sigma": slope_sigma, "slope_thr": slope_thr,
        "k_hi": float(chosen_k_hi), "k_lo_eff": float((low_thr - mu) / sigma),
        "low_thr": float(low_thr), "high_thr": float(high_thr),
        "baseline_len": int(n0), "min_consec": int(min_consec), 
        "first_index": i
    }
    return t_cross, info
    
stress_time, volt = np.loadtxt(file, delimiter= ',',skiprows=2, unpack=True)
noise_thresh = np.mean(volt[:1000]) + 2*np.std(volt[:1000])
stress_time = stress_time*10e-3

force_peaks, properties = find_peaks(volt, prominence=(None, None), width=(None, None), plateau_size=(None, None))
max_prom_force_idx = np.argmax(properties["prominences"])
force_peak_max_idx = force_peaks[max_prom_force_idx]
peak_width = properties["widths"][max_prom_force_idx]
window_length = peak_width//3
if window_length % 2 == 0:
    window_length += 1
smooth_volt = savgol_filter(volt, 50, 3)

# --- 1) Adaptive prominence threshold from baseline noise ---
baseline = smooth_volt[:int(0.1 * len(smooth_volt))]
noise_std = np.std(smooth_volt)
min_prom = noise_std # 3–5× is typical


smooth_stress_peaks, properties = find_peaks(smooth_volt, prominence=(None, None), width=(None, None), plateau_size=True)
top_n = 3
stress_top_10_prom = np.argsort(properties["prominences"])[-top_n:]
filtered_peaks = smooth_stress_peaks[stress_top_10_prom]
mu, sigma = np.mean(baseline), np.std(baseline)

# Hysteresis thresholds: low = "baseline", high = "definitely above baseline"
k_lo, k_hi = 1.0, 4.0     # tune: 1.5–3.0 and 3.0–6.0 are typical
low_thr  = mu + k_lo * sigma
high_thr = mu + k_hi * sigma
strain_time, strain_unsmooth = generate_strain_graph(stress_time, volt, vidpath)
strain = median_filter(strain_unsmooth, 3)
strain_peaks, properties = find_peaks(-1*strain, prominence=(None, None), width=(None, None), plateau_size=True) #-1 factor because peaks is valley
max_prom_strain_idx = np.argmax((properties["plateau_sizes"]+properties["prominences"])/2)
strain_peak_max_idx = properties["left_edges"][max_prom_strain_idx]
fig = plt.figure(figsize=(12, 6))
plt.plot(strain_time, strain, '.', label = 'Data')
for peak in strain_peaks: plt.plot(strain_time[peak], strain[peak], 'X', markersize=10, color='red', label='peak')
plt.plot(strain_time[strain_peak_max_idx], strain[strain_peak_max_idx], 'X', markersize = 15, color='green')
plt.legend()
plt.show()
for i, t in enumerate(strain):
    if t != 0:
        strain_start = i
        break
    elif i == len(strain):
        raise ValueError("No non zero strain in video detected")
strain_time_cropped = strain_time[strain_start:strain_peak_max_idx+1]
collision_time = strain_time_cropped[-1]-strain_time_cropped[0]
print(collision_time)
strain_cropped = strain[strain_start:strain_peak_max_idx+1]
#plt.plot(strain_time_cropped, strain_cropped, 'o')
#plt.show()
#strain_period = strain_time_cropped[-1] - strain_time_cropped[0]
#uphill_times = abs(uphill_times - strain_period)
#stress_peak_idx = filtered_peaks[np.argmin(uphill_times)]
#plt.plot(stress_time, smooth_volt, '.', color='blue')
#plt.plot(stress_time[stress_peak_idx], smooth_volt[stress_peak_idx], 'X', color='green', markersize=10)
#plt.show()
t_contact, info = first_contact_auto(stress_time, volt)
print(f"Contact @ {t_contact:.6f}s; k_hi={info['k_hi']:.2f}, "
      f"k_lo≈{info['k_lo_eff']:.2f}, σ={info['sigma']:.3g}")
first_stress_index = info["first_index"]

strain_time_cropped = strain_time_cropped - strain_time_cropped[0]
stress_time_synced = stress_time - t_contact
stress_synced = np.interp(strain_time_cropped, stress_time_synced, smooth_volt, left = np.nan, right = np.nan)

plt.plot(strain_cropped, stress_synced, 'o')
plt.xlabel('Strain')
plt.ylabel('Stress Density(mv)')
plt.title('Stress-strain curve')
plt.show()