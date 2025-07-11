import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import image_tools as it


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

def force_tranducer_graph(file):
    time, volt = np.loadtxt(file, delimiter=',', skiprows=3, unpack=True)
    time=time*10e-4
    if time[0] != 0.0: time = time - time[0]
    volt=volt*10e-4
    force_time_total = time[-1]
    return time, volt, force_time_total

def contours_touching(cnta, cntb, img):
    h, w = img.shape[:2]
    mska = np.zeros((h, w), dtype=np.uint8)
    mskb = np.zeros((h, w), dtype=np.uint8)
    cv.drawContours(mska, cnta, -1, 255, thickness=cv.FILLED)
    cv.drawContours(mskb, cntb, -1, 255, thickness=cv.FILLED)
    kernel = np.ones((3,3), dtype=np.uint8)
    dialated = cv.dilate(mska, kernel, iterations=1)
    overlap = cv.bitwise_and(dialated, mskb)
    return bool(np.any(overlap))

def find_impact_frame(file,vidpath):
    time, volt, total_force_time = force_tranducer_graph(file)
    sg_volt=savgol_filter(volt, 100, 2)
    noise_thresh = np.mean(volt[:1000]) + 2*np.std(volt[:1000])

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
    vid_fps, vid_frame_count, vid_total_time = video_properties(vidpath)
    frame_guess = round((imp_time[0]*vid_fps)*(vid_frame_count/(total_force_time*vid_fps)))

    cap = cv.VideoCapture(vidpath)
    if cap.isOpened() == False: raise TypeError('Not correct path to video')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, init_frame = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 507)
    ret, late_frame = cap.read()
    blur = cv.GaussianBlur(cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY), (11, 11), 0)
    _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                #frame_thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, -1)
    init_contour, hierarchy = it.contours(frame_thresh)
    if len(init_contour) == 1:
        init_points = init_contour[0].reshape(-1, 2)
        init_height = init_points.min()
    #else: raise ImportError('First Frame must only contain sample, please crop video again.')
    found = False
    while not found:
        print(frame_guess)
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
                #roi = cv.selectROI('Select ROI',frame, showCrosshair=True, fromCenter= False)
                #cv.destroyWindow('Select ROI')
                #x, y, w, h = roi
                #cropped = frame[y:y+h, x:x+w]
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                norm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
                blur = cv.GaussianBlur(norm, (9, 9), 0)
                _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                contours, hierarchy = it.contours(frame_thresh)
                contour_counts.append(len(contours))
                if len(contours) == 2:
                    drop_weight = contours[1]
                    sample = contours[0]
                    weight_points = drop_weight.reshape(-1, 2)
                    sample_points = sample.reshape(-1, 2)
                    lowest_point_weight = weight_points[:, 1].max()
                    highest_point_sample = sample_points[:, 1].min()
                    print(highest_point_sample, lowest_point_weight)
                    vert_dist = highest_point_sample-lowest_point_weight
                    seed_contours = contours
                    if vert_dist<0:
                        toofar[i] = True
                        contour_dist.append(-3)
                    else: contour_dist.append(vert_dist)
                    if 0<vert_dist<30: close_to_collision = True
                    print(contour_dist)
                elif len(contours)>2:
                    toofar[i] = True
                    contour_dist.append(-3)
                    print(contour_dist)
                elif len(contours) == 1:
                    points = contours[0].reshape(-1, 2)
                    if points[:, 1].min()<11:
                        collision_frame[i] = True
                        contour_dist.append(-2)
                        print(contour_dist)
                    else:
                        toosmall[i] = True
                        contour_dist.append(-1)
                        print(contour_dist)
                '''
                output = frame.copy()
                cv.drawContours(output, contours, -1, (0, 255, 0), 1)
                cv.imshow('contours', output)
                cv.waitKey(0)
                cv.destroyAllWindows()
                '''
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
    return seed_contours, seed_frame, impact_frame_index, impact_frame, late_frame

"""
    Given:
      - seed_frame: a BGR image where your objects (sample, weight, drop) are touching
      - seed_contours: list of contours (np.ndarray of points) detected on seed_frame

    Returns:
      - markers: the labeled marker image after watershed (–1 = boundary)
      - segmented: color‐coded BGR image showing each segment
"""


vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/Footage_00132.cine"
file = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/ForceData_00132.csv"
#crop_video_with_roi(vidpath, "/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/Footage_00051_cropped.mp4")
cropped_vidpath = vidpath #"/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/Footage_00051_cropped.mp4"
seed_contours, late_frame, impact_index, impact_frame, seed_frame = find_impact_frame(file, cropped_vidpath)


sample_contour = seed_contours[0]
sample_points = sample_contour.reshape(-1, 2)
weight_contour = seed_contours[1]
weight_points = weight_contour.reshape(-1, 2)
highest_point_sample = sample_points[:, 1].min()
lowest_point_weight = weight_points[:,1].max()
vert_border = highest_point_sample
horizontal_max = sample_points[:, 0].max()
horizontal_min = sample_points[:, 0].min()
cropped = seed_frame[lowest_point_weight:, horizontal_min:horizontal_max]
grey = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
it.canny_edge(grey)
blur = cv.GaussianBlur(grey, (5, 5), 0)
norm = cv.normalize(blur, None, 0, 255, cv.NORM_MINMAX, dtype = cv.CV_8U)
#ret, thresh = cv.threshold(norm,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
thresh = cv.adaptiveThreshold(norm, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10)
cv.imshow('Threshold OTSU', thresh)
cv.waitKey(0)
cv.destroyAllWindows()
#it.canny_edge(thresh)
#conts, hier = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#output = cropped.copy()
#cv.drawContours(output, conts, -1, (0, 255, 0), 1)
#cv.imshow('Adaptive contours', output)
#cv.waitKey(0)
#cv.destroyAllWindows()
# noise removal
kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 5)
 
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=5)
 
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.4*dist_transform.max(),255,0)
 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(cropped,markers)
cropped[markers == -1] = [255,0,0]
cv.imshow('thresh', cropped)
cv.waitKey(0)
cv.destroyAllWindows()