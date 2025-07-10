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
    cap.set(cv.CAP_PROP_POS_FRAMES, 1)
    ret, init_frame = cap.read()
    blur = cv.GaussianBlur(cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY), (9, 9), 0)
    _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                #frame_thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, -1)
    init_contour, hierarchy = it.contours(frame_thresh)
    if len(init_contour) == 1:
        init_points = init_contour[0].reshape(-1, 2)
        init_height = init_points.min()
    else: raise ImportError('First Frame must only contain sample, please crop video again.')

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
    
        if collision_frame[2] and collision_frame[1] and not (collision_frame[0]):
            found = True
            impact_frame_index = frame_guess
            seed_frame = frames[0]
            print(f"first impact frame found. Frame {impact_frame_index}")
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
    return seed_contours, seed_frame, impact_frame_index