import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import savgol_filter

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
    cap.set(cv.CAP_PROP_POS_FRAMES, 538)
    ret, late_frame = cap.read()
    blur = cv.GaussianBlur(cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY), (11, 11), 0)
    _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                #frame_thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, -1)
    init_contour, hierarchy = cv.findContours(frame_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)                       
    if len(init_contour) == 1:
        init_points = init_contour[0].reshape(-1, 2)
        init_height = init_points[:, 1].max() - init_points[:, 1].min()
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
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                norm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
                blur = cv.GaussianBlur(norm, (9, 9), 0)
                _, frame_thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
                contours, hierarchy = cv.findContours(frame_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        print(f"initial height is {init_height}")
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
    return seed_contours, seed_frame, impact_frame_index, init_height

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
    
def obtain_height_from_markers(markers, base_index, init_height):
    obtained = False
    index = 2
    h, w = markers.shape
    col_num = w//2
    while not obtained:
        if (markers[index ,col_num] != markers[index - 1, col_num]):
            obtained = True
            if (base_index-index)<=init_height:
                return base_index - index
            else: return init_height 
        else: 
            index += 1
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/Footage_00065.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/Footage_00132.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/Footage_00065.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-015/Data/Actions/DropWeight/Task_0059/Footage_00084.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-019/Data/Actions/DropWeight/Task_0066/Footage_00082.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-021/Data/Actions/DropWeight/Task_0043/Footage_00098.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-029/Data/Actions/DropWeight/Task_0038/Footage_00173.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-031/Data/Actions/DropWeight/Task_0029/Footage_00139.cine"
#vidpath = "/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/Footage_00051.cine"
#vidpath = ""
#vidpath = ""
#vidpath = ""

#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/ForceData_00065.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-002/Data/Actions/DropWeight/Task_0092/ForceData_00132.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-005/Data/Actions/DropWeight/Task_0085/ForceData_00065.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-015/Data/Actions/DropWeight/Task_0059/ForceData_00084.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-019/Data/Actions/DropWeight/Task_0066/ForceData_00082.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-021/Data/Actions/DropWeight/Task_0043/ForceData_00098.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-029/Data/Actions/DropWeight/Task_0038/ForceData_00173.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-031/Data/Actions/DropWeight/Task_0029/ForceData_00139.csv"
#file = "/home/makmak/cv2/Project/Data-20240930T100835Z-001/Data/Actions/DropWeight/Task_0082/ForceData_00051.csv"
#file = ""
#file = ""
#file = ""


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
    seed_contours, seed_frame, impact_index, init_height = find_impact_frame(time, volt, vidpath)
    hormin, hormax, vermin = obtain_crop_dimensions(seed_frame, seed_contours)
    cap = cv.VideoCapture(vidpath)
    cap.set(cv.CAP_PROP_POS_FRAMES, impact_index-1)
    heights = []
    base_indices = False
    frames = []
    #print(init_height)
    for i in range(impact_index-1, impact_index+150):
        ret, frame = cap.read()
        if ret:
            markers = obtain_markers(frame, hormin, hormax, vermin)
            if base_indices == False:
                base_index, base_height = obtain_base_index(markers) 
                base_indices = True 
                init_height = init_height-base_height
                #print(init_height)
                #print(base_index)
            height = obtain_height_from_markers(markers, base_index, init_height)
            heights.append(height)
            frames.append(i)
            #print(i/25000, height)
        else: continue

    return frames, (heights-init_height)/init_height

#time, volt = np.loadtxt(file, delimiter= ',',skiprows=2, unpack=True )
#generate_strain_graph(time, volt, vidpath)