import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
"""Assumptions Made:
- We are only interested in the distances between the vertical ends of the bars (if oriented differently need to switch x and y in 
obtain_bar_distance.)
- The vertical parts of the bar have an angle less than 5 degrees to the vertical
- Camera operates at a constant framerate
- Apart from the bar lines, there are no other (near) vertical lines present within frames
- constant ROI for timestamps (horizontally subject to change depending on number size)
- Constant ROI for bars
- Bars do not touch and neither bar occupies or goes past pixel 114
- Frame filenames can be sorted such that they are numbered name_001, name_002, name_003, ...

"""

def merge_all_on_one_line(results):
    """
    Sorts and merges all detected text from EasyOCR, assuming it's on a single line.
    """
    # Sort results based on the x-coordinate of the top-left corner
    results.sort(key=lambda r: r[0][0][0])
    all_confidences = [res[2] for res in results]
    
    # Extract just the text from each sorted result
    all_text = [res[1] for res in results]
    
    # Join all the text pieces into a single string
    merged_text = "".join(all_text)
    
    return merged_text, np.mean(all_confidences)

def prep_numeric_roi(gray):
    # Upscale to help the LSTM see stroke shapes
    big = cv.resize(gray, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
    blur = cv.GaussianBlur(big, (5,5), 0)
    # Otsu binarize; Tesseract prefers dark text on white
    _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    if th.mean() > 127: th = 255 - th
    # Slight close to beef up thin strokes
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    return th

def obtain_times(folder, reader):
    folder_path = Path(folder).expanduser()
    # specific extension(s)
    times = []
    frames = []
    true_frames = []
    true_times = []
    total_frame_num = sum(1 for _ in folder_path.glob("*.tiff"))
    for n, f in enumerate(sorted(folder_path.glob("*.tiff"))):
        frames.append(n)
        test_frame = f
        frame = cv.imread(test_frame)
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(img, (3,3), 0)
        _, thresh = cv.threshold(blur, 254, 255, type=cv.THRESH_BINARY)
        Black = True
        index = 325
        while Black:
            for i in range(238, 250):
                if img[i, index] > 250:
                    Black = False
            index += 1

        cropped = img[239:, (index-2):385]
        resized = prep_numeric_roi(cropped)
        results = reader.readtext(resized, allowlist='0123456789,')
        text, prob = merge_all_on_one_line(results)
        print(f'Detected text: "{text}" with confidence {prob:.2f}')
        time_ns = int(text.replace(',',''))
        if prob>= 0.999:
            true_frames.append(f)
            true_times.append(time_ns)
            if len(true_frames) >= 2 and len(true_times) >= 2:
                period = (true_times[-1] - true_times[0]) / (frames[-1] - frames[0])
                frames = np.arange(total_frame_num)
                times = period * frames
                return times
    return ("No true timestamps found, cannot assess the time series with certainty")        

def obtain_bar_distance(filepath):
    test_frame = filepath
    img = cv.imread(test_frame)[:220, :]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(2.0, (3,3))
    equalised = clahe.apply(gray)
    blur = cv.GaussianBlur(equalised, (13,13), 10)
    edges = cv.Canny(blur, 19, 22)
    lines = cv.HoughLinesP(edges, 1, np.pi/5000, 40, minLineLength=40, maxLineGap=20)
    img_with_lines = img.copy()
    img_with_edges = img.copy()
    img_with_edges[edges == 255] = [0, 255, 0]
    left = []
    right = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(np.arctan((x2-x1)/(y2-y1)))<0.1:
                midpoint = (x1+x2)/2
                if midpoint <= 116:
                    left.append(midpoint)
                else: right.append(midpoint)
                cv.line(img_with_lines,(x1, y1), (x2, y2), (0, 255, 0), 1)
        left_pos = np.mean(left)
        right_pos = np.mean(right)
        bar_distance = right_pos - left_pos
        bar_std = np.std(left) + np.std(right)
    else: raise RuntimeError("Could not locate any lines, removing datapoint")

    return bar_distance, bar_std

def obtain_video_data(folder, times):
    folder_path = Path(folder).expanduser()
    distances = []
    errors = []
    for n, f in enumerate(sorted(folder_path.glob("*.tiff"))):
        try:
            distance, std = obtain_bar_distance(f)
            distances.append(distance)
            errors.append(std)
        except:
            del times[n]
    return times, distances, errors

def analyse_folder(folder, output_dir, reader):
    """
    Analyzes data from a folder, plots the results, and saves the plot
    to the specified output directory.
    """
    # CHANGE 2: Remove this line ---> output_dir = 'results'
    # The output directory is now passed in as an argument.

    # --- Process the data ---
    times = obtain_times(folder, reader)
    times, distances, errors = obtain_video_data(folder, times)

    # --- Generate the plot ---
    plt.figure()
    plt.plot(times, distances, 'o')
    plt.title('Bar Distance against Time')
    plt.xlabel('Time (ns)')
    plt.ylabel('Distance (Pixels)')
    plt.errorbar(times, distances, yerr=errors, fmt='o')

    # --- Define the filename and save the figure ---
    folder_name = os.path.basename(os.path.normpath(folder))
    # This will now use the directory passed from your main script
    save_path = os.path.join(output_dir, f'{folder_name}.png') 

    plt.savefig(save_path)
    plt.close()

    print(f"Plot for '{folder}' saved to '{save_path}'")