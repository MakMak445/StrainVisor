import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys
import pandas as pd
from py2DIC.sources import Main
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
    clahe = cv.createCLAHE(1.5, (2,2))
    equalised = clahe.apply(gray)
    #cv.imshow("contrast improved", equalised)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
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
        std_left = np.std(left)
        std_right = np.std(right)
        cropped = equalised[:, int(np.floor(left_pos-std_left)):int(np.ceil(right_pos+std_right))]
        print(left_pos, right_pos)
        clahe.apply(cropped)
        #cv.imshow("cropped image", cropped)
        #cv.waitKey(300)
        bar_distance = right_pos - left_pos
        bar_std = std_left + std_right
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
    Analyzes data from a folder, saves the data to a CSV file, plots the
    results, and saves the plot to the specified output directory.
    """
    print(f"Processing data for '{folder}'...")
    
    times = obtain_times(folder, reader)

    # Gracefully handle cases where no valid timestamps were found
    if isinstance(times, str):
        print(f"Skipping folder '{folder}': {times}", file=sys.stderr)
        return

    times, distances, errors = obtain_video_data(folder, times)

    # --- 2. Save the data to a CSV file ---
    folder_name = os.path.basename(os.path.normpath(folder))
    csv_save_path = os.path.join(output_dir, f'{folder_name}.csv')
    
    try:
        # Create a pandas DataFrame to hold the results
        df = pd.DataFrame({
            'Time (ns)': times,
            'Distance (Pixels)': distances,
            'Error (Pixels)': errors
        })
        
        # Save the DataFrame to a CSV file, without the index column
        df.to_csv(csv_save_path, index=False)
        print(f"Data for '{folder}' saved to '{csv_save_path}'")

    except Exception as e:
        print(f"Could not save CSV for '{folder}': {e}", file=sys.stderr)


    # --- 3. Generate and save the plot ---
    # Changed the file extension from .png to .svg for a vector image
    plot_save_path = os.path.join(output_dir, f'{folder_name}.svg')
    
    try:
        plt.figure()
        plt.plot(times, distances, 'o')
        plt.title(f'Bar Distance vs. Time for {folder_name}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Distance (Pixels)')
        plt.errorbar(times, distances, yerr=errors, fmt='o', capsize=3)
        
        plt.savefig(plot_save_path)
        print(f"Plot for '{folder}' saved to '{plot_save_path}'")
        print(f"mean error of {np.mean(errors)}")

    except Exception as e:
        print(f"Could not save plot for '{folder}': {e}", file=sys.stderr)
    finally:
        # Close the plot to free up memory
        plt.close()


#folder_path = Path("/home/makmak/Projects/cv2/Images/Camera_Njord/Njord_09_57_18").expanduser()
#for n, f in enumerate(sorted(folder_path.glob("*.tiff"))):
#    obtain_bar_distance(f)
#cv.destroyAllWindows()