from scipy.io import loadmat
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import math
from preprocessors.bilateral_filter import bilateral_filter
from preprocessors.gaussian_blur import gaussian_blur
from preprocessors.noise_reduction import denoise_image
from preprocessors.average_blur import layered_average_blur

# Load the data
def load_data(path):
    if not path.endswith('.mat'):
        raise ValueError('File must be a .mat file')
    
    mat_contents = loadmat(path)

    # Access the TVDdata struct
    tvd_data = mat_contents['TVDdata'][0, 0] # Assuming TVDdata is a 1x1 struct

    # Extract the fields from TVDdata
    fnum = tvd_data['Fnum'][0, 0] # Numerical value
    im = tvd_data['Im'] # 3D array
    return fnum, im


def binary_threshold(im, method=None):
    row_interval = im.shape[1]//7
    top_rows = np.argsort(np.mean(im, axis=1))[-row_interval:]
    median_row = int(np.median(top_rows))
    
    threshold = 100
    if method == 'q1':
        q1 = np.percentile(im[top_rows, :], 25)
        threshold = q1
    elif method == 'avg_min_intensity':
        min_intensities = [np.min(im[int(row)]) for row in top_rows[:median_row]]
        avg_min_intensity = np.mean(min_intensities)
        threshold = avg_min_intensity
        
    _, im_binary = cv2.threshold(im, threshold, 255, cv2.THRESH_BINARY)
    return im_binary


# Preprocess the image with the following techniques:
# 1. Denoising
# 2. Bilateral filtering
# 3. Gaussian blurring
# 4. Binary thresholding
# 5. Layered average blurring
# def preprocess_image(im, technique, method=None):
#     if technique == 'denoise':
#         return denoise_image(im)
#     elif technique == 'bilateral':
#         return bilateral_filter(im)
#     elif technique == 'gaussian':
#         return gaussian_blur(im)
#     elif technique == 'binary':
#         return binary_threshold(im, method)
#     elif technique == 'layered':
#         return layered_average_blur(im)
#     else:
#         raise ValueError('Invalid preprocessing technique')
    

# Perform intensity thresholding to extract the aponeuroses
    # 1. Use the q1
    # 2. Use the avg_min_intensity
def isolate_intensity(im, method=None, skip=False):
    row_interval = im.shape[1]//7
    top_rows = np.argsort(np.mean(im, axis=1))[-row_interval:]
    median_row = int(np.median(top_rows))
    
    threshold = 100
    aponeuroses = im.copy()
    if skip == False:
        if method == 'q1':
            # Find the q1 of the top 5 rows
            q1 = np.percentile(im[top_rows, :], 25)
            threshold = q1
        elif method == 'avg_min_intensity':
            min_intensities = [np.min(im[int(row)]) for row in top_rows[:median_row]]
            avg_min_intensity = np.mean(min_intensities)
            threshold = avg_min_intensity

        aponeuroses[aponeuroses < threshold] = 0
    
    return aponeuroses

# Edge detection
def edge_detection(im):
    edges = cv2.Canny(im.astype('uint8'), 50, 150)
    return edges

# Hough transform
def extract_lines(edges, threshold=6):
    min_line_length = edges.shape[1]//3
    max_line_gap = edges.shape[1]//5
    lines = cv2.lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        print("No lines found, reducing the min line length")
        min_line_length = edges.shape[1]//5
        lines = cv2.lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# Filter the lines by angle
def filter_lines_by_angle(lines, angle_min=-30, angle_max=-1):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if angle_min <= angle <= angle_max:
                filtered_lines.append(line)
    return filtered_lines

# Filter to get only lines that are above a certain line
def filter_lines_by_y(lines, y_start, y_end):
    filtered_lines = []
    y_min = min(y_start, y_end)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 < y_min and y2 < y_min:
                filtered_lines.append(line)
    return filtered_lines

def calculate_slope_intercept(lines):
    m_values = []
    c_values = []
    len(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 != 0:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                m_values.append(m)
                c_values.append(c)
            else:  # Vertical line, slope is infinite, so we skip or handle differently
                pass

    slope = np.median(m_values)
    intercept = np.median(c_values)
    return slope, intercept

def get_aponeurosis_points(im, slope, intercept):
    x_start = 0
    x_end = im.shape[1]
    y_1 = slope * x_start + intercept
    y_2 = slope * x_end + intercept
    return [[x_start, y_1], [x_end, y_2]]

def slope_to_angle(slope):
    return int(math.atan(slope) * 180.0 / np.pi)

def correct_scale(original, current, points):
    # Given an original image and the current image, scale the points from the current image to the original image
    scale_x = original.shape[1] / current.shape[1]
    scale_y = original.shape[0] / current.shape[0]
    for i in range(len(points)):
        points[i][0] *= scale_x
        points[i][1] *= scale_y
    return points

def extract_aponeuroses_gaussian(im, k=3, method=None):
    # Preprocess the image
    # print(f"Preprocessing the image using {technique}")
    im_preproc = gaussian_blur(im, k)
    # Isolate the intensity
    # print(f"Isolating the intensity using {method}")
    features = isolate_intensity(im_preproc, method)
    # Edge detection
    # print("Edge detection")
    edges = edge_detection(features)
    # Hough transform
    # print("Hough transform")
    lines = extract_lines(edges)
    if lines is None:
        return im, []
    # print(f"Number of lines: {len(lines)}")

    # Get the deep aponeruosis
    # Filter the lines
    filtered_lines = filter_lines_by_angle(lines)
    if len(filtered_lines) == 0:
        return im, []
    # print(f"Number of filtered lines: {len(filtered_lines)}")
    # Calculate the slope and intercept
    slope, intercept = calculate_slope_intercept(filtered_lines)
    # Calculate slope to angle
    angle = 0
    new_angle = slope_to_angle(slope)
    while (new_angle != angle):
        angle = new_angle
        new_filtered_lines = filter_lines_by_angle(filtered_lines, angle - 2, angle + 2)
        # print(f"Number of new filtered lines: {len(new_filtered_lines)}")
        if len(new_filtered_lines) == 0:
            break
        filtered_lines = new_filtered_lines
        slope, intercept = calculate_slope_intercept(filtered_lines)
        new_angle = slope_to_angle(slope)        

    pts_d = get_aponeurosis_points(im_preproc, slope, intercept)


    # print(f"Deep aponeurosis: {pts_d}")
    # Get the superficial aponeurosis
    # Filter the lines
    filtered_lines = filter_lines_by_y(lines, pts_d[0][1], pts_d[1][1])

    filtered_lines = filter_lines_by_angle(filtered_lines, -10, 10)
    if len(filtered_lines) == 0:
        return im, []
    # print(f"Number of filtered lines: {len(filtered_lines)}")
    # Calculate the slope and intercept
    slope, intercept = calculate_slope_intercept(filtered_lines)
    # Calculate slope to angle
    angle = 0
    new_angle = slope_to_angle(slope)
    while (new_angle != angle):
        angle = new_angle
        new_filtered_lines = filter_lines_by_angle(filtered_lines, angle - 2, angle + 2)
        # print(f"Number of new filtered lines: {len(new_filtered_lines)}")
        if len(new_filtered_lines) == 0:
            break
        filtered_lines = new_filtered_lines
        slope, intercept = calculate_slope_intercept(filtered_lines)
        new_angle = slope_to_angle(slope)        


    pts_s = get_aponeurosis_points(im_preproc, slope, intercept)
    if len(pts_s) == 0:
        pts_s = [[0, 0], [im_preproc.shape[1], 0]]

    # Draw a polygon connecting the points
    pts = np.array([pts_d[0], pts_s[0], pts_s[1], pts_d[1]], np.int32)

    # Scale the points to the original image
    pts = correct_scale(im, im_preproc, pts)

    # Draw the polygon on the image
    if len(im.shape) == 2 or im.shape[2] == 1:
        im = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_GRAY2BGR)
    im = cv2.polylines(im.astype('uint8'), [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    return im, pts




# Test the functions
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    data_dir_path = "YOUR MAT FILES DIRECTORY PATH HERE"
    # All files in the data directory with the .mat extension
    files = [f for f in os.listdir(data_dir_path) if f.endswith('.mat')]

    for file_name in files:
        print(f"Processing file: {file_name}")
        file_path = data_dir_path + file_name
        fnum, im = load_data(file_path)

        # CSV filename
        csv_file_name = 'gaussian_data.csv'
        # Define DataFrame columns
        columns = ['Filename', 'Technique', 'Kernel', 'Method', 'Frame1', 'Frame2', 'Frame3', 'Frame4', 'Frame5', 'Frame6', 'Frame7', 'Frame8', 'Frame9', 'Frame10']
        # Check if CSV already exists
        if not os.path.exists(csv_file_name):
            pd.DataFrame(columns=columns).to_csv(csv_file_name, index=False)

        # Get 10 equidistant frames
        frames = np.linspace(0, fnum-1, 10, dtype=int)
        
        technique = 'gaussian'
        kernel_sizes = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        methods = ['q1', 'avg_min_intensity']

        for k in kernel_sizes:
            print(f"Kernel size: {k}")
            for method in methods:
                # print('-' * 50)
                # print(f"Technique: {technique}, Method: {method}")
                frame_data = {'Filename': file_name, 'Technique': technique, 'Kernel': k, 'Method': method}

                for i, f in enumerate(frames):
                    frame = im[:, :, f]
                    print(f"Frame: {f}")
                    _, pts = extract_aponeuroses_gaussian(frame, k, method)
                    frame_data[f'Frame{i+1}'] = str(pts)  # Convert points to string for CSV storage

                # Append the data for this technique and method to the CSV
                pd.DataFrame([frame_data]).to_csv(csv_file_name, mode='a', header=False, index=False)
