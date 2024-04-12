import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Hough transform
def extract_lines(edges, threshold=10):
    min_line_length = edges.shape[1]//3
    max_line_gap = edges.shape[1]//5
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        print("No lines found, reducing the min line length")
        min_line_length = edges.shape[1]//5
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
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
def filter_lines_by_y(lines, y_min):
    filtered_lines = []
    for line in lines:
        for _, y1, _, y2 in line:
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
    if not m_values or not c_values:  # Check if lists are empty
        return None, None
    
    slope = np.median(m_values)
    intercept = np.median(c_values)
    return slope, intercept

def slope_to_angle(slope):
    # If slope is not a number, print an error message and return None
    if math.isnan(slope):  # Check for NaN
        print("Slope is not a number")
        print(type(slope))
        return None
    return int(math.atan(slope) * 180.0 / np.pi)

def visualise_line(im, slope, intercept):
    # Draw a line with the calculated slope and intercept on a copy of the original image
    if slope is None or intercept is None:  # Check for invalid input
        print("Invalid slope or intercept for visualization")
        return im
    aponeuroses_line = im.copy().astype('uint8')
    # aponeuroses_line = aponeuroses_line.copy().astype('uint8')
    
    # Convert to 3 channels
    if len(aponeuroses_line.shape) == 2:
        aponeuroses_line = cv2.cvtColor(aponeuroses_line, cv2.COLOR_GRAY2BGR)
    width = aponeuroses_line.shape[1]

    # Draw the line
    x1 = 0
    y1 = int(intercept)
    x2 = width
    y2 = int(slope * x2 + intercept)

    cv2.line(aponeuroses_line, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return aponeuroses_line

def hough_transform_deep(edges, angle_min=-30, angle_max=-1, visualise=None, threshold=20):
    lines = extract_lines(edges, threshold)
    if lines is None:
        return edges, None, None
    filtered_lines = filter_lines_by_angle(lines, angle_min, angle_max)
    if filtered_lines == []:
        print("No lines found, reducing the angle range")
        return edges, None, None
    slope, intercept = calculate_slope_intercept(filtered_lines)
    if slope is None or intercept is None:  # Check for valid slope and intercept
        print("Slope or intercept calculation failed in deep transform.")
        return edges, None, None
    angle = slope_to_angle(slope)
    if angle is None:  # Check for valid angle
        print("Angle calculation failed in deep transform.")
        return edges, None, None
    # Repeat the process with the angle range narrowed down +/- 2 degrees until convergence
    while True:
        filtered_lines = filter_lines_by_angle(filtered_lines, angle - 2, angle + 2)
        new_slope, new_intercept = calculate_slope_intercept(filtered_lines)
        if new_slope is None or new_intercept is None:
            break 
        new_angle = slope_to_angle(new_slope)
        if new_angle is None or abs(new_angle - angle) == 0:
            slope, intercept = new_slope, new_intercept
            break
        angle = new_angle
    if visualise is not None:
        aponeuroses_line = visualise_line(visualise, slope, intercept)
    else:
        aponeuroses_line = visualise_line(edges, slope, intercept)
    return aponeuroses_line, slope, intercept

def hough_transform_superficial(edges, y_max, angle_min=-10, angle_max=10, visualise=None, threshold=20):
    slope = 0
    intercept = 0
    lines = extract_lines(edges, threshold)
    if lines is None:
        return edges, slope, intercept
    filtered_lines = filter_lines_by_angle(lines, angle_min, angle_max)
    if not filtered_lines:  # Check if list is empty
        print("No lines found after angle filtering in superficial transform.")
        return edges, slope, intercept
    filtered_lines = filter_lines_by_y(filtered_lines, y_max)
    if not filtered_lines:  # Check if list is empty after y filtering
        print("No lines found after y filtering in superficial transform.")
        return edges, slope, intercept
    slope, intercept = calculate_slope_intercept(filtered_lines)
    if slope is None or intercept is None:  # Check for valid slope and intercept
        print("Slope or intercept calculation failed in superficial transform.")
        return edges, 0, 0
    angle = slope_to_angle(slope)
    if angle is None:  # Check for valid angle
        print("Angle calculation failed in superficial transform.")
        return edges, 0, 0
    while True:
        filtered_lines = filter_lines_by_angle(filtered_lines, angle - 2, angle + 2)
        new_slope, new_intercept = calculate_slope_intercept(filtered_lines)
        if new_slope is None or new_intercept is None:
            break  # Exit if slope or intercept becomes invalid
        new_angle = slope_to_angle(new_slope)
        if new_angle is None or abs(new_angle - angle) == 0:
            slope, intercept = new_slope, new_intercept
            break
        angle = new_angle
    if visualise is not None:
        aponeuroses_line = visualise_line(visualise, slope, intercept)
    else:
        aponeuroses_line = visualise_line(edges, slope, intercept)
    return aponeuroses_line, slope, intercept

def hough_transform(edges, angle_min_d=-30, angle_max_d=-1, angle_min_s=-10, angle_max_s=10, visualise=None, threshold=20):
    print(threshold)
    # Deep aponeuroses
    aponeuroses_line_d, slope_d, intercept_d = hough_transform_deep(edges, angle_min_d, angle_max_d, visualise, threshold)
    # plt.imshow(aponeuroses_line_d)
    # get the y_max of the deep aponeuroses
    if slope_d == None:
        return aponeuroses_line_d, None, None
    y_max = min((intercept_d, intercept_d + slope_d * edges.shape[1]))
    # Superficial aponeuroses
    aponeuroses_line_s_d, slope_s, intercept_s = hough_transform_superficial(edges, y_max, angle_min_s, angle_max_s, aponeuroses_line_d, threshold)
    return aponeuroses_line_s_d, (slope_d, intercept_d), (slope_s, intercept_s)