import json
import numpy as np
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import Polygon

def sort_points_clockwise(points):
    """
    Sorts the points in a clockwise order.
    Args:
        points: A numpy array of shape (n, 2) representing the points.
    Returns:
        A numpy array of shape (n, 2) representing the sorted points.
    """

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Calculate the angles of each point with respect to the centroid
    def angle_with_centroid(point):
        delta = point - centroid
        return np.arctan2(delta[1], delta[0])
    
    # Sort the points by the angles
    sorted_points = sorted(points, key=angle_with_centroid)
    
    return np.array(sorted_points)


def read_roi(file_name, filepath, frame_num):
    """
    Reads the ROI from the CSV file.
    Args:
        file_name: The name of the file.
        filepath: The path to the CSV file.
        frame_num: The frame number.
    Returns:
        A numpy array of shape (n, 2) representing the ROI.
    """

    df = pd.read_csv(filepath)
    file_name = file_name.split('.')[0]
    row = df.loc[df['Filename'] == file_name.split('.')[0]]
    pts = np.asarray((json.loads(row[f'Frame{frame_num}'].values[0])))

    pts = sort_points_clockwise(pts)

    return pts


def calculate_IoU(roi, extracted):
    """
    Calculates the Intersection over Union (IoU) between the ROI and the extracted polygon.
    Args:
        roi: A numpy array of shape (n, 2) representing the ROI.
        extracted: A numpy array of shape (n, 2) representing the extracted polygon.
    Returns:
        A float representing the IoU.
    """
    
    poly1 = Polygon(roi)
    poly2 = Polygon(extracted)

    poly1 = make_valid(poly1)
    poly2 = make_valid(poly2)

    # Calculate intersection and union areas
    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
    except Exception as e:
        print(f"Error processing intersection or union: {e}")
        print(f"poly1: {poly1}")
        print(f"poly2: {poly2}")


    # Calculate IoU
    iou = intersection_area / union_area

    return iou, intersection_area, union_area

