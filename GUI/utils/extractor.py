import numpy as np

def __sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Calculate the angles of each point with respect to the centroid
    def angle_with_centroid(point):
        delta = point - centroid
        return np.arctan2(delta[1], delta[0])
    
    # Sort the points by the angles
    sorted_points = sorted(points, key=angle_with_centroid)
    
    return np.array(sorted_points)

# Get the 4 coordinates of the quadrilateral
def get_quadrilateral_coordinates(line_d, line_s, final_width, initial_width):
    # Line is represented as (slope, intercept)
    
    if line_d is None or line_s is None:
        return []

    line_d = (line_d[0], line_d[1] * (initial_width / final_width))
    line_s = (line_s[0], line_s[1] * (initial_width / final_width))
    
    slope_d, intercept_d = line_d
    slope_s, intercept_s = line_s
    

    # if nonetype return []
    if slope_d is None or intercept_d is None or slope_s is None or intercept_s is None:
        return []

    # Get the 4 points of the quadrilateral
    x1 = 0
    y1 = int(intercept_d)
    x2 = initial_width-1
    y2 = int(slope_d * x2 + intercept_d)
    x3 = 0
    y3 = int(intercept_s)
    x4 = initial_width-1
    y4 = int(slope_s * x4 + intercept_s)

    coordinates = [[x3, y3],[x4, y4], [x2, y2], [x1, y1]]
    coordinates = [[max(0, x), max(0, y)] for x, y in coordinates]
    coordinates = [[min(initial_width-1, x), min(initial_width-1, y)] for x, y in coordinates]

    coordinates = __sort_points_clockwise(coordinates)
    return coordinates



# def extract_coordinates(frame_path, technique, method, threshold, flip=False):
#     original_frame = load_image(frame_path)
#     frame = original_frame.copy()
#     if flip:
#         frame = cv2.flip(frame, 1)
#     if technique == 'gaussian':
#         frame = gaussian_blur(frame)
#     elif technique == 'bilateral':
#         frame = bilateral_filter(frame)
#     elif technique == 'average':
#         frame = layered_average_blur(frame)
#     elif technique == 'denoised':
#         frame = denoise_image(frame)

#     frame = isolate_intensity(frame, method)
#     edges = edge_detection(frame)
#     _, line_d, line_s = hough_transform(edges, threshold=threshold)
#     if line_d is None or line_s is None:
#         return []
    
#     # if the technique is average, we need to scale the lines back from the down sampled image to the original image
#     if technique == 'average':
#         # Down sampling -> 1/5 -> 1/3
#         # Get the shape of the original image
#         original_width = load_image(frame_path).shape[1]
#         # Down sampled width
#         down_sampled_width = frame.shape[1]
#         # Scale the lines back to the original image
#         line_d = (line_d[0], line_d[1] * (original_width / down_sampled_width))
#         line_s = (line_s[0], line_s[1] * (original_width / down_sampled_width))
        
    
#     coordinates = get_quadrilateral_coordinates(line_d, line_s, original_frame.shape[1])
#     #  set any negative coordinates to 0
#     coordinates = [[max(0, x), max(0, y)] for x, y in coordinates]
#     return coordinates
