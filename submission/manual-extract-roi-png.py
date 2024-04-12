from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

def append_row_to_csv(file_path, row_data):
    """Append a single row of data to a CSV file."""
    with open(file_path, 'a') as f:
        pd.DataFrame([row_data]).to_csv(f, header=f.tell()==0, index=False)

# Get all folders in data_path
def get_folders(data_path):
    return [os.path.join(data_path,f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

# Get all subfolders in data_path
def get_subfolders(data_path):
    return [os.path.join(data_path,f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

# Get all files in data_path
def get_files(data_path):
    # [os.path.join(data_path,f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.png')]
    all_files = []
    for f in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.png'):
            all_files.append(os.path.join(data_path, f))

    # Get 10 equispaced files
    file_indexes = np.linspace(0, len(all_files)-1, 10, dtype=int)
    selected_files = [all_files[i] for i in file_indexes]
    return selected_files

# Load png at file_path as a numpy array
def load_image(file_path):
    return np.asarray(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))

if __name__ == '__main__':
    data_path = '/Users/aarjavjain/Desktop/Education/Y4/PRJ/data'
    # Initialize DataFrame
    columns = ['Filename', 'Frame1', 'Frame2', 'Frame3', 'Frame4', 'Frame5',
               'Frame6', 'Frame7', 'Frame8', 'Frame9', 'Frame10']
    
    roi_df = pd.DataFrame(columns=columns)

    plt.ion()

    folders = get_folders(data_path)
    folders.remove('/Users/aarjavjain/Desktop/Education/Y4/PRJ/data/WW174')
    folders.remove('/Users/aarjavjain/Desktop/Education/Y4/PRJ/data/WW112')
    subfolders = []
    for folder in folders:
        subfolders += get_subfolders(folder)
    
    for file_name in subfolders:
        file_path = file_name
        print(f'Processing {file_name}')
        # Dictionary to store ROIs for each frame
        roi_data = {'Filename': file_name.split('.')[0]}
        for i, file in enumerate(get_files(file_path)):
            frame = load_image(file)
            frame = cv2.flip(frame, 1)

            fig, ax = plt.subplots()
            ax.imshow(frame, cmap='gray')
            ax.set_title(f'Frame: {file}')

            # print('Please click 4 points in the image to define the ROI corners.')
            # print('Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left')

            pts = []
            while len(pts) < 4:
                pt = plt.ginput(1, show_clicks=True)
                pts.extend(pt)  # Add the new point to the list

                # Convert list of tuples to a numpy array for easier manipulation
                np_pts = np.array(pts, dtype=int)

                # Clear the previous lines/points and redraw the image and the new line
                ax.clear()
                ax.imshow(frame, cmap='gray')
                ax.plot(np_pts[:, 0], np_pts[:, 1], marker='x', color='r', linestyle='-', linewidth=2)  # Draw line connecting points
                plt.draw()

            plt.close(fig)

            # Now, process pts as before to extract the ROI...
            pts = np.array(pts, dtype=int)  # This conversion is redundant in this context but included for consistency with the previous setup

            # Sort the points by their y-coordinates
            # The assumption is the top two points have lower y values, and the bottom two have higher y values
            top_points = pts[np.argsort(pts[:, 1])[:2]]
            bottom_points = pts[np.argsort(pts[:, 1])[-2:]]

            # Further sort the top and bottom pairs by their x-coordinates to get left and right
            tl = top_points[np.argmin(top_points[:, 0])]
            tr = top_points[np.argmax(top_points[:, 0])]
            br = bottom_points[np.argmax(bottom_points[:, 0])]
            bl = bottom_points[np.argmin(bottom_points[:, 0])]

            # Construct the ROI array
            roi = np.array([tl, tr, br, bl])

            # Convert the ROI array to a JSON string
            roi_json = json.dumps(roi.tolist())
            roi_data[f'Frame{i}'] = roi_json

        # Append the ROI data to the DataFrame
        append_row_to_csv('roi_data-png.csv', roi_data)

plt.ioff()
print('ROI data saved to roi.csv')
    
