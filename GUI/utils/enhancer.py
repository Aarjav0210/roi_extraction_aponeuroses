import numpy as np

def isolate_intensity(im, method=None):
    row_interval = im.shape[1]//7
    top_rows = np.argsort(np.mean(im, axis=1))[-row_interval:]
    median_row = int(np.median(top_rows))
    
    threshold = 100
    aponeuroses = im.copy()
    if method == 'Q1':
        # Find the q1 of the top 5 rows
        q1 = np.percentile(im[top_rows, :], 25)
        threshold = q1
    elif method == 'AMI':
        min_intensities = [np.min(im[int(row)]) for row in top_rows[:median_row]]
        avg_min_intensity = np.mean(min_intensities)
        threshold = avg_min_intensity
    else:
        # Error
        raise ValueError("Invalid method for isolating intensity")

    aponeuroses[aponeuroses < threshold] = 0
    
    return aponeuroses
