import covmatPCA
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def dtw(s1, s2, threshold):
    distance, _ = fastdtw(s1, s2, dist=euclidean)
    if distance > threshold:
        return np.inf  #if distance exceeds threshold, abandon
    return distance

windowed_sensor_data = covmatPCA.windowed_sensor_data #windowed_sensor_data from covmatPCA

sensor_names = list(windowed_sensor_data.keys()) #applying DTW to two different sets of time-series data

if len(sensor_names) >= 2 and len(windowed_sensor_data[sensor_names[0]]) > 0 and len(windowed_sensor_data[sensor_names[1]]) > 0:
    s1 = windowed_sensor_data[sensor_names[0]][0]
    s2 = windowed_sensor_data[sensor_names[1]][0]

    threshold = 50  #threshold for early abandoning
    distance = dtw(s1, s2, threshold)
    if distance == np.inf:
        print(f"Abandoned computation early for sensors {sensor_names[0]} and {sensor_names[1]}. DTW distance exceeds threshold.")
    else:
        print(f"DTW distance between {sensor_names[0]} and {sensor_names[1]}: {distance}")

else:
    print("Not enough data to compare two sensors.")