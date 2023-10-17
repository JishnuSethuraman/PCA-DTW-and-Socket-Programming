import pandas as pd
import numpy as np
import flasksocketserver
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sensor_data = {
    "accelerometer": [],
    "gravity": [],
    "gyroscope": [],
    "orientation": [],
    "magnetometer": [],
    "barometer": [],
    "location": [],
    "microphone": [],
    "pedometer": [],
    "headphone": [],
    "battery": [],
    "brightness": [],
    "network": [],
}


datacsv = pd.read_csv('20231009102957.csv')

for index, row in datacsv.iterrows():
    payloads = json.loads(row['payload'].replace("'", '"'))
    for payload in payloads:
        sensor_type = payload['name']
        if sensor_type in sensor_data:
            values = list(payload['values'].values())
            sensor_data[sensor_type].append(values)

for sensor_name, data in sensor_data.items():
    if len(data) > 1:
        cov_matrix = np.cov(data, rowvar=False)
        print(f"Covariance matrix for {sensor_name}:\n", cov_matrix)

WINDOW_SIZE = 10

def create_windows(data, window_size): #create fixed windows from time series
    return [data[i:i+window_size] for i in range(len(data) - window_size + 1)]

windowed_sensor_data = {sensor: create_windows(data, WINDOW_SIZE) for sensor, data in sensor_data.items()} #applying the function on sensor data


def plot_pca(sensor_name, data, windows):
    if not data or len(data[0]) <= 1:  # Check if data is empty or not 2D
        return  # If so, skip the plotting for this sensor
    
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)

    # Get the first window of this sensor
    first_window = windows[sensor_name][0]

    # Get the PCA transformation of the first window
    first_window_transformed = pca.transform(first_window)

    print(first_window_transformed)
    
    # Retrieve the first two principal components of the first window
    pc1, pc2 = first_window_transformed[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5, label="All data")
    plt.arrow(0, 0, pc1, pc2, head_width=0.5, head_length=0.5, fc='black', ec='black', label="1st window PCA")
    plt.title(f"2D PCA of {sensor_name} Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot PCA for each sensor
for sensor_name in windowed_sensor_data:
    plot_pca(sensor_name, sensor_data[sensor_name], windowed_sensor_data)