import pandas as pd
import numpy as np
import flasksocketserver
import json
import matplotlib.pyplot as plt

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

for sensor, data in sensor_data.items(): #process each sensor type
    if not data:  #skip if the sensor data is empty
        print(f"Empty sensor data, skipping PCA for {sensor}")
        continue

    data_array = np.array(data).T
    if data_array.ndim == 1: #if data_array has 1 dimension
        data_centered = data_array - np.mean(data_array)
    else:
        data_centered = data_array - np.mean(data_array, axis=1, keepdims=True)
    if data_array.ndim > 1:  #if data_array has more than 1 dimension
        data_centered = data_array - np.mean(data_array, axis=1, keepdims=True)
    else:
        data_centered = data_array - np.mean(data_array)
    if data_centered.size > 0:
        if np.unique(data_centered).size > 1:
            cov_matrix = np.cov(data_centered)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            eigenvalues = eigenvalues[::-1] #reverse order

            eigenvectors = eigenvectors[:, ::-1] #reverse column order

            variance = eigenvalues / np.sum(eigenvalues)

            plt.bar(range(len(eigenvalues)), variance)
            plt.xlabel('Principal Component')
            plt.ylabel('Variance')
            plt.show()

            projected_data = np.dot(data_centered.T, eigenvectors[:, :2])

            scale_factor = 0.7  #to scale the length of the arrow
            for i in range(2):  #since we're considering 2 principal components in 2D
                plt.arrow(0, 0, 
                        eigenvectors[i, 0]*eigenvalues[i]*scale_factor, 
                        eigenvectors[i, 1]*eigenvalues[i]*scale_factor,
                        head_width=1, head_length=1, fc='black', ec='black')

            plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
            plt.xlabel('Unit 1')
            plt.ylabel('Unit 2')
            plt.title('2D PCA from Sensor Data')
            plt.show()
        else:
            print(f"Sensor data has no variance, skipping PCA for {sensor}")