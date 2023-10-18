import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Define the structure of your sensor data
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

reduced_data_dict = {}

# Load your data
# Ensure your CSV is in your working directory or provide the full path
data_csv = pd.read_csv('18131605.csv')

# Parsing the CSV and storing sensor data
for index, row in data_csv.iterrows():
    payloads = json.loads(row['payload'].replace("'", '"'))
    for payload in payloads:
        sensor_type = payload['name']
        if sensor_type in sensor_data:
            values = list(payload['values'].values())
            sensor_data[sensor_type].append(values)

# Iterate over each sensor type and perform PCA
for sensor, data in sensor_data.items():
    if not data:  # Skip if the sensor data is empty
        print(f"Empty sensor data, skipping PCA for {sensor}")
        continue
    
    # Convert the list of sensor readings into a DataFrame
    data_df = pd.DataFrame(data)
    
    cov_matrix = data_df.cov()
    print(f"Covariance matrix for {sensor}:\n", cov_matrix)

    # Skip if there's no variance or insufficient data
    if data_df.nunique().eq(1).all() or len(data_df) < 2:
        print(f"Sensor data has no variance or insufficient data, skipping PCA for {sensor}")
        continue
    
    # Perform PCA
    pca = PCA(n_components=2)  # You can adjust the number of components if needed
    data_reduced = pca.fit_transform(data_df)
    reduced_data_dict[sensor] = data_reduced

    # Visualizing the PCA results
    plt.figure()  # Create a new figure for each sensor
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA for {sensor}')
    plt.show()

    # Print the variance ratio
    print(f"Explained variance ratio for {sensor}:", pca.explained_variance_ratio_) 

def get_reduced_data():
    return reduced_data_dict
