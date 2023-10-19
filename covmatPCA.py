import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parsing the CSV and storing sensor data
def parse_and_store_data(row, sensor_data_dict):
    payloads = json.loads(row['payload'].replace("'", '"'))
    for payload in payloads:
        sensor_type = payload['name']
        if sensor_type in sensor_data_dict:
            values = list(payload['values'].values())
            sensor_data_dict[sensor_type].append(values)

# Iterate over each sensor type and perform PCA
def perform_pca(sensor_data):
    reduced_data_dict = {}
    for sensor, data in sensor_data.items():
        if not data:  # Skip if the sensor data is empty
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
    return reduced_data_dict

def process_csv_file(file_path):
    """Load data from a CSV file, parse it, and perform PCA."""
    # Initial dictionary for storing sensor data
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
    # Load and parse data
    data_csv = pd.read_csv(file_path)
    for index, row in data_csv.iterrows():
        parse_and_store_data(row, sensor_data)

    # Perform PCA on collected data and get the reduced data
    reduced_data_dict = perform_pca(sensor_data)

    return reduced_data_dict

def get_reduced_data():
    # Process each CSV file with the respective function and get the reduced data dictionaries
    reduced_data_dict = process_csv_file('18131605.csv')
    reduced_data_dict1 = process_csv_file('18170414.csv')

    # Return the separate dictionaries as needed
    return reduced_data_dict, reduced_data_dict1