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

def create_windows(data, window_size):
    return [data[i:i+window_size] for i in range(0, len(data), window_size)]

# Function to apply PCA and return the transformed data and explained variance
def apply_pca(data):
    pca = PCA(n_components=2)  # We're using 2 components for visualization
    transformed_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return transformed_data, explained_variance

# Main process function for section 2.2
def process_sensor_data(sensor_data):
    window_size = 10  # Size of each data window

    # Store the PCA results: the transformed windows and the variance explained
    pca_results = {}
    print("Debug: Starting the visualization process...")  # Debug statement
    for sensor_type, data in sensor_data.items():
        
        if not data:
            continue  # Skip if no data for this sensor type

        windows = create_windows(data, window_size)

        # For visualization, we'll store one plot of original data and one with the principal components
        original_data_for_plot = None 
        pca_transformed_for_plot = []
        explained_variances = []

        for window in windows:
            if len(window) < window_size:
                continue  # Avoid processing windows that don't meet the desired size

            print(f"Debug: window content: {window}")

            pca_data, explained_variance = apply_pca(window)

            # Assuming you want to keep all the transformed windows for this sensor
            pca_transformed_for_plot.append(pca_data)
            explained_variances.append(explained_variance)

            # Storing only the first window for the original data plot
            if original_data_for_plot is None:
                original_data_for_plot = np.array(window)

        # Now, we have the data needed for plotting
        if pca_transformed_for_plot:
            pca_results[sensor_type] = {
                'original_data': original_data_for_plot,
                'pca_data': pca_transformed_for_plot,
                'explained_variance': explained_variances
            }

    # Once all data is processed, you can create visualizations
    for sensor_type, results in pca_results.items():
        print(f"Debug: Processing visualization for {sensor_type}...")  # Debug statement
        # Plot the original data (just the first window, as you specified)
        original_data_array = np.array(results['original_data'])

        # Plot the original data (just the first window, as you specified)
        plt.figure(figsize=(10, 5))
        plt.title(f'Original Data for {sensor_type}')
        plt.scatter(original_data_array[:, 0], original_data_array[:, 1])  # adjusted indexing here
        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')
        plt.show()

        # Plot the principal components for the first window
        plt.figure(figsize=(10, 5))
        plt.title(f'Principal Components for {sensor_type}')
        plt.scatter(results['pca_data'][0][:, 0], results['pca_data'][0][:, 1], c='red', label='Principal Components')  # Only the first window

        # If you want to overlay the original data points in the background of the PCA plot
        plt.scatter(results['original_data'][:, 0], results['original_data'][:, 1], c='blue', alpha=0.5, label='Original Data')
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        # Output the explained variance
        print(f'Explained variance for {sensor_type}: {results["explained_variance"][0]}')  # Only for the first window

    # At this point, you might want to save 'pca_results' or return it for further processing.

print("Debug: About to start processing sensor data...")  # Debug statement
process_sensor_data(sensor_data)