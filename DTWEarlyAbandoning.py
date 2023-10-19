import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from covmatPCA import get_reduced_data


def create_windows(data, window_size):
    # Split data into windows
    return [data[i:i + window_size] for i in range(0, len(data), window_size)]

def calculate_dtw_with_early_abandoning(reference_window, other_windows, distance_threshold):
    distances = []  # Store valid distances
    for window in other_windows:
        distance, _ = fastdtw(reference_window, window, dist=euclidean)  # Calculate DTW
        if distance < distance_threshold:  # Check threshold
            distances.append(distance)
    return distances  # Return distances meeting criteria


def calculate_dtw_between_series(series1, series2):
    distance, _ = fastdtw(series1, series2, dist=euclidean)
    return distance



def calculate_dtw_features():
    window_size = 10  # Window size set
    distance_threshold = 50  # Distance threshold set

    reduced_data_dict, reduced_data_dict1 =  get_reduced_data() # Retrieve PCA data

    if len(reduced_data_dict) == 0 or len(reduced_data_dict1) == 0:
        print("One or both of the time series data are empty. Cannot perform DTW.")
        return

    # Calculate DTW between the two full series
    dtw_distance = calculate_dtw_between_series(reduced_data_dict['accelerometer'], reduced_data_dict1['accelerometer'])

    print(f"DTW distance between the two series: {dtw_distance}")

    dtw_feature_matrix = []

    for sensor_type, pca_data in reduced_data_dict.items():  # Iterate through sensors

        if len(pca_data) == 0:  # Skip if no data
            continue

        windows = create_windows(pca_data, window_size)  # Create data windows

        if len(windows) < 2:  # Check sufficient data
            continue

        reference_window = windows[0]  # First window as reference

        dtw_distances = calculate_dtw_with_early_abandoning(reference_window, windows[1:], distance_threshold)  # DTW calculation

        if dtw_distances:  # If distances found
            print(f"DTW analysis for sensor: {sensor_type}")  # Analysis results
            print(f"Minimum DTW distance: {min(dtw_distances)}")
            print(f"Maximum DTW distance: {max(dtw_distances)}")
            print(f"Average DTW distance: {sum(dtw_distances) / len(dtw_distances)}")
            dtw_feature_matrix.append([
                min(dtw_distances),
                max(dtw_distances),
                sum(dtw_distances) / len(dtw_distances)  # average distance
            ])

        print("-" * 30)  # Separator
    return np.array(dtw_feature_matrix)

def main():
    try:
        feature_matrix = calculate_dtw_features()
        print("DTW feature matrix calculated successfully.")
        print(feature_matrix) # Print the feature matrix
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()  # Program entry point