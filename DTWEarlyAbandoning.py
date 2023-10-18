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

def main():
    window_size = 10  # Window size set
    distance_threshold = 50  # Distance threshold set

    reduced_data_dict = get_reduced_data()  # Retrieve PCA data

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

        print("-" * 30)  # Separator for readability

if __name__ == "__main__":
    main()  # Program entry point