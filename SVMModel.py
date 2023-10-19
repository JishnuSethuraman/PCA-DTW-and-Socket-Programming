import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from covmatPCA import get_reduced_data
from DTWEarlyAbandoning import calculate_dtw_features

def visualize_one_class_svm(model, scaler, reduced_data):
    aggregateData = []
    for sensor_data in reduced_data.values():
        aggregateData.extend(sensor_data)

    if not aggregateData:
        print("No reduced data available for visualization.")
        return

    aggregateData_np = np.array(aggregateData)

    # Standardize the data using the same scaler used for training
    aggregateData_np_scaled = scaler.transform(aggregateData_np)

    # Predict using the One-Class SVM model
    predictions = model.predict(aggregateData_np_scaled)  # Outliers are -1 and inliers are 1

    # Separate inliers and outliers for plotting
    inliers = aggregateData_np_scaled[predictions == 1]
    outliers = aggregateData_np_scaled[predictions == -1]

    # Plotting the inliers and outliers
    plt.figure()

    if len(inliers) > 0:
        plt.scatter(inliers[:, 0], inliers[:, 1], label='Inliers', edgecolors='b', s=60, alpha=0.8, marker="o")

    if len(outliers) > 0:
        plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', edgecolors='r', s=60, alpha=0.8, marker="x")

    plt.title('One-Class SVM Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def train_one_class_svm(reduced_data):
    """
    Train a One-Class SVM on the reduced data from PCA.
    """
    aggregateData = []
    for sensor_data in reduced_data.values():
        aggregateData.extend(sensor_data)

    if not aggregateData:
        print("No reduced data available for training.")
        return None, None  # No model or scaler can be created without data

    aggregateData_np = np.array(aggregateData)

    # Create a scaler instance
    scaler = StandardScaler()
    aggregateData_np_scaled = scaler.fit_transform(aggregateData_np)

    # One-Class SVM
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)  # Hyperparameters
    model.fit(aggregateData_np_scaled)

    return model, scaler

def train_one_class_svm_with_dtw():
    # Fetch the DTW features from your other script
    dtw_features = calculate_dtw_features()

    if dtw_features.size == 0:
        print("No DTW features available for training.")
        return None, None

    # Standardize the features
    scaler = StandardScaler()
    dtw_features_scaled = scaler.fit_transform(dtw_features)

    # Train the One-Class SVM model
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)  # Hyperparameters
    model.fit(dtw_features_scaled)

    return model, scaler

def visualize_one_class_svm_with_dtw(model, scaler, dtw_features):
    if dtw_features.size == 0:
        print("No DTW features available for visualization.")
        return

    # Standardize the features using the same scaler used for training
    dtw_features_scaled = scaler.transform(dtw_features)

    # Predict using the One-Class SVM model
    predictions = model.predict(dtw_features_scaled)  # Outliers are -1, inliers are 1

    # Separate inliers and outliers for plotting
    inliers = dtw_features_scaled[predictions == 1]
    outliers = dtw_features_scaled[predictions == -1]

    # Plotting the inliers and outliers
    plt.figure()

    if inliers.size > 0:
        plt.scatter(inliers[:, 0], inliers[:, 1], label='Inliers', edgecolors='b', s=60, alpha=0.8, marker="o")

    if outliers.size > 0:
        plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', edgecolors='r', s=60, alpha=0.8, marker="x")

    plt.title('One-Class SVM Results with DTW Features')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def main():
    
    # Get the DTW features
    dtw_features = calculate_dtw_features()

    # Train the new SVM model with DTW features
    model_dtw, scaler_dtw = train_one_class_svm_with_dtw()

    if model_dtw is not None and scaler_dtw is not None:
        # Visualize the results of the new model
        visualize_one_class_svm_with_dtw(model_dtw, scaler_dtw, dtw_features)
    else:
        print("Could not train the model due to insufficient data.")
    reduced_data_dict, _ = get_reduced_data()

    model, scaler = train_one_class_svm(reduced_data_dict)

    if model is not None and scaler is not None:
        visualize_one_class_svm(model, scaler, reduced_data_dict)
    else:
        print("Could not train the model due to insufficient data.")

if __name__ == "__main__":
    main()
