# PCA DTW and Socket Programming
To run, execute SVMModel.py.
This program collects data by creating a socket server. 
It takes in JSON objects in the form of POST requests. 
It then applies PCA to reduce the dimensionality. 
It performs Dynamic Time Warping to extract features from the data.
The one-class-SVM from Keras is then applied to the data with the results from the DTW algorithm as features.
