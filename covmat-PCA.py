import pandas as pd
import numpy as np
import flasksocketserver
import json
import matplotlib.pyplot as plt

datacsv = pd.read_csv('20231009102957.csv')

x_values = []
y_values = []
z_values = []

for index, row in datacsv.iterrows():
    payloads = json.loads(row['payload'].replace("'", '"'))
    for payload in payloads:
        if payload['name'] == 'accelerometer':
            x_values.append(payload['values']['x'])
            y_values.append(payload['values']['y'])
            z_values.append(payload['values']['z'])

x = np.array(x_values)
y = np.array(y_values)
z = np.array(z_values)

data = np.vstack((x, y, z))

print(data)

cov_matrix = np.cov(data)

print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) 

eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

variance = eigenvalues / np.sum(eigenvalues)

plt.bar(range(len(eigenvalues)), variance)
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance')
plt.show()

projected_data = np.dot(data.T, eigenvectors[:, :2])

scale_factor = 0.7  # this is to scale the length of the arrow, adjust as needed
for i in range(2):  # since we have 2 principal components in 2D
    plt.arrow(0, 0, 
              eigenvectors[i, 0]*eigenvalues[i]*scale_factor, 
              eigenvectors[i, 1]*eigenvalues[i]*scale_factor,
              head_width=0.2, head_length=0.3, fc='black', ec='black')

plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Sensor Data')
plt.show()

