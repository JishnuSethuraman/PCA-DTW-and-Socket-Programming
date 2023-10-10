import pandas as pd
import numpy as np
import HW2.flasksocketserver as flasksocketserver
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
eigenvectors = eigenvectors[:, ::-1] #yurr

