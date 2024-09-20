import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

#Load all coeficients
path = './outputs/coefsB'
files = sorted([file for file in os.listdir(path) if file.endswith('.txt')], key=lambda x: int(x[5:-4]))

#Input parameters
n_window = 128

nr = 30
n_sensors = 30
n_cases = len(files)

x_data = np.zeros((len(files)*n_window, nr*n_sensors))
y_data = np.zeros((len(files)*n_window))

#Loads to x_data variable
print('Loading files...')
for i in range(len(files)):
    file_path = os.path.join(path,files[i])
    aux_in = i*n_window
    aux_end = (i+1)*n_window
    x_data[aux_in:aux_end] = np.loadtxt(file_path)
    y_data[aux_in:aux_end] = i 


#Shuffle data
seed = np.random.randint(1,1000)
np.random.seed(seed)
np.random.shuffle(x_data)
np.random.seed(seed)
np.random.shuffle(y_data)

#Model prediction
model = load('./outputs/modelos/classifierRec.joblib')
print('Making predictions...')
score = model.score(x_data,y_data)
preds = model.predict_proba(x_data)

print(f'R² score: {score}')

#Visualize the results

for i in range(len(y_data)):
    heat = np.reshape(preds[i,1:], (6,5))
    fig = plt.figure(i)
    plt.imshow(heat, interpolation='none', cmap='jet', vmin=0, vmax=1)
    plt.grid()
    plt.colorbar(extend='both')
    title = f'PoD distribution [damage in {y_data[i]}]'
    plt.title(title)
    plt.xlabel('Sensors in x')
    plt.ylabel('Sensors in y')
    # plt.savefig(f'./outputs/graficos/validation heatmap {i}.png')
    plt.show()