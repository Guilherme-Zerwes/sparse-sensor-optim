import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Load all coeficients
path = './outputs/coefsRec'
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

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size=0.8, random_state=42)

#Model training
print('Begining training...')
if 'classifierRec.joblib' not in os.listdir('./outputs/modelos'):
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    dump(model, './outputs/modelos/classifierRec.joblib')
else:
    model = load('./outputs/modelos/classifierRec.joblib')

print(f'Model score during training: {model.score(x_train,y_train)}')
print(f'Model score during testing: {model.score(x_test,y_test)}')
preds = model.predict_proba(x_test)

#Visualize the results

for i in range(len(y_test)):
    heat = np.reshape(preds[i,1:], (6,5))
    fig = plt.figure(i)
    plt.imshow(heat, interpolation='bilinear', cmap='jet', vmin=0, vmax=1)
    plt.grid()
    plt.colorbar(extend='both')
    title = f'PoD distribution [damage in {y_test[i]}]'
    plt.title(title)
    plt.xlabel('Sensors in x')
    plt.ylabel('Sensors in y')
    plt.savefig(f'./outputs/graficos/03 - Reconstrucao completa/validation heatmap damage in {y_test[i]}.png')
    plt.show()