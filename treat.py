import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
import utils

filt = utils.Filter()
rec = utils.Reconstruct()
window = utils.Window()
coef_train = utils.TrainAr()

feature_pipe = Pipeline(steps=[('Filter', filt),
                               ('Reconstruction', rec),
                               ('Windowing', window),
                               ('ArTraining', coef_train)])

path = '../Dataset/Teste dataset'
files = sorted(os.listdir(path), key=lambda x: int(x[5:-4]))

for i in range(len(files)):
    file_name = os.path.join(path, files[i])
    #Loading the dataset
    df = np.transpose(np.asarray(
        pd.read_table(file_name, sep='\t', skiprows=10))[:, 1:])
    
    coeficients = feature_pipe.fit_transform(df)
    coef_name = files[i][:-4] + '.txt'
    coef_name = os.path.join('./outputs', 'coefsB', coef_name)
    np.savetxt(coef_name,coeficients)