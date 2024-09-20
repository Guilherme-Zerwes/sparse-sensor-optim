import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import filtfilt, butter
import ar

class Filter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        '''
        Creates a lowpass filter at 80hz for all sensors. 
        Input: ndarray with shape (n_sensors, n_samples).
        Output: filtered ndarray with shape (n_sensors, n_samples).
        '''
        return self
    
    def transform(self, X):
        b, a = butter(2, 80, 'low', fs=1024)

        for i in range(X.shape[0]):
            X[i,:] = filtfilt(b, a, X[i, :], padlen=150)
        return X
    
class Reconstruct(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.recon_mat = np.loadtxt('recon.txt')
        self.samp_mat = np.loadtxt('sensors.txt', dtype=int)

    def fit(self, X, y=None):
        '''
        Samples and reconstructs the dataset.
        Input: ndarray with shape (n_sensors, n_samples).
        Output: reconstructed ndarray with shape (n_sensors, n_samples).
        '''
        return self
    
    def transform(self, X):
        x_sampled = X[self.samp_mat]
        x_hat = np.dot(self.recon_mat, x_sampled)
        return x_hat
    
class Window(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        '''
        Divides the data into windows to generate more datapoints for the classification model.
        Input: ndarray with shape (n_sensors, n_samples).
        Output: ndarray with shape (n_windows, n_samples, n_sensors)
        '''
        return self
    
    def transform(self, X):
        X = np.transpose(X)

        #Divide in nf frames
        nf = 128    #number of frames

        aux = X.shape[0] - int(X.shape[0]%nf) #Determines the number of entries removed from the end to ensure reshape
        X = X[0:aux,:]

        X = np.reshape(X, (nf, int(X.shape[0]/nf), X.shape[1]))
        #resulting vector is of the form (n_frames, n_samples, n_features)
        return X
    
class TrainAr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        '''
        Trains the AR model and returns the coeficients.
        Input: ndarray with shape (n_windows, n_samples, n_sensors)
        Output: ndarray of coeficients with shape (n_windows, n_coeficients*n_sensors)
        '''
        return self
    
    def transform(self, X):
        #AR model training
        nr = 30 #number of regressive coeficients (ie. lag number)
        coeficients = np.zeros((X.shape[0], X.shape[2], nr))
        # scores = np.zeros((y_train.shape[1]))

        for i in range(X.shape[0]):
            print(i)
            for j in range(X.shape[2]):
                model = ar.ar()
                model.fit(X[i,:,j], order=nr)
                coeficients[i,j] = model.coefs
            
        coeficients = np.reshape(coeficients, (X.shape[0], X.shape[2]*nr))
        return coeficients
