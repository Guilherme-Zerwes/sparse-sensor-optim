import numpy as np

#Autoregressive model
class ar:
    def __init__(self, order=0, coefs=0):
        '''
        Autoregressive model
        '''
        self.order = order
        self.coefs = coefs
        return None
    
    def fit(self, X, order:int):
        '''
        Trains the Autoregressive model. Where X is a time series of shape (n_samples,) and order is the order, 
        or the number of lags, for the AR model. This performs a linear regression to find the coeficients
        to solve the linear system.
        '''
        X_train = np.zeros((X.shape[0] - order, order))
        Y_train = X[order:]
        for i in range(order):
            aux1 = (order-i-1)
            aux2 = -(i+1)
            X_train[:,i] = X[aux1:aux2]
        coefs = np.dot(np.linalg.pinv(X_train), Y_train)
        self.order = order
        self.coefs = coefs

    def predict(self, X):
        '''
        Returns a prediction y_pred of values given an input series X.
        '''
        X_train = np.zeros((X.shape[0] - self.order, self.order))
        for i in range(self.order):
            aux1 = (self.order - i - 1)
            aux2 = -(i + 1)
            X_train[:, i] = X[aux1:aux2]
        return np.dot(X_train, self.coefs)
    
    def score(self, X):
        '''
        Returns the R² score given an input series X and its true values Y.
        '''
        Y_true = X[self.order:]
        Y_pred = self.predict(X)
        u = ((Y_true - Y_pred)**2).sum()
        v = ((Y_true - Y_true.mean())**2).sum()
        score = 1 - u/v
        return score