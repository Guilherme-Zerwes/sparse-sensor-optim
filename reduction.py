import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.linalg import qr
from scipy.signal import welch, filtfilt, butter
from scipy.linalg import svd

def lineplot(xdata, ydata, xlabel:str ='', ylabel:str ='', legend:str ='', title:str ='') -> None:
        plt.plot(xdata,ydata,label=legend)
        plt.grid()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

files = sorted(os.listdir('../Dataset/Treino dataset'), key=lambda x: int(x[5:-4]))

df = np.transpose(
      np.asarray(
            pd.read_table(
                os.path.join('..\Dataset\Treino dataset', files[0]), sep='\t', skiprows=10)))

t = df[0, :]
df = df[1:, :]

print('Filtering...')
b, a = butter(20, 100, 'low', fs=1024)

for i in range(df.shape[0]):
        df[i,:] = filtfilt(b, a, df[i, :], padlen=150)


for l in range(len(files) - 1):
    print(files[l+1][:-4])
    fileName = os.path.join('..\Dataset\Treino dataset', files[l+1])

    #Loading the dataset
    df2 = np.transpose(
          np.asarray(pd.read_table(fileName, sep='\t', skiprows=10))[:, 1:])

    #Filtering the data with a order 2 lowpass filter at 80 Hz
    print('Filtering...')
    b, a = butter(20, 100, 'low', fs=1024)


    for i in range(df2.shape[0]):
        df2[i,:] = filtfilt(b, a, df2[i, :], padlen=150)

    df = np.concatenate((df, df2), axis=1)
    print(df.shape)
    
#Performing the SVD
print('Performing SVD...')
Psi, S, V = svd(df, full_matrices=False)
print(Psi.shape, S.shape, V.shape, S)
lineplot(np.arange(0,len(S),1), S, 'Rank number', 'Singular values', title='Singular Values x Rank')

#Determining optimal hard threshold for singular values
beta = df.shape[0]/df.shape[1]
omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

threshold_sv = omega*np.median(S)

print(S[S>=threshold_sv].shape)

# scores = np.zeros((S.shape[0]))

#Reducing number of basis
rank = S[S >= threshold_sv].shape[0]
print(f'Rank {rank}')

Psi_r = Psi[:,0:rank]

#Finding near-optimal sampling points
Q, R, pivots = qr(np.transpose(Psi_r), pivoting=True)

aux = int(rank - df.shape[0])
pivots = pivots[:aux]
print('Selected sensors: ', pivots)
#Fill in sampling matrix C
C = np.zeros((rank,df.shape[0]))

for i in range(rank):
        C[i,pivots[i]] = 1

#Defining the reconstruction operator
recon = np.dot(Psi_r,np.linalg.inv(np.dot(C,Psi_r)))

#Sample the data
y = df2[pivots, :]

#Reconstruct the data
x_hat = np.dot(recon,y)

#Evaluate error
res = np.linalg.norm(df2[:,:] - x_hat, 'fro')/np.linalg.norm(df2,'fro')
# scores = res
print(res)

# np.savetxt('./recon.txt',recon)
# np.savetxt('./sensors.txt',pivots)
# lineplot(np.arange(0,30,1), scores, 'Number of sensors', 'Aproximation error', 
#         title='Reconstruction Aproximation error x Number of sensors')

#Evaluate reconstruction
plt.plot(t[:], df2[10,:], label='Original data')
plt.plot(t[:], x_hat[10,:], label='Reconstructed data')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title(f'Reconstruction comparison ({len(pivots)} sensors)')
plt.show()

#FRF Evaluation
nperseg_frf = len(t)//128
f, amp = welch(df2[10,:], fs=1024.0, window='hann', nperseg=nperseg_frf)

f2, amp2 = welch(x_hat[10,:], fs=1024.0, window='hann', nperseg=nperseg_frf)


plt.semilogy(f,abs(amp), label='Original data')
plt.semilogy(f,abs(amp2), label='Reconstructed data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplification')
plt.title(f'FRF Reconstruction comparison ({len(pivots)} sensors)')
plt.grid()
plt.legend()
plt.show()