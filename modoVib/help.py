import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft
import circlefit
import linefit

# Definimos os parâmetros amostrais
fs = 1024 # taxa de amostragem
tf = 256 # tempo total 
Np = fs*tf # número de pontos
t = np.linspace(0,tf,Np,endpoint=False)

df = np.asarray(pd.read_table('..\..\Dataset\Treino dataset\zzzAA0.TXT', sep='\t', skiprows=10))
t = df[:,0:]
df = df[:,1:]

w, H18 = signal.csd(df[:,8], df[:,8], fs=fs, window='hann', nperseg=int(df.shape[0]/64), 
                nfft=int(df.shape[0]/64), detrend='constant', return_onesided=True, scaling='density', 
                axis=-1, average='mean')

# w, H11 = signal.csd(df[:,1], df[:,1], fs=fs, window='hann', nperseg=int(df.shape[0]/64), 
#                 nfft=int(df.shape[0]/64), detrend='constant', return_onesided=True, scaling='density', 
#                 axis=-1, average='mean')

plt.semilogy(w, abs(H18*1j*w), label='Sensor 1')
plt.semilogy(w, abs(H18/(1j*w*2*np.pi)**2), label='Sensor 8')
plt.xlabel('Frequency (Hz)')
plt.ylabel('H18 [m/s²/N]')
plt.grid()
plt.legend()
plt.show()

zeta1,wn1,A18_1 = linefit.line_fit((H18/(1j*w*2*np.pi)**2).reshape(-1,1),w,15,20)
print(zeta1,wn1,A18_1)
# zeta1,wn1,B11_1 = circlefit.circle_fit((H18/(1j*w*2*np.pi)).reshape(-1,1),w,15,35)