import numpy as np

def estimate_delta(ww_local, H_local, Omega):
    index_Omega = np.where(ww_local == Omega)[0][0]
    H_omega = H_local[index_Omega,]
    Den = []
    Num = []
    ww_delta = []
    for index in range(0,len(ww_local)):
        if index == index_Omega:
            continue
        else:
            Den.append(H_local[index,] - H_omega)
            Num.append(ww_local[index,]**2 - Omega**2)
            ww_delta.append(ww_local[index,])
    Den = np.asarray(Den, dtype = 'complex_')
    Num = np.asarray(Num, dtype = 'complex_')
    ww_delta = np.asarray(ww_delta)
    delta = Num.reshape(-1,)/Den.reshape(-1,)
    # print(delta)
    return ww_delta, delta

def moindre_carre(x, y):
    N = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x*x)
    Sxy = np.sum(x*y)
    b = (Sxx*Sy - Sx*Sxy)/(N*Sxx-Sx**2)
    a = (N * Sxy - Sx * Sy) / (N * Sxx - Sx**2)
    return a, b

def line_fit(H,freq,fmin,fmax):
    # H: a FRF a ser estudada
    # freq: vetor frequência em Hz
    # fmin: frequência mínima do intervalo que se deseja analisar
    # fmax: frequência máxima do intervalo que se deseja analisar 

    temp = np.argwhere(freq>fmin)  # Find the indices of array elements that are non-zero, grouped by element.
    index_low = temp[0,0]-1 
    temp = np.argwhere(freq>fmax)  # Find the indices of array elements that are non-zero, grouped by element.
    index_high = temp[0,0]


    H_local = H[index_low:index_high]
    freq_local = freq[index_low:index_high]
    ww_local = 2 * np.pi * freq_local

    # In the case of insufficient number of samples
    N_pts = len(freq_local)
    Delta = np.zeros((N_pts-1, N_pts), dtype = 'complex_')
    ww_Delta = np.zeros((N_pts-1, N_pts))

    # Dobson's method application
    for ind in range(0,N_pts):
        ww_delta, delta = estimate_delta(ww_local, H_local, ww_local[ind])
        Delta[:, ind] = delta
        ww_Delta[:, ind] = ww_delta

    tr = np.zeros((N_pts,))
    cr = np.zeros((N_pts,))
    ti = np.zeros((N_pts,))
    ci = np.zeros((N_pts,))

    # Best straight line finding
    for index in range(N_pts):
        a, b = moindre_carre(ww_Delta[:, index]**2, np.real(Delta[:, index]))
        tr[index,] = a
        cr[index,] = b
        a, b = moindre_carre(ww_Delta[:, index]**2, np.imag(Delta[:, index]))
        ti[index,] = a
        ci[index,] = b

    ur, dr = moindre_carre(ww_local**2, tr)
    ui, di = moindre_carre(ww_local**2, ti)
    p = ui / ur
    q = di / dr

    # Modal parameters calculation
    loss_factor = (q - p) / (1 + p * q)
    wr = np.sqrt(dr / ((p * loss_factor - 1) * ur))
    ar = -wr**2 * (p * loss_factor - 1) / ((1 + p**2) * dr)
    br = ar * p
    B = ar + 1j * br

    # loss_factor/2: fator de amortecimento
    # wr: frequência natural 
    # B: constante modal para o modo estudado (muda conforme mudo a posição entrada-saída)
    
    return(loss_factor/2,wr,B)