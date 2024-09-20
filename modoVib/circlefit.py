import numpy as np

def modify_ref_angle(theta_brut):
    theta = 2*np.pi-theta_brut 
    theta_ref = theta[0,0] 
    theta = theta-theta_ref 
    theta = np.mod(theta,2*np.pi) 
    return(theta,theta_ref)

def interpol(f1,theta1,f2,theta2):
    f_interpol = (f2+f1)/2 
    a = (theta2-theta1)/(f2-f1) 
    b = theta2-np.dot(a,f2)
    theta_interpol = np.dot(a,f_interpol)+b 
    return(f_interpol,theta_interpol)

def err_fit_circle(x,y):
    L = len(x) 
    SXX = np.sum(x**2)
    SYY = np.sum(y**2)
    SXY = np.sum(x*y)  
    SX = np.sum(x) 
    SY = np.sum(y) 
    xtemp = x**2 ; 
    ytemp = y**2 ; 
    SXXX = np.sum(xtemp*x)  
    SYYY = np.sum(ytemp*y)  
    SXXY = np.sum(xtemp*y)  
    SXYY = np.sum(x*ytemp)    
    A = np.array([[SXX,SXY,SX],[SXY, SYY, SY], [SX, SY, L]]) 
    B = np.array(([[-(SXXX+SXYY)] , [-(SXXY+SYYY)] , [-(SXX+SYY)]]))
    X = np.dot(np.linalg.pinv(A),B) 
    a = X[0] 
    b = X[1] 
    c = X[2] 
    x0 = -a/2 
    y0 = -b/2 
    R0 = np.sqrt(a**2/4+b**2/4-c) 
    return(x0,y0,R0)

def  estimate_natural_frequency(ind_max,delta2,f_delta2,theta_delta2):
    fr = []
    theta_r = []
    for jj in range(0,len(ind_max)):
        xa = f_delta2[ind_max[jj]-1] 
        xb = f_delta2[ind_max[jj]]
        ya = delta2[ind_max[jj]-1]
        yb = delta2[ind_max[jj]]
        a = (yb-ya)/(xb-xa) 
        b = yb-a*xb
        P = np.array([a,b]).reshape(-1,)
        aux = np.roots(P) 
        print(delta2)
        fr.append(aux[0])
        f1 = f_delta2[ind_max[jj]-1]
        f2 = f_delta2[ind_max[jj]]
        theta1 = theta_delta2[ind_max[jj]-1]
        theta2 = theta_delta2[ind_max[jj]]
        a = (theta2-theta1)/(f2-f1) 
        b = theta2-a*f2 
        theta_r.append(a*fr[jj]+b)
    return(np.asarray(fr),np.asarray(theta_r))

def circle_fit(H,freq,fmin,fmax):
    # H: a FRF a ser estudada
    # freq: vetor frequência em Hz
    # fmin: frequência mínima do intervalo que se deseja analisar
    # fmax: frequência máxima do intervalo que se deseja analisar 
    temp = np.argwhere(freq>fmin)  # Find the indices of array elements that are non-zero, grouped by element.
    index_low = temp[0,0]-1 
    temp = np.argwhere(freq>fmax)  # Find the indices of array elements that are non-zero, grouped by element.
    index_high = temp[0,0]

    #% Data extraction
    H_local = H[index_low:index_high] 
    freq_local = freq[index_low:index_high] 
    ww_mode = 2*np.pi*freq_local 
    x = np.real(H_local) 
    y = np.imag(H_local) 
    N_pts = len(freq_local)

    #% Best circle finding
    x0,y0,R0 = err_fit_circle(x,y) 
    theta_brut = np.arctan2(y-y0,x-x0) 
    theta_brut = np.mod(theta_brut,2*np.pi) # Checar se funcionou em python (Returns the element-wise remainder of division)
    [theta,theta_ref] = modify_ref_angle(theta_brut) 

    delta = []
    f_delta = []
    theta_delta = []
    for jj in range (0,N_pts-1):
        delta.append(theta[jj+1]-theta[jj])  
        [f_interpol,theta_interpol] = interpol(freq_local[jj],theta[jj],freq_local[jj+1],theta[jj+1]) 
        f_delta.append(f_interpol)
        theta_delta.append(theta_interpol)

    delta = np.asarray(delta)
    f_delta = np.asarray(f_delta)
    theta_delta = np.asarray(theta_delta)

    delta2 = []
    f_delta2 = []
    theta_delta2 = []
    for jj in range (0,N_pts-2):
        delta2.append(delta[jj+1]-delta[jj])
        [f_interpol,theta_interpol] = interpol(f_delta[jj],theta_delta[jj],f_delta[jj+1],theta_delta[jj+1]) 
        f_delta2.append(f_interpol)
        theta_delta2.append(theta_interpol)

    delta2 = np.asarray(delta2)
    f_delta2 = np.asarray(f_delta2)
    theta_delta2 = np.asarray(theta_delta2)

    #% Local and overall maximum finding
    sign_ref = np.sign(delta2[0]) 
    kk = 0 

    ind_zero = [] 
    for index in range(0,len(delta2)):
        if np.sign(delta2[index]) == -sign_ref: 
            kk = kk+1 
            sign_ref = -sign_ref 
            ind_zero.append(index) 
    ind_zero = np.asarray(ind_zero)

    #Resonance calculation
    [fr, theta_r] = estimate_natural_frequency(ind_zero,delta2,f_delta2,theta_delta2) 
    ss = np.argwhere(delta==np.max(delta)) 
    kkk = np.argwhere(ind_zero==ss)[0,0] 
    

    eig_theta = theta_r[kkk]
    eig_frequency = fr[kkk] 
    wr = 2*np.pi*eig_frequency

    #% Damping (loss factor) calculation
    temp = np.argwhere(freq_local>eig_frequency) 
    k1 = temp[0,0]-1 
    k2 = temp[0,0] 
    wb = freq_local[k1:0:-1]*2*np.pi  
    wa = freq_local[k2:N_pts]*2*np.pi  
    theta_b = eig_theta-theta[k1:0:-1]
    theta_a = theta[k2:N_pts]-eig_theta
    xi = []
    for jj in range(0,len(wb)):
        for kk in range(0,len(wa)): 
            if theta_b[jj] > np.pi:
                continue
            elif theta_a[kk] > np.pi:
                continue
            N = (wa[kk]**2-wb[jj]**2) 
            D = wr**2*(np.tan(theta_a[kk]/2)+ np.tan(theta_b[jj]/2)) 
            xi.append(N/D) 
    xi = np.asarray(xi)
    loss_factor = np.mean(xi) 

    #% Modal constant calculation
    ref = 2*np.pi-(theta_r[kkk]+theta_ref) 
    xa = R0*np.cos(ref)+x0 
    ya = R0*np.sin(ref)+y0 
    phi = -np.arctan2(xa-x0,ya-y0) 
    Bmod = 2*R0*loss_factor*wr*wr 
    B = complex(Bmod*np.cos(phi),Bmod*np.sin(phi)) 

    # loss_factor/2: fator de amortecimento
    # wr: frequência natural 
    # B: constante modal para o modo estudado (muda conforme mudo a posição entrada-saída)
    return(loss_factor/2,wr,B)