'''
This .py computes the Fourier estimation of the volatility process using Teichmann and Cucheiro's procedure
'''

import numpy as np

def FourierCoeff(log_price, n_of_fourier_coeff, time_horizon):

    # number of Fourier Coeff
    N = n_of_fourier_coeff

    # number of time steps
    nT = int(log_price.size/log_price.shape[0] - 1)

    # log_price[i+1] - log_price[i]
    dX = np.diff(log_price)

    n = int(nT/time_horizon)

    if log_price.shape[0] == 2:
        g_func = lambda y: np.array([[np.cos(y[0]), np.cos(y[0] + y[1])], [np.cos(y[0] + y[1]), np.cos(y[1])]])
    else:
        g_func = lambda y: np.array([np.cos(y)])

    g = g_func(np.sqrt(n)*dX)

    V = 1.0/n*np.fft.fft(g)

    fourier_coeff = np.concatenate((V[:, :, (nT-N):], V[:, :, :(N+1)]), axis=2)

    return fourier_coeff

def ComputeV0(log_price, n_of_fourier_coeff, func_type='cos'):

    # number of Fourier Coeff
    N = n_of_fourier_coeff

    # number of time steps
    n = log_price.size/log_price.shape[0] - 1

    # log_price[i+1] - log_price[i]
    dX = np.diff(log_price)

    if func_type == 'cos':
        if log_price.shape[0] == 2:
            g_func = lambda y: np.array([[np.cos(y[0]), np.cos(y[0] + y[1])], [np.cos(y[0] + y[1]), np.cos(y[1])]])
        else:
            g_func = lambda y: np.array([np.cos(y[0])])

    elif isinstance(func_type, int) or isinstance(func_type, tuple):
        if log_price.shape[0] == 2:
            r = func_type[0]
            s = func_type[1]
            g_func = lambda y: np.array([[np.abs(y[0])**r, np.abs(y[0])**r*np.abs(y[1])**s], [np.abs(y[0])**r*np.abs(y[1])**s, np.abs(y[1])**s]])
        else:
            r = func_type[0]
            g_func = lambda y: np.array([np.abs(y[0])**r])

    g = g_func(np.sqrt(n)*dX)

    V = 1.0/n*np.sum(g, axis=2)

    #V0 = V[:,:,0]
    V0 = V

    return V0

def FourierEstimation(log_price, n_of_fourier_coeff, time_horizon):

    # number of Fourier Coeff
    N = n_of_fourier_coeff

    F = FourierCoeff(log_price, n_of_fourier_coeff, time_horizon)

    # Fourier coeff index
    k = np.arange(-N, N+1)

    f = F*(1.0 - np.abs(k)/float(N))

    Fejersum = (2*N+1)*np.exp(-1j*2*np.pi*N*(k+N)/(2*N+1))*np.fft.ifft(f) # k+N is m here (the time)

    Xhat = -2*np.log(1.0/time_horizon*np.real(Fejersum))

    Xhat = np.sqrt(Xhat[0][0])

    return Xhat

def ComputeFourierEstimation(log_price, time, N=None):

    if N is None:
        n = log_price.size / log_price.shape[0] - 1
        N = int(np.sqrt(n))

    fV_hat = FourierEstimation(log_price, N, time)

    return fV_hat