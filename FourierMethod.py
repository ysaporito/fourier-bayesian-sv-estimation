import numpy as np
'''
This .py computes the Fourier estimation of the volatility process using Cucheiro and Teichmann's procedure, see https://arxiv.org/abs/1301.3602
'''

# in the functions below: T is the time horizon, 2*N+1 the number of Fourier coefficients and X is the observed process that we want to compute the volatility of it

def FourierCoeff(X, N, T):

    # number of time steps
    nT = int(X.size/X.shape[0] - 1)

    # number of simulated times in each time horizon
    n = int(nT/T)

    # return of log price
    dX = np.diff(X)    

    # g transformation from Cucheiro and Teichmann. We use the cossine transform
    if X.shape[0] == 2:
        g_func = lambda y: np.array([[np.cos(y[0]), np.cos(y[0] + y[1])], [np.cos(y[0] + y[1]), np.cos(y[1])]])
    else:
        g_func = lambda y: np.array([np.cos(y)])

    # transformation of dX
    g = g_func(np.sqrt(n)*dX)

    # Fourier coefficients of the transformed increments dX
    V = (1.0/n)*np.fft.fft(g)

    # organize the Fouier coefficients -N,...,0,....N
    fourier_coeff = np.concatenate((V[:, :, (nT-N):], V[:, :, :(N+1)]), axis=2)

    return fourier_coeff

def FourierEstimation(X, N, T):

    # Fourier coefficients
    F = FourierCoeff(X, N, T)

    # Fourier coeff index
    k = np.arange(-N, N+1)

    # Fejer kernel
    f = F*(1.0 - np.abs(k)/float(N))

    # Fejer sum; k+N is the index in the sum
    Fejersum = (2*N+1)*np.exp(-1j*2*np.pi*N*(k+N)/(2*N+1))*np.fft.ifft(f) 

    # estimated volatility; the -2log comes from the choice of the g function
    Vhat = -2*np.log(1.0/T*np.real(Fejersum))
    Vhat = np.sqrt(Vhat[0][0])

    return Vhat