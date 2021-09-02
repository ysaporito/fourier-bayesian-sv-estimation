import numpy as np
import matplotlib.pyplot as plt
import math

import SVmodel as SV
import FourierMethod as fourier
import Estimation as est

# parameters for example
param = {'mu': np.array([0.0]), 'kappa': np.array([5.0]), 'm': np.array([0.02]),
             'xi': np.array([0.5]), 'rho': np.array([-0.3])} 
S = np.array([1.0])
v = np.array([0.09])

def estimate(real, plot_vol, model, qqplot):
    '''
    real (boolean) - True to use the simulated vol process, False to use Fourier estimated vol process
    plot_vol (boolean) - True to save plot of volatility estimation
    model (string) - choose a SV model among Heston, InvGamma, Exp-OU or GARCH
    qqplot (boolean) - True to plot qqplot
    '''
    T = 1.0  # time horizon
    n = 2 ** 18  # fine grid
    nT = int(T * n)
    N = 2 ** 7 # it is related to n

    X, V = SV.Simulation(S, v, param, T, nT, model)
    time = np.linspace(0.0, T, 2 * N + 1)[1:-1]
    dt = time[1] - time[0]

    step = int(0.5*n/N)

    V_real = V[range(0 + step, n+1 - step, step)]
    X_real = X[range(0 + step, n+1 - step, step)]

    if model == 'Heston':
        f = lambda v: np.sqrt(v)
        g = lambda v: np.sqrt(v)
        phi = lambda v: param['mu'] - param['kappa'] * param['m'] * param['rho'] / param['xi'] + v * param['kappa'] * param['rho'] / param['xi']

    elif model == 'Exp-OU':
        f = lambda v: np.exp(v)
        g = lambda v: 1.0
        phi = lambda v: param['mu'] - param['kappa'] * param['m'] * param['rho'] * np.exp(v) / param['xi'] + v * np.exp(v) * param['kappa'] * param['rho'] /  param['xi']

    elif model == 'GARCH':
        f = lambda v: np.sqrt(v)
        g = lambda v: v
        phi = lambda v: param['mu'] - param['kappa'] * param['m'] * param['rho'] / (np.sqrt(v) * param['xi']) + np.sqrt(v) * param['kappa'] * param['rho'] /  param['xi']

    if not real:
        fV_hat = fourier.ComputeFourierEstimation(log_price=np.array([X]), time=T, N=N)
        if model == 'Heston':
            V_hat = fV_hat**2
        elif model == 'InvGamma':
            V_hat = fV_hat
        elif model == 'Exp-OU':
            V_hat = np.log(fV_hat)
        elif model == 'GARCH':
            V_hat = fV_hat**2

        V_hat = V_hat[1:-1]

    else:
        V_hat = V_real

    if plot_vol:
        plt.figure(figsize=(8.0, 5.0))
        plt.plot(time, V_real, '#013E7D', lw=3, label='V')
        plt.plot(time, V_hat, '#068FCB', lw=1, label=r'$\widehat{V}$')
        plt.xlabel('Time')
        plt.ylabel('V')
        plt.legend()
        plt.savefig('Plot/plot_vol_' + model + '.pdf')

    if qqplot:
        eps_func = lambda v: (np.diff(X_real) - phi(v[:-1]) * dt - param['rho'] / param['xi'] * f(v[:-1])/g(v[:-1]) * np.diff(v)) / (f(v[:-1]) * np.sqrt(dt))
        eps = eps_func(V_real)
        eps_hat = eps_func(V_hat)

        eps.sort()
        eps_hat.sort()

        maxval = max(eps[-1],eps_hat[-1])
        minval = min(eps[0],eps_hat[0])

        plt.figure()
        plt.scatter(eps, eps_hat)
        plt.plot([minval,maxval],[minval,maxval],'k-')
        plt.savefig('Plot/qqplot_eps_' + model + '.pdf')
    
    est.EstimateProfileBayesianPystan(X[range(0 + step, n + 1 - step, step)], V_hat, time, model=model, exact_param=param)

if __name__ == '__main__':
    estimate(real=False, plot_vol=True, model='Heston', qqplot=False)