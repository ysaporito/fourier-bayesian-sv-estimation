import numpy as np
import matplotlib.pyplot as plt

import SVmodel as SV
import FourierMethod as fourier
import Estimation as est

# stochastic volatility parameters for the example shown in the paper
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
    n = 2 ** 18  # number of observed times to compute the Fourier estimation
    N = 2 ** 7 # number of Fourier modes; it is related to n

    X, V = SV.Simulation(S, v, param, T, n, model) #simulated paths
    time = np.linspace(0.0, T, 2*N + 1)[1:-1] #observed times for estimation (remove the inicial and final point because the Fourier estimation does not work well there); these times is when we compute the estimated volatility process
    dt = time[1] - time[0] # delta t

    step = int(0.5*n/N) # interger step where the volatility process with be estimated at (coaser than X and V)

    V_real = V[range(0 + step, n+1 - step, step)] # real volatility at coarser grid
    X_real = X[range(0 + step, n+1 - step, step)] # log-stock price at coarser grid

    # functions f, g and phi from the paper for each model
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
        fV_hat = fourier.FourierEstimation(np.array([X]), N, T) # compute the Fourier estimation of the volatility process
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
        V_hat = V_real # use the observed vol as the estimated one. Used to test the Bayesian estimation without noise from volatility estimation

    # plot volatility and its estimation
    if plot_vol:
        
        plt.figure(figsize=(8.0, 5.0))
        plt.plot(time, V_real, '#013E7D', lw=3, label='V')
        plt.plot(time, V_hat, '#068FCB', lw=1, label=r'$\widehat{V}$')
        plt.xlabel('Time')
        plt.ylabel('V')
        plt.legend()
        plt.savefig('Plot/plot_vol_' + model + '.pdf')

    # qq plot 
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
    
    # estimate the stochastic volatility model
    est.EstimateProfileBayesianPystan(X_real, V_hat, time, model=model, exact_param=param)

if __name__ == '__main__':
    estimate(real=False, plot_vol=True, model='GARCH', qqplot=False)