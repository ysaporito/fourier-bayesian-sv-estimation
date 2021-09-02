import numpy as np
import matplotlib.pylab as plt
import pystan
import corner

def EstimateProfileBayesianPystan(logPrice, V_hat, time, model, exact_param):
    '''
    PyStan estimation

    logPrice - log of spot price
    V_hat - estimated volatility
    time - observed times
    model (string) - SV model
    exact_param (dict) - true parameters
    '''

    dt = np.diff(time)
    dX = np.diff(logPrice)
    N = dX.shape[0]

    if model == 'Heston':
        dV = np.diff(V_hat)
        int_V = V_hat[:-1] * dt #0.5 * (V_hat[:-1] + V_hat[1:]) * dt # (trapezoidal rule, if desired)

    elif model == 'Exp-OU':
        exp_v = np.exp(V_hat)
        exp_2v = np.exp(2*V_hat)
        dV = np.diff(V_hat)
        int_eV2 = 0.5 * (exp_2v[:-1] + exp_2v[1:]) * dt
        int_eV = 0.5 * (exp_v[:-1] + exp_v[1:]) * dt
        int_VeV = 0.5 * (V_hat[:-1]*exp_v[:-1] + V_hat[1:]*exp_v[1:]) * dt
        int_VdV = exp_v[:-1] * dV

    elif model == 'GARCH':
        vol = np.sqrt(V_hat)
        over_vol = 1.0/vol
        dV = np.diff(V_hat)
        int_1_over_vol = 0.5 * (over_vol[:-1] + over_vol[1:]) * dt
        int_1_over_voldV = over_vol[:-1] * dV
        int_vol = 0.5 * (vol[:-1] + vol[1:]) * dt
        int_V = 0.5 * (V_hat[:-1] + V_hat[1:]) * dt

    if model == 'Heston':

        code = """
        data {
          int<lower=0> N; 
          vector[N] dX; 
          vector[N] dt; 
          vector[N] dV; 
          vector[N] int_V;
          vector[N] sqrt_int_V;
        }
        parameters {
          real<lower=-1,upper=1> rho;
          real<lower=0, upper=20> kappa;
          real<lower=0, upper=2.0> xi;
          real<lower=0, upper=0.1> m;
          real<lower=-1,upper=1> mu;
        }
        transformed parameters {
            real beta1;
            real beta2;
            real beta3;
            real sigma;

            sigma = sqrt(1 - rho * rho);

            beta3 = rho / xi;
            beta2 = kappa * beta3 ;
            beta1 = mu - beta3 * kappa * m;
            }
        model {
            for (n in 1:N)
                dX[n] ~ normal(beta1 * dt[n] + beta2 * int_V[n] + beta3 * dV[n], sigma * sqrt_int_V[n]);
        }
        """

        data = {'N': N,
                'dX': dX,
                'dt': dt,
                'dV': dV,
                'int_V': int_V,
                'sqrt_int_V': np.sqrt(int_V)}

    elif model == 'Exp-OU':

        code = """
        data {
          int<lower=0> N; 
          vector[N] dX; 
          vector[N] dt;
          vector[N] int_eV2; 
          vector[N] int_VeV;
          vector[N] int_eV;
          vector[N] int_VdV;
          vector[N] sqrt_int_eV2;
        }
        parameters {
          real<lower=-1,upper=1> rho;
          real<lower=0, upper=20> kappa;
          real<lower=0, upper=2.0> xi;
          real<lower=0, upper=0.1> m;
          real<lower=-1,upper=1> mu;
        }
        transformed parameters {
            real beta1;
            real beta2;
            real beta3;
            real sigma;

            sigma = sqrt(1 - rho * rho);

            beta3 = rho / xi;
            beta2 = kappa * beta3;
            beta1 = beta2 * m;
            }
        model {
            for (n in 1:N)
                dX[n] ~ normal(mu * dt[n] - beta1 * int_eV[n] + beta2 * int_VeV[n] + beta3 * int_VdV[n], sigma * sqrt_int_eV2[n]);
        }
        """

        data = {'N': N,
                'dX': dX,
                'dt': dt,
                'int_eV2': int_eV2,
                'int_VeV': int_VeV,
                'int_eV': int_eV,
                'int_VdV': int_VdV,
                'sqrt_int_eV2': np.sqrt(int_eV2)}

    elif model == 'GARCH':

        code = """
        data {
          int<lower=0> N;
          vector[N] dX;
          vector[N] dt;
          vector[N] int_1_over_vol;
          vector[N] int_1_over_voldV;
          vector[N] int_vol;
          vector[N] int_V;
          vector[N] sqrt_int_V;
        }
        parameters {
          real<lower=-1,upper=1> rho;
          real<lower=0, upper=20> kappa;
          real<lower=0, upper=2.0> xi;
          real<lower=0, upper=0.1> m;
          real<lower=-1,upper=1> mu;
        }
        transformed parameters {
            real beta1;
            real beta2;
            real beta3;
            real sigma;

            sigma = sqrt(1 - rho * rho);

            beta3 = rho / xi;
            beta2 = kappa * beta3;
            beta1 = beta2 * m;
            }
        model {
            for (n in 1:N)
                dX[n] ~ normal(mu * dt[n] - beta1 * int_1_over_vol[n] + beta2 * int_vol[n] +
                beta3 * int_1_over_voldV[n], sigma * sqrt_int_V[n]);
        }
        """

        data = {'N': N,
                'dX': dX,
                'dt': dt,
                'int_1_over_vol': int_1_over_vol,
                'int_1_over_voldV': int_1_over_voldV,
                'int_vol': int_vol,
                'int_V': int_V,
                'sqrt_int_V': np.sqrt(int_V)}

    stan_model = pystan.StanModel(model_code=code)
    fit = stan_model.sampling(data=data, iter=1000, chains=4, seed=21) #seed fixed for reproducibility
    result = fit.extract(permuted=True)

    plt.figure(figsize=(8.0, 5.0))
    if model in ['Exp-OU', 'GARCH']:
        samples = np.array([result['rho'], result['xi'], result['kappa'], result['m'], result['mu']]).T

        corner.corner(samples, labels=[r'$\rho$', r'$\xi$', r'$\kappa$', r'$m$', r'$\mu$'],
                      truths=[exact_param['rho'][0], exact_param['xi'][0], exact_param['kappa'][0],
                                exact_param['m'][0], exact_param['mu'][0]], quantiles=[0.05, 0.5, 0.95], 
                      show_titles=True)

    else:
        samples = np.array([result['rho'], result['xi'], result['kappa']]).T

        corner.corner(samples, labels=[r'$\rho$', r'$\xi$', r'$\kappa$'],
                      truths=[exact_param['rho'][0], exact_param['xi'][0], exact_param['kappa'][0]],
                      quantiles=[0.05, 0.5, 0.95], show_titles=True)


    plt.savefig('Plot/estimation_bayes_' + model + '.pdf')