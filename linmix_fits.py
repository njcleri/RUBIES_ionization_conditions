import numpy as np
import pandas as pd
import linmix
from nikkos_tools import stat_functions as sf
import data_management
import globals

line_fluxes = pd.read_csv(globals.RUBIES_DATA.joinpath('line_flux_df.csv'), index_col=0)
line_fluxes_prism = data_management.make_df_prism(line_fluxes)
line_fluxes_g395m = data_management.make_df_g395m(line_fluxes)

O32df_prism = data_management.signal_to_noise_O32_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O32df_g395m = data_management.signal_to_noise_O32_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O32df_both = pd.merge(O32df_prism, O32df_g395m, on='id', how='inner')

O32Hadf_prism = data_management.signal_to_noise_O32Ha_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O32Hadf_g395m = data_management.signal_to_noise_O32Ha_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O32Hadf_both = pd.merge(O32Hadf_prism, O32Hadf_g395m, on='id', how='inner')

R23df_prism = data_management.signal_to_noise_R23_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
R23df_g395m = data_management.signal_to_noise_R23_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
R23df_both = pd.merge(R23df_prism, R23df_g395m, on='id', how='inner')

R23Hadf_prism = data_management.signal_to_noise_R23Ha_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
R23Hadf_g395m = data_management.signal_to_noise_R23Ha_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
R23Hadf_both = pd.merge(R23Hadf_prism, R23Hadf_g395m, on='id', how='inner')

O3Hbdf_prism = data_management.signal_to_noise_O3Hb_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O3Hbdf_g395m = data_management.signal_to_noise_O3Hb_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O3Hbdf_both = pd.merge(O3Hbdf_prism, O3Hbdf_g395m, on='id', how='inner')

O3HbHadf_prism = data_management.signal_to_noise_O3HbHa_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O3HbHadf_g395m = data_management.signal_to_noise_O3HbHa_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O3HbHadf_both = pd.merge(O3HbHadf_prism, O3HbHadf_g395m, on='id', how='inner')

Ne3O2df_prism = data_management.signal_to_noise_Ne3O2_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2df_g395m = data_management.signal_to_noise_Ne3O2_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2df_both = pd.merge(Ne3O2df_prism, Ne3O2df_g395m, on='id', how='inner')

Ne3O2Hadf_prism = data_management.signal_to_noise_Ne3O2Ha_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2Hadf_g395m = data_management.signal_to_noise_Ne3O2Ha_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2Hadf_both = pd.merge(Ne3O2Hadf_prism, Ne3O2Hadf_g395m, on='id', how='inner')

Ne3O32df_prism = data_management.signal_to_noise_Ne3O32_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
Ne3O32df_g395m = data_management.signal_to_noise_Ne3O32_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
Ne3O32df_both = pd.merge(Ne3O32df_prism, Ne3O32df_g395m, on='id', how='inner')

def fit_OIIIHb_versus_redshift_linmix():
    x = pd.concat([O3Hbdf_prism.z_prism, O3Hbdf_g395m.z_g395m])
    y = pd.concat([np.log10(O3Hbdf_prism.OIII_Hb), np.log10(O3Hbdf_g395m.OIII_Hb)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(O3Hbdf_prism.OIII_Hb,O3Hbdf_prism.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(O3Hbdf_prism.OIII_Hb,O3Hbdf_prism.OIII_Hb_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O3Hbdf_g395m.OIII_Hb,O3Hbdf_g395m.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(O3Hbdf_g395m.OIII_Hb,O3Hbdf_g395m.OIII_Hb_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_OIIIHb_versus_redshift_less_than_5_linmix():
    df_prism = O3Hbdf_prism[O3Hbdf_prism.z_prism < 5]
    df_g395m = O3Hbdf_g395m[O3Hbdf_g395m.z_g395m < 5]
    x = pd.concat([df_prism.z_prism, df_g395m.z_g395m])
    y = pd.concat([np.log10(df_prism.OIII_Hb), np.log10(df_g395m.OIII_Hb)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(df_prism.OIII_Hb,df_prism.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(df_prism.OIII_Hb,df_prism.OIII_Hb_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(df_g395m.OIII_Hb,df_g395m.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(df_g395m.OIII_Hb,df_g395m.OIII_Hb_ERR_84)) / 2])
    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_OIIIHb_versus_redshift_greater_than_5_linmix():
    df_prism = O3Hbdf_prism[O3Hbdf_prism.z_prism > 5]
    df_g395m = O3Hbdf_g395m[O3Hbdf_g395m.z_g395m > 5]
    x = pd.concat([df_prism.z_prism, df_g395m.z_g395m])
    y = pd.concat([np.log10(df_prism.OIII_Hb), np.log10(df_g395m.OIII_Hb)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(df_prism.OIII_Hb,df_prism.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(df_prism.OIII_Hb,df_prism.OIII_Hb_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(df_g395m.OIII_Hb,df_g395m.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(df_g395m.OIII_Hb,df_g395m.OIII_Hb_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_NeIIIOII_versus_redshift_linmix():
    df_prism = Ne3O2df_prism[Ne3O2df_prism.z_prism > 0]
    df_g395m = Ne3O2df_g395m[Ne3O2df_g395m.z_g395m > 0]

    x = pd.concat([df_prism.z_prism, df_g395m.z_g395m])
    y = pd.concat([np.log10(df_prism.NeIII_OII), np.log10(df_g395m.NeIII_OII)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(df_prism.NeIII_OII,df_prism.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(df_prism.NeIII_OII,df_prism.NeIII_OII_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(df_g395m.NeIII_OII,df_g395m.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(df_g395m.NeIII_OII,df_g395m.NeIII_OII_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_O32_versus_redshift_linmix():
    x = pd.concat([O32df_prism.z_prism, O32df_g395m.z_g395m])
    y = pd.concat([np.log10(O32df_prism.O32), np.log10(O32df_g395m.O32)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(O32df_prism.O32,O32df_prism.O32_ERR_16) + sf.propagate_uncertainty_log10(O32df_prism.O32,O32df_prism.O32_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O32df_g395m.O32,O32df_g395m.O32_ERR_16) + sf.propagate_uncertainty_log10(O32df_g395m.O32,O32df_g395m.O32_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_R23_versus_redshift_linmix():
    x = pd.concat([R23df_prism.z_prism, R23df_g395m.z_g395m])
    y = pd.concat([np.log10(R23df_prism.R23), np.log10(R23df_g395m.R23)])
    xsig = np.zeros(len(x)) 
    ysig = pd.concat([(sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_16) + sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_16) + sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_OIIIHb_versus_Ha_linmix():
    x = pd.concat([np.log10(O3HbHadf_prism.L_Ha), np.log10(O3HbHadf_g395m.L_Ha)])
    y = pd.concat([np.log10(O3HbHadf_prism.OIII_Hb), np.log10(O3HbHadf_g395m.OIII_Hb)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(O3HbHadf_prism.L_Ha,O3HbHadf_prism.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(O3HbHadf_prism.L_Ha,O3HbHadf_prism.L_Ha_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O3HbHadf_g395m.L_Ha,O3HbHadf_g395m.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(O3HbHadf_g395m.L_Ha,O3HbHadf_g395m.L_Ha_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(O3HbHadf_prism.OIII_Hb,O3HbHadf_prism.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(O3HbHadf_prism.OIII_Hb,O3HbHadf_prism.OIII_Hb_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O3HbHadf_g395m.OIII_Hb,O3HbHadf_g395m.OIII_Hb_ERR_16) + sf.propagate_uncertainty_log10(O3HbHadf_g395m.OIII_Hb,O3HbHadf_g395m.OIII_Hb_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_NeIIIOII_versus_Ha_linmix():
    x = pd.concat([np.log10(Ne3O2Hadf_prism.L_Ha), np.log10(Ne3O2Hadf_g395m.L_Ha)])
    y = pd.concat([np.log10(Ne3O2Hadf_prism.NeIII_OII), np.log10(Ne3O2Hadf_g395m.NeIII_OII)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(Ne3O2Hadf_prism.L_Ha,Ne3O2Hadf_prism.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(Ne3O2Hadf_prism.L_Ha,Ne3O2Hadf_prism.L_Ha_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(Ne3O2Hadf_g395m.L_Ha,Ne3O2Hadf_g395m.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(Ne3O2Hadf_g395m.L_Ha,Ne3O2Hadf_g395m.L_Ha_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(Ne3O2Hadf_prism.NeIII_OII,Ne3O2Hadf_prism.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(Ne3O2Hadf_prism.NeIII_OII,Ne3O2Hadf_prism.NeIII_OII_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(Ne3O2Hadf_g395m.NeIII_OII,Ne3O2Hadf_g395m.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(Ne3O2Hadf_g395m.NeIII_OII,Ne3O2Hadf_g395m.NeIII_OII_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_O32_versus_Ha_linmix():
    x = pd.concat([np.log10(O32Hadf_prism.L_Ha), np.log10(O32Hadf_g395m.L_Ha)])
    y = pd.concat([np.log10(O32Hadf_prism.O32), np.log10(O32Hadf_g395m.O32)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(O32Hadf_prism.L_Ha,O32Hadf_prism.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(O32Hadf_prism.L_Ha,O32Hadf_prism.L_Ha_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O32Hadf_g395m.L_Ha,O32Hadf_g395m.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(O32Hadf_g395m.L_Ha,O32Hadf_g395m.L_Ha_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(O32Hadf_prism.O32,O32Hadf_prism.O32_ERR_16) + sf.propagate_uncertainty_log10(O32Hadf_prism.O32,O32Hadf_prism.O32_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(O32Hadf_g395m.O32,O32Hadf_g395m.O32_ERR_16) + sf.propagate_uncertainty_log10(O32Hadf_g395m.O32,O32Hadf_g395m.O32_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_R23_versus_Ha_linmix():
    x = pd.concat([np.log10(R23Hadf_prism.L_Ha), np.log10(R23Hadf_g395m.L_Ha)])
    y = pd.concat([np.log10(R23Hadf_prism.R23), np.log10(R23Hadf_g395m.R23)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(R23Hadf_prism.L_Ha,R23Hadf_prism.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(R23Hadf_prism.L_Ha,R23Hadf_prism.L_Ha_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(R23Hadf_g395m.L_Ha,R23Hadf_g395m.L_Ha_ERR_16) + sf.propagate_uncertainty_log10(R23Hadf_g395m.L_Ha,R23Hadf_g395m.L_Ha_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(R23Hadf_prism.R23,R23Hadf_prism.R23_ERR_16) + sf.propagate_uncertainty_log10(R23Hadf_prism.R23,R23Hadf_prism.R23_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(R23Hadf_g395m.R23,R23Hadf_g395m.R23_ERR_16) + sf.propagate_uncertainty_log10(R23Hadf_g395m.R23,R23Hadf_g395m.R23_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_R23_versus_O32_linmix():
    x = pd.concat([np.log10(R23df_prism.O32), np.log10(R23df_g395m.O32)])
    y = pd.concat([np.log10(R23df_prism.R23), np.log10(R23df_g395m.R23)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(R23df_prism.O32,R23df_prism.O32_ERR_16) + sf.propagate_uncertainty_log10(R23df_prism.O32,R23df_prism.O32_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(R23df_g395m.O32,R23df_g395m.O32_ERR_16) + sf.propagate_uncertainty_log10(R23df_g395m.O32,R23df_g395m.O32_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_16) + sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_16) + sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def fit_NeIIIOII_versus_O32_linmix():
    x = pd.concat([np.log10(Ne3O32df_prism.O32), np.log10(Ne3O32df_g395m.O32)])
    y = pd.concat([np.log10(Ne3O32df_prism.NeIII_OII), np.log10(Ne3O32df_g395m.NeIII_OII)])
    xsig = pd.concat([(sf.propagate_uncertainty_log10(Ne3O32df_prism.O32,Ne3O32df_prism.O32_ERR_16) + sf.propagate_uncertainty_log10(Ne3O32df_prism.O32,Ne3O32df_prism.O32_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(Ne3O32df_g395m.O32,Ne3O32df_g395m.O32_ERR_16) + sf.propagate_uncertainty_log10(Ne3O32df_g395m.O32,Ne3O32df_g395m.O32_ERR_84)) / 2])
    ysig = pd.concat([(sf.propagate_uncertainty_log10(Ne3O32df_prism.NeIII_OII,Ne3O32df_prism.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(Ne3O32df_prism.NeIII_OII,Ne3O32df_prism.NeIII_OII_ERR_84)) / 2,
                        (sf.propagate_uncertainty_log10(Ne3O32df_g395m.NeIII_OII,Ne3O32df_g395m.NeIII_OII_ERR_16) + sf.propagate_uncertainty_log10(Ne3O32df_g395m.NeIII_OII,Ne3O32df_g395m.NeIII_OII_ERR_84)) / 2])

    lm = linmix.LinMix(x, y, xsig, ysig, K=2)
    lm.run_mcmc(silent=True)

    return lm.chain

def plot_linmix(ax, chain, xmin, xmax):
    xs = np.linspace(xmin, xmax, 100)
    for i in range(0, len(chain), 25):
        ys = chain[i]['alpha'] + xs * chain[i]['beta']
        ax.plot(xs, ys, color='k', alpha=0.01, zorder=-10)
    ax.plot(xs, np.median(chain['alpha']) + xs * np.median(chain['beta']), color='k', alpha=1, zorder=-9)

def linmix_slope_and_intercept(chain):
    b = np.median(chain['alpha'])
    b_std = np.std(chain['alpha'])

    m = np.median(chain['beta'])
    m_std = np.std(chain['beta'])

    return f'y = {m} +/- {m_std}x + {b} +/- {b_std}'

if __name__ == "__main__":
   pass