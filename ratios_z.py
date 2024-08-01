from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from astropy.table import Table
import linmix
from nikkos_tools import stat_functions as sf
import data_management
import globals

SPHINX_DATA = Path('/Users/njc5787/Research/SPHINX-20-data/data').resolve()
RUBIES_DATA = Path('../data').resolve()
FIGURES = Path('./figures').resolve()

COLORS = ['#CC79A7', '#D55E00', '#0072B2', '#56B4E9', '#009E73', '#E69F00', '#000000', '#F0E442',]
globals.PRISM_COLOR = COLORS[2]
globals.G395M_COLOR = COLORS[1]

line_fluxes = pd.read_csv(globals.RUBIES_DATA.joinpath('line_flux_df.csv'), index_col=0)
line_fluxes_prism = data_management.make_df_prism(line_fluxes)
line_fluxes_g395m = data_management.make_df_g395m(line_fluxes)

O32df_prism = data_management.signal_to_noise_O32_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O32df_g395m = data_management.signal_to_noise_O32_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O32df_both = pd.merge(O32df_prism, O32df_g395m, on='id', how='inner')

R23df_prism = data_management.signal_to_noise_R23_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
R23df_g395m = data_management.signal_to_noise_R23_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
R23df_both = pd.merge(R23df_prism, R23df_g395m, on='id', how='inner')

O3Hbdf_prism = data_management.signal_to_noise_O3Hb_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
O3Hbdf_g395m = data_management.signal_to_noise_O3Hb_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
O3Hbdf_both = pd.merge(O3Hbdf_prism, O3Hbdf_g395m, on='id', how='inner')

Ne3O2df_prism = data_management.signal_to_noise_Ne3O2_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2df_g395m = data_management.signal_to_noise_Ne3O2_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
Ne3O2df_both = pd.merge(Ne3O2df_prism, Ne3O2df_g395m, on='id', how='inner')


sphinxdf = data_management.make_sphinx_df(globals.SPHINX_DATA)
sphinx_binned = data_management.make_sphinx_binned_df(sphinxdf)

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

def plot_OIIIHb_versus_redshift(ax):
    chain = fit_OIIIHb_versus_redshift_linmix()
    chain_z05 = fit_OIIIHb_versus_redshift_less_than_5_linmix()
    chain_z5plus = fit_OIIIHb_versus_redshift_greater_than_5_linmix()
    for i in range(0, len(chain), 25):
        xs = np.arange(0,11)
        ys = chain[i]['alpha'] + xs * chain[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)

    for i in range(0, len(chain_z05), 25):
        xs = np.arange(0,6)
        ys = chain_z05[i]['alpha'] + xs * chain_z05[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)
    
    for i in range(0, len(chain_z5plus), 25):
        xs = np.arange(5,11)
        ys = chain_z5plus[i]['alpha'] + xs * chain_z5plus[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)

    ax.errorbar(x=sphinx_binned.redshift, y=sphinx_binned.log_OIII_Hb_sphinx, 
                yerr=[sphinx_binned.log_OIII_Hb_sphinx_16, sphinx_binned.log_OIII_Hb_sphinx_84], 
                ms=10, marker='h', c='k', zorder=-9, label='SPHINX')
    ax.errorbar(x=O3Hbdf_prism.z_prism, y=np.log10(O3Hbdf_prism.OIII_Hb), 
                yerr=[sf.propagate_uncertainty_log10(O3Hbdf_prism.OIII_Hb,O3Hbdf_prism.OIII_Hb_ERR_16), sf.propagate_uncertainty_log10(O3Hbdf_prism.OIII_Hb,O3Hbdf_prism.OIII_Hb_ERR_84)], 
                ls='None', color=globals.PRISM_COLOR, marker='o', mec='k', label='PRISM')
    ax.errorbar(x=O3Hbdf_g395m.z_g395m, y=np.log10(O3Hbdf_g395m.OIII_Hb), 
                yerr=[sf.propagate_uncertainty_log10(O3Hbdf_g395m.OIII_Hb,O3Hbdf_g395m.OIII_Hb_ERR_16), sf.propagate_uncertainty_log10(O3Hbdf_g395m.OIII_Hb,O3Hbdf_g395m.OIII_Hb_ERR_84)], 
                ls='None', color=globals.G395M_COLOR, marker='s', mec='k', label='G395M')
    z = np.linspace(1,9,1000)
    ax.fill_between(z, 0.04*z + 0.16, 0.10*z + 0.36, color='k', alpha=0.1)
    ax.plot(z, 0.07*z + 0.26, color='k', alpha=0.3, label='Backhaus+2024')
    ax.annotate(f'N = {len(O3Hbdf_prism)+len(O3Hbdf_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\log\frac{\rm{[O~III]}~\lambda5008}{\rm{H}\beta}$')
    ax.axis([0, 11, -0.75, 1.5])

def plot_NeIIIOII_versus_redshift(ax):
    chain = fit_NeIIIOII_versus_redshift_linmix()
    for i in range(0, len(chain), 25):
        xs = np.arange(0,11)
        ys = chain[i]['alpha'] + xs * chain[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)
    ax.errorbar(x=sphinx_binned.redshift, y=sphinx_binned.log_NeIII_OII_sphinx, 
                yerr=[sphinx_binned.log_NeIII_OII_sphinx_16, sphinx_binned.log_NeIII_OII_sphinx_84], 
                ms=10, marker='h', c='k', zorder=-9, label='SPHINX')
    ax.errorbar(x=Ne3O2df_prism.z_prism, y=np.log10(Ne3O2df_prism.NeIII_OII), 
             yerr=[sf.propagate_uncertainty_log10(Ne3O2df_prism.NeIII_OII, Ne3O2df_prism.NeIII_OII_ERR_16), sf.propagate_uncertainty_log10(Ne3O2df_prism.NeIII_OII, Ne3O2df_prism.NeIII_OII_ERR_84)], 
             ls='None', color=globals.PRISM_COLOR, marker='o', mec='k', label='PRISM')
    ax.errorbar(x=Ne3O2df_g395m.z_g395m, y=np.log10(Ne3O2df_g395m.NeIII_OII), 
                yerr=[sf.propagate_uncertainty_log10(Ne3O2df_g395m.NeIII_OII, Ne3O2df_g395m.NeIII_OII_ERR_16), sf.propagate_uncertainty_log10(Ne3O2df_g395m.NeIII_OII, Ne3O2df_g395m.NeIII_OII_ERR_84)], 
                ls='None', color=globals.G395M_COLOR, marker='s', mec='k', label='G395M')
    z = np.linspace(1,9,1000)
    ax.fill_between(z, 0.03*z - 0.71, 0.07*z - 0.51, color='k', alpha=0.1)
    ax.plot(z, 0.05*z - 0.61, color='k', alpha=0.3, label='Backhaus+2024')
    ax.annotate(f'N = {len(O32df_prism)+len(O32df_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\log\frac{\rm{[Ne~III]}~\lambda3870}{\rm{[O~II]}~\lambda\lambda3727,3730}$')
    ax.axis([0, 11, -1, 1.5])

def plot_O32_versus_redshift(ax):
    chain = fit_O32_versus_redshift_linmix()
    for i in range(0, len(chain), 25):
        xs = np.arange(0,11)
        ys = chain[i]['alpha'] + xs * chain[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)

    ax.errorbar(x=sphinx_binned.redshift, y=sphinx_binned.log_O32_sphinx, 
                yerr=[sphinx_binned.log_O32_sphinx_16, sphinx_binned.log_O32_sphinx_84], 
                ms=10, marker='h', c='k', zorder=-9, label='SPHINX')
    ax.errorbar(x=O32df_prism.z_prism, y=np.log10(O32df_prism.O32), 
                yerr=[sf.propagate_uncertainty_log10(O32df_prism.O32,O32df_prism.O32_ERR_16), sf.propagate_uncertainty_log10(O32df_prism.O32,O32df_prism.O32_ERR_84)], 
                ls='None', color=globals.PRISM_COLOR, marker='o', mec='k', label='PRISM')
    ax.errorbar(x=O32df_g395m.z_g395m, y=np.log10(O32df_g395m.O32), 
                yerr=[sf.propagate_uncertainty_log10(O32df_g395m.O32,O32df_g395m.O32_ERR_16), sf.propagate_uncertainty_log10(O32df_g395m.O32,O32df_g395m.O32_ERR_84)],
                ls='None', color=globals.G395M_COLOR, marker='s', mec='k', label='G395M')
    ax.annotate(f'N = {len(O32df_prism)+len(O32df_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\rm{O32} \equiv \frac{\rm{[O~III]}~\lambda5008}{\rm{[O~II]}~\lambda\lambda3727,3729}$')
    ax.axis([0, 11, -1.5, 2])

def plot_R23_versus_redshift(ax):
    chain = fit_R23_versus_redshift_linmix()
    for i in range(0, len(chain), 25):
        xs = np.arange(0,11)
        ys = chain[i]['alpha'] + xs * chain[i]['beta']
        ax.plot(xs, ys, color='r', alpha=0.02)

    ax.errorbar(x=sphinx_binned.redshift, y=sphinx_binned.log_R23_sphinx, 
                yerr=[sphinx_binned.log_R23_sphinx_16, sphinx_binned.log_R23_sphinx_84], 
                ms=10, marker='h', c='k', zorder=-9, label='SPHINX')
    ax.errorbar(x=R23df_prism.z_prism, y=np.log10(R23df_prism.R23), 
                yerr=[sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_16), sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_84)], 
                ls='None', color=globals.PRISM_COLOR, marker='o', mec='k', zorder=-1, label='PRISM')
    ax.errorbar(x=R23df_g395m.z_g395m, y=np.log10(R23df_g395m.R23), 
                yerr=[sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_16), sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_84)], 
                ls='None', color=globals.G395M_COLOR, marker='s', mec='k', zorder=-1, label='G395M')
    ax.annotate(f'N = {len(R23df_prism)+len(R23df_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\log\rm{R23} \equiv \log\frac{\rm{[O~III]}~\lambda\lambda 4959,5008 + \rm{[O~II]}~\lambda\lambda3727,3730}{H\beta}$')
    ax.axis([0, 11, 0, 1.3])

def generate_legend_elements_ratios_versus_redshift():
    legend_elements = [
                   Line2D([1], [1], color='k', alpha=0.3, label='Backhaus+2024', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='none', label='PRISM', markerfacecolor=globals.PRISM_COLOR, markeredgecolor='black', markersize=np.sqrt(100)),
                   Line2D([0], [0], marker='s', color='none', label='G395M', markerfacecolor=globals.G395M_COLOR, markeredgecolor='black', markersize=np.sqrt(100)),
                   Line2D([0], [0], marker='h', color='none', label='SPHINX', markerfacecolor='k', markeredgecolor='black', markersize=np.sqrt(100)),
                    ] 
    return legend_elements

def make_ratios_versus_redshift_plot():
    fig = plt.figure(figsize=(20,20))
    gs = GridSpec(nrows=2, ncols=2)
    gs.update(wspace=0.3, hspace=0.2)

    ax0 = fig.add_subplot(gs[0:1, 0:1])
    plot_OIIIHb_versus_redshift(ax=ax0)

    ax1 = fig.add_subplot(gs[0:1, 1:2])
    plot_NeIIIOII_versus_redshift(ax=ax1)

    ax2 = fig.add_subplot(gs[1:2, 0:1])
    plot_O32_versus_redshift(ax=ax2)

    ax3 = fig.add_subplot(gs[1:2, 1:2])
    plot_R23_versus_redshift(ax=ax3)

    legend_elements = generate_legend_elements_ratios_versus_redshift()     
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.925), facecolor='white', ncol=4, fontsize=20)

    plt.savefig(globals.FIGURES.joinpath('ratios_z.pdf'))


if __name__ == "__main__":
   make_ratios_versus_redshift_plot()