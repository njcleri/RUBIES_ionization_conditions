from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from nikkos_tools import stat_functions as sf
import data_management
import globals
import linmix_fits

line_fluxes = pd.read_csv(globals.RUBIES_DATA.joinpath('line_flux_df.csv'), index_col=0)
line_fluxes_prism = data_management.make_df_prism(line_fluxes)
line_fluxes_g395m = data_management.make_df_g395m(line_fluxes)


R23df_prism = data_management.signal_to_noise_R23_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
R23df_g395m = data_management.signal_to_noise_R23_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
R23df_both = pd.merge(R23df_prism, R23df_g395m, on='id', how='inner')

Ne3O32df_prism = data_management.signal_to_noise_Ne3O32_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)
Ne3O32df_g395m = data_management.signal_to_noise_Ne3O32_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)
Ne3O32df_both = pd.merge(Ne3O32df_prism, Ne3O32df_g395m, on='id', how='inner')

sphinxdf = data_management.make_sphinx_df(globals.SPHINX_DATA)
sphinx_binned = data_management.make_sphinx_binned_df(sphinxdf)

def plot_R23_versus_O32(ax):
    chain = linmix_fits.fit_R23_versus_O32_linmix()
    linmix_fits.plot_linmix(ax, chain, -1, 2)
    # ax.scatter(sphinxdf.log_O32, sphinxdf.log_R23, marker='h', c='k', alpha=0.1, label='SPHINX')
    ax.errorbar(x=sphinx_binned.log_O32_sphinx, y=sphinx_binned.log_R23_sphinx, 
                xerr=[sphinx_binned.log_O32_sphinx_16, sphinx_binned.log_O32_sphinx_84], 
                yerr=[sphinx_binned.log_R23_sphinx_16, sphinx_binned.log_R23_sphinx_84], 
                ls='None', c='k', zorder=-9, label='SPHINX')
    ax.scatter(x=sphinx_binned.log_O32_sphinx, y=sphinx_binned.log_R23_sphinx, 
                s=180, marker='X', ec='k', c=sphinx_binned.redshift, cmap='Reds', vmin=0, vmax=10, zorder=-9, label='SPHINX')
    ax.errorbar(x=np.log10(R23df_prism.O32), y=np.log10(R23df_prism.R23), 
                xerr=[sf.propagate_uncertainty_log10(R23df_prism.O32,R23df_prism.O32_ERR_16), sf.propagate_uncertainty_log10(R23df_prism.O32,R23df_prism.O32_ERR_84)], 
                yerr=[sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_16), sf.propagate_uncertainty_log10(R23df_prism.R23,R23df_prism.R23_ERR_84)], 
                ls='None', c='k', zorder=-1)
    ax.scatter(x=np.log10(R23df_prism.O32), y=np.log10(R23df_prism.R23), 
                marker='o', ec='k', c=R23df_prism.z_prism, cmap='Reds', vmin=0, vmax=10, s=60, label='PRISM')
    ax.errorbar(x=np.log10(R23df_g395m.O32), y=np.log10(R23df_g395m.R23), 
                xerr=[sf.propagate_uncertainty_log10(R23df_g395m.O32,R23df_g395m.O32_ERR_16), sf.propagate_uncertainty_log10(R23df_g395m.O32,R23df_g395m.O32_ERR_84)], 
                yerr=[sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_16), sf.propagate_uncertainty_log10(R23df_g395m.R23,R23df_g395m.R23_ERR_84)], 
                ls='None', c='k', zorder=-1)
    ax.scatter(x=np.log10(R23df_g395m.O32), y=np.log10(R23df_g395m.R23), 
                marker='s', ec='k', c=R23df_g395m.z_g395m, cmap='Reds', vmin=0, vmax=10, s=60, label='G395M')
    ax.annotate(f'N = {len(R23df_prism)+len(R23df_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel(r'$\log\rm{O32}$')
    ax.set_ylabel(r'$\log\rm{R23}$')
    ax.axis([-1, 2, 0, 1.5])

def plot_NeIIIOII_versus_O32(ax):
    chain = linmix_fits.fit_NeIIIOII_versus_O32_linmix()
    linmix_fits.plot_linmix(ax, chain, -1, 2)
    # ax.scatter(sphinxdf.log_O32, sphinxdf.log_NeIII_OII, marker='h', c='k', alpha=0.1, label='SPHINX')
    ax.errorbar(x=sphinx_binned.log_O32_sphinx, y=sphinx_binned.log_NeIII_OII_sphinx, 
                xerr=[sphinx_binned.log_O32_sphinx_16, sphinx_binned.log_O32_sphinx_84], 
                yerr=[sphinx_binned.log_NeIII_OII_sphinx_16, sphinx_binned.log_NeIII_OII_sphinx_84], 
                ls='None', c='k', zorder=-9, label='SPHINX')
    ax.scatter(x=sphinx_binned.log_O32_sphinx, y=sphinx_binned.log_NeIII_OII_sphinx, 
                s=180, marker='X', ec='k', c=sphinx_binned.redshift, cmap='Reds', vmin=0, vmax=10, zorder=-9, label='SPHINX')
    ax.errorbar(x=np.log10(Ne3O32df_prism.O32), y=np.log10(Ne3O32df_prism.NeIII_OII), 
                xerr=[sf.propagate_uncertainty_log10(Ne3O32df_prism.O32,Ne3O32df_prism.O32_ERR_16), sf.propagate_uncertainty_log10(Ne3O32df_prism.O32,Ne3O32df_prism.O32_ERR_84)], 
                yerr=[sf.propagate_uncertainty_log10(Ne3O32df_prism.NeIII_OII,Ne3O32df_prism.NeIII_OII_ERR_16), sf.propagate_uncertainty_log10(Ne3O32df_prism.NeIII_OII,Ne3O32df_prism.NeIII_OII_ERR_84)], 
                ls='None', c='k', zorder=-1)
    ax.scatter(x=np.log10(Ne3O32df_prism.O32), y=np.log10(Ne3O32df_prism.NeIII_OII), 
                marker='o', ec='k', c=Ne3O32df_prism.z_prism, cmap='Reds', vmin=0, vmax=10, s=60, label='PRISM')
    ax.errorbar(x=np.log10(Ne3O32df_g395m.O32), y=np.log10(Ne3O32df_g395m.NeIII_OII), 
                xerr=[sf.propagate_uncertainty_log10(Ne3O32df_g395m.O32,Ne3O32df_g395m.O32_ERR_16), sf.propagate_uncertainty_log10(Ne3O32df_g395m.O32,Ne3O32df_g395m.O32_ERR_84)], 
                yerr=[sf.propagate_uncertainty_log10(Ne3O32df_g395m.NeIII_OII,Ne3O32df_g395m.NeIII_OII_ERR_16), sf.propagate_uncertainty_log10(Ne3O32df_g395m.NeIII_OII,Ne3O32df_g395m.NeIII_OII_ERR_84)], 
                ls='None', c='k', zorder=-1)
    ax.scatter(x=np.log10(Ne3O32df_g395m.O32), y=np.log10(Ne3O32df_g395m.NeIII_OII), 
                marker='s', ec='k', c=Ne3O32df_g395m.z_g395m, cmap='Reds', vmin=0, vmax=10, s=60, label='G395M')
    ax.annotate(f'N = {len(Ne3O32df_prism)+len(Ne3O32df_g395m)}', xy=(0.1, 0.9), xycoords='axes fraction')
    ax.set_xlabel(r'$\log\rm{O32}$')
    ax.set_ylabel(r'$\log\frac{\rm{[Ne~III]}~\lambda3870}{\rm{[O~II]}~\lambda\lambda3727,3730}$')
    ax.axis([-1,2,-1,1])

def generate_legend_elements_ratios_versus_O32():
    legend_elements = [
                   Line2D([0], [0], marker='o', color='none', label='PRISM', markerfacecolor='red', markeredgecolor='black', markersize=np.sqrt(100)),
                   Line2D([0], [0], marker='s', color='none', label='G395M', markerfacecolor='red', markeredgecolor='black', markersize=np.sqrt(100)),
                   Line2D([0], [0], marker='X', color='none', label='SPHINX', markerfacecolor='red', markeredgecolor='black', markersize=np.sqrt(100)),
                    ] 
    return legend_elements

def make_ratios_versus_O32_plot():
    fig = plt.figure(figsize=(25,10))
    gs = GridSpec(nrows=1, ncols=100)
    gs.update(wspace=0, hspace=0)

    ax0 = fig.add_subplot(gs[0:1, 0:42])
    plot_R23_versus_O32(ax=ax0)
    
    ax1 = fig.add_subplot(gs[0:1, 53:95])
    plot_NeIIIOII_versus_O32(ax=ax1)

    cax = fig.add_subplot(gs[0:1, 98:100])
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=cax, use_gridspec=True, label='Redshift')

    legend_elements = generate_legend_elements_ratios_versus_O32()     
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.975), facecolor='white', ncol=4, fontsize=20)

    plt.savefig(globals.FIGURES.joinpath('ratios_O32.pdf'))

if __name__ == "__main__":
   make_ratios_versus_O32_plot()