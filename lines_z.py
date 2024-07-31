import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from nikkos_tools import stat_functions as sf
import data_management
import globals

OII_COLOR = globals.COLORS[0]
NEIII_COLOR = globals.COLORS[1]
HB_COLOR = globals.COLORS[2]
OIII_4959_COLOR = globals.COLORS[3]
OIII_5007_COLOR = globals.COLORS[4]
HA_COLOR = globals.COLORS[5]
SII_COLOR = globals.COLORS[5]

line_fluxes = pd.read_csv(globals.RUBIES_DATA.joinpath('line_flux_df.csv'), index_col=0)
line_fluxes_prism = data_management.make_df_prism(line_fluxes)
line_fluxes_g395m = data_management.make_df_g395m(line_fluxes)

Hadf_prism = line_fluxes_prism[data_management.signal_to_noise_Ha_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)]
Hadf_g395m = line_fluxes_g395m[data_management.signal_to_noise_Ha_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)]
Hadf_both = pd.merge(Hadf_prism, Hadf_g395m, on='id', how='inner')

Hbdf_prism = line_fluxes_prism[data_management.signal_to_noise_Hb_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)]
Hbdf_g395m = line_fluxes_g395m[data_management.signal_to_noise_Hb_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)]
Hbdf_both = pd.merge(Hbdf_prism, Hbdf_g395m, on='id', how='inner')


O3df_prism = line_fluxes_prism[data_management.signal_to_noise_5007_prism(line_fluxes_prism, globals.LINE_SIGNAL_TO_NOISE)]
O3df_g395m = line_fluxes_g395m[data_management.signal_to_noise_5007_g395m(line_fluxes_g395m, globals.LINE_SIGNAL_TO_NOISE)]
O3df_both = pd.merge(O3df_prism, O3df_g395m, on='id', how='inner')

def make_luminosity_versus_redshift_plot():
    fig = plt.figure(figsize=(13,8))
    gs = GridSpec(nrows=12, ncols=10)
    gs.update(wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[0:2, 0:10], frameon=False)
    ax1.tick_params(left=False, labelleft=False, right = False , labelbottom = False, bottom = False, labeltop = False, top = False, which = 'both')
    sns.kdeplot(data=np.concatenate((Hadf_prism.z_prism, Hadf_g395m.z_g395m)), color=HA_COLOR, alpha=0.3, fill=True, ax=ax1)
    sns.kdeplot(data=np.concatenate((O3df_prism.z_prism, O3df_g395m.z_g395m)), color=OIII_5007_COLOR, alpha=0.3, fill=True, ax=ax1)
    sns.kdeplot(data=np.concatenate((Hbdf_prism.z_prism, Hbdf_g395m.z_g395m)), color=HB_COLOR, alpha=0.3, fill=True, ax=ax1)
    ax1.set_xlim([0,12])

    ax = fig.add_subplot(gs[2:12, 0:10])

    ax.errorbar(x=Hadf_prism.z_prism, y=np.log10(Hadf_prism.L_Ha), 
                yerr=[sf.propagate_uncertainty_log10(Hadf_prism.L_Ha,Hadf_prism.L_Ha_ERR_16), sf.propagate_uncertainty_log10(Hadf_prism.L_Ha,Hadf_prism.L_Ha_ERR_84)],
                ls='None', marker='o', mec='k', color=HA_COLOR, label=r'$H\alpha$ PRISM')
    ax.errorbar(x=Hadf_g395m.z_g395m, y=np.log10(Hadf_g395m.L_Ha), 
                yerr=[sf.propagate_uncertainty_log10(Hadf_g395m.L_Ha,Hadf_g395m.L_Ha_ERR_16), sf.propagate_uncertainty_log10(Hadf_g395m.L_Ha,Hadf_g395m.L_Ha_ERR_84)],
                ls='None', marker='s', mec='k', color=HA_COLOR, label=r'$H\alpha$ G395M')

    ax.errorbar(x=O3df_prism.z_prism, y=np.log10(O3df_prism.L_OIII_5007), 
                yerr=[sf.propagate_uncertainty_log10(O3df_prism.L_OIII_5007,O3df_prism.L_OIII_5007_ERR_16), sf.propagate_uncertainty_log10(O3df_prism.L_OIII_5007,O3df_prism.L_OIII_5007_ERR_84)],
                ls='None', marker='o', mec='k', color=OIII_5007_COLOR, label=r'[O III] PRISM')
    ax.errorbar(x=O3df_g395m.z_g395m, y=np.log10(O3df_g395m.L_OIII_5007), 
                yerr=[sf.propagate_uncertainty_log10(O3df_g395m.L_OIII_5007,O3df_g395m.L_OIII_5007_ERR_16), sf.propagate_uncertainty_log10(O3df_g395m.L_OIII_5007,O3df_g395m.L_OIII_5007_ERR_84)],
                ls='None', marker='s', mec='k', color=OIII_5007_COLOR, label=r'[O III] G395M')

    ax.errorbar(x=Hbdf_prism.z_prism, y=np.log10(Hbdf_prism.L_Hb), 
                yerr=[sf.propagate_uncertainty_log10(Hbdf_prism.L_Hb,Hbdf_prism.L_Hb_ERR_16), sf.propagate_uncertainty_log10(Hbdf_prism.L_Hb,Hbdf_prism.L_Hb_ERR_84)],
                ls='None', marker='o', mec='k', color=HB_COLOR, label=r'$H\beta$ PRISM')
    ax.errorbar(x=Hbdf_g395m.z_g395m, y=np.log10(Hbdf_g395m.L_Hb), 
                yerr=[sf.propagate_uncertainty_log10(Hbdf_g395m.L_Hb,Hbdf_g395m.L_Hb_ERR_16), sf.propagate_uncertainty_log10(Hbdf_g395m.L_Hb,Hbdf_g395m.L_Hb_ERR_84)],
                ls='None', marker='s', mec='k', color=HB_COLOR, label=r'$H\beta$ G395M')

    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\log L_{line}$')

    ax.axis([0,12,37.5,44.5])
    ax.legend()

    plt.savefig(globals.FIGURES.joinpath('lines_z.pdf'))

if __name__ == "__main__":
   make_luminosity_versus_redshift_plot()
