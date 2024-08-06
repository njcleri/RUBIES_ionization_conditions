import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pyplot as ax
from matplotlib.gridspec import GridSpec
import seaborn as sns
from nikkos_tools import stat_functions as sf
import data_management
import globals

line_fluxes = pd.read_csv(globals.RUBIES_DATA.joinpath('line_flux_df.csv'), index_col=0)
line_fluxes_prism = data_management.make_df_prism(line_fluxes)
line_fluxes_g395m = data_management.make_df_g395m(line_fluxes)

def make_redshift_histogram_plot():
    bins = np.arange(0,15,1)
    ax.figure(figsize=(10,10))
    ax.hist(line_fluxes_prism.z_prism, color=globals.PRISM_COLOR, alpha=0.5, bins=bins, label='PRISM')
    ax.hist(line_fluxes_g395m.z_g395m, color=globals.G395M_COLOR, alpha=0.5, bins=bins, label='G395M')
    ax.xlabel('Redshift')
    ax.ylabel('Number of Galaxies')
    ax.legend()
    plt.savefig(globals.FIGURES.joinpath('zhist.pdf'))


if __name__ == "__main__":
   make_redshift_histogram_plot()