from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import globals

def get_zmin_prism(wave):
    return 6000/wave - 1
def get_zmax_prism(wave):
    return 53000/wave - 1
def get_zmin_g395m(wave):
    return 29000/wave - 1
def get_zmax_g395m(wave):
    return 52000/wave - 1
def get_line_rotation(x1,x2,y1,y2):
    return np.rad2deg(np.arctan((y2 - y1)/(x2 - x1)))

def plot_prism_coverage(ax, lines, labels, line_colors, offsets_prism, z_annotations):
    z = np.linspace(0, 15, 1000)
    for line, label, line_color, offset, z_annotation in zip(lines,labels,line_colors,offsets_prism,z_annotations):
        observed_wavelengths = line*(1+z)/10**4
        ax.plot(z, observed_wavelengths, lw=3, color=line_color, label=label)
        rotn = get_line_rotation(z[0], z[-1], observed_wavelengths[0], observed_wavelengths[-1])
        ax.annotate(label, xy=(z_annotation,line*(1+z_annotation)/10**4+offset), fontsize=10, va='center', ha='center', color=line_color,
                    rotation=rotn, transform_rotates_text=True, rotation_mode='anchor')
    ax.annotate('PRISM',  xy=(0.01,0.95), va='top', ha='left', xycoords='axes fraction')

    ax.axis([0,13.5,0.6,5.3])
    ax.set_yticks([0.6,5.3])
    

def plot_g395m_coverage(ax, lines, labels, line_colors, offsets_g395m, z_annotations):
    z = np.linspace(0, 15, 1000)
    for line, label, line_color, offset, z_annotation in zip(lines,labels,line_colors,offsets_g395m,z_annotations):
        observed_wavelengths = line*(1+z)/10**4
        ax.plot(z, observed_wavelengths, lw=3, color=line_color, label=label)
        rotn = get_line_rotation(z[0], z[-1], observed_wavelengths[0], observed_wavelengths[-1])
        ax.annotate(label, xy=(z_annotation,line*(1+z_annotation)/10**4+offset), fontsize=10, va='center', ha='center', color=line_color,
                    rotation=rotn, transform_rotates_text=True, rotation_mode='anchor')
    ax.axis([0,13.5,2.9,5.2])
    ax.annotate('G395M',  xy=(0.01,0.95), va='top', ha='left', xycoords='axes fraction')
    ax.set_yticks([2.9,5.2])
    
def make_coverage_plot():
    lines = [6563, 5008, 4862, 3870, 3728]
    labels = [r'H$\alpha$', '[O III]', r'H$\beta$', '[Ne III]', '[O II]']
    line_colors = globals.COLORS
    offsets_prism = [.25, .25, -.25, .25, -.25]
    offsets_g395m = [.15, .15, -.15, .15, -.15]
    z_annotations = [4, 6, 6, 8, 8]

    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(nrows=10, ncols=10)
    gs.update(wspace=0, hspace=2)

    ax0 = fig.add_subplot(gs[0:5, 0:10])
    plot_prism_coverage(ax0, lines, labels, line_colors, offsets_prism, z_annotations)
    ax0.tick_params(labelbottom=False, which='both', top=True, labeltop=True)


    ax1 = fig.add_subplot(gs[5:10, 0:10])
    plot_g395m_coverage(ax1, lines, labels, line_colors, offsets_g395m, z_annotations)
    ax1.tick_params(which='both', top=True)

    plt.xlabel('Redshift')
    plt.annotate(r'Observed Wavelength [$\mu$m]', va='center', ha='center', xy=(0.03,0.5), xycoords='figure fraction', rotation=90)
    plt.savefig(globals.FIGURES.joinpath('coverage.pdf'))

if __name__ == "__main__":
   make_coverage_plot()