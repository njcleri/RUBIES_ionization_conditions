from pathlib import Path
import numpy as np
import pandas as pd
from nikkos_tools import stat_functions as sf
from nikkos_tools import physics_functions as pf

def r23_uncertainty(oiii_5007, oiii_5007_ERR, oiii_4959, oiii_4959_ERR, oii, oii_ERR, Hb, Hb_err):
    numerator = oiii_5007 + oiii_4959 + oii
    numerator_ERR = sf.propagate_uncertainty_addition(sf.propagate_uncertainty_addition(oiii_5007_ERR, oiii_4959_ERR), oii_ERR)
    return sf.propagate_uncertainty_division(numerator, numerator_ERR, Hb, Hb_err)

def make_df_prism(df):
    new_df = (df.loc[:, df.columns.str.contains('prism')|(df.columns == 'id')|(df.columns == 'ra')|(df.columns == 'dec')]
                     .assign(
                            OIII_Hb = lambda x: x.Oiii_5007_prism/x.Hb_prism,
                            O32 = lambda x: x.Oiii_5007_prism/x.Oii_3727_prism,
                            NeIII_OII = lambda x: x.Neiii_3869_prism/x.Oii_3727_prism,
                            R23 = lambda x: (x.Oiii_5007_prism+x.Oiii_4959_prism+x.Oii_3727_prism)/x.Hb_prism,
                            OIII_Hb_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_prism, x.Oiii_5007_prism_p84_err,
                                                  x.Hb_prism, x.Hb_prism_p84_err),
                            OIII_Hb_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_prism, x.Oiii_5007_prism_p16_err,
                                                  x.Hb_prism, x.Hb_prism_p16_err),
                            O32_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_prism, x.Oiii_5007_prism_p84_err,
                                                  x.Oii_3727_prism, x.Oii_3727_prism_p84_err),
                            O32_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_prism, x.Oiii_5007_prism_p16_err,
                                                  x.Oii_3727_prism, x.Oii_3727_prism_p16_err),
                            NeIII_OII_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Neiii_3869_prism, x.Neiii_3869_prism_p84_err,
                                                  x.Oii_3727_prism, x.Oii_3727_prism_p84_err),
                            NeIII_OII_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Neiii_3869_prism, x.Neiii_3869_prism_p16_err,
                                                  x.Oii_3727_prism, x.Oii_3727_prism_p16_err), 
                            R23_ERR_84 = lambda x: r23_uncertainty(x.Oiii_5007_prism, x.Oiii_5007_prism_p84_err, 
                                                                   x.Oiii_4959_prism, x.Oiii_4959_prism_p84_err, 
                                                                   x.Oii_3727_prism, x.Oii_3727_prism_p84_err,
                                                                   x.Hb_prism, x.Hb_prism_p84_err),
                            R23_ERR_16 = lambda x: r23_uncertainty(x.Oiii_5007_prism, x.Oiii_5007_prism_p16_err, 
                                                                   x.Oiii_4959_prism, x.Oiii_4959_prism_p16_err, 
                                                                   x.Oii_3727_prism, x.Oii_3727_prism_p16_err,
                                                                   x.Hb_prism, x.Hb_prism_p16_err),
                            L_Ha = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Ha_prism)*10**-20),
                            L_Ha_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Ha_prism_p84_err)*10**-20),
                            L_Ha_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Ha_prism_p16_err)*10**-20),
                            L_Hb = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Hb_prism)*10**-20),
                            L_Hb_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Hb_prism_p84_err)*10**-20),
                            L_Hb_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Hb_prism_p16_err)*10**-20),
                            L_OIII_5007 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Oiii_5007_prism)*10**-20),
                            L_OIII_5007_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Oiii_5007_prism_p84_err)*10**-20),
                            L_OIII_5007_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_prism), np.array(x.Oiii_5007_prism_p16_err)*10**-20)
                            )
    )
    
    return new_df

def make_df_g395m(df):
    new_df = (df.loc[:, df.columns.str.contains('g395m')|(df.columns == 'id')|(df.columns == 'ra')|(df.columns == 'dec')]
                     .assign(
                            OIII_Hb = lambda x: x.Oiii_5007_g395m/x.Hb_g395m,
                            O32 = lambda x: x.Oiii_5007_g395m/x.Oii_3727_g395m,
                            NeIII_OII = lambda x: x.Neiii_3869_g395m/x.Oii_3727_g395m,
                            R23 = lambda x: (x.Oiii_5007_g395m+x.Oiii_4959_g395m+x.Oii_3727_g395m)/x.Hb_g395m,
                            OIII_Hb_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p84_err,
                                                  x.Hb_g395m, x.Hb_g395m_p84_err),
                            OIII_Hb_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p16_err,
                                                  x.Hb_g395m, x.Hb_g395m_p16_err),
                            O32_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p84_err,
                                                  x.Oii_3727_g395m, x.Oii_3727_g395m_p84_err),
                            O32_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p16_err,
                                                  x.Oii_3727_g395m, x.Oii_3727_g395m_p16_err),
                            NeIII_OII_ERR_84 = lambda x: sf.propagate_uncertainty_division(x.Neiii_3869_g395m, x.Neiii_3869_g395m_p84_err,
                                                  x.Oii_3727_g395m, x.Oii_3727_g395m_p84_err),
                            NeIII_OII_ERR_16 = lambda x: sf.propagate_uncertainty_division(x.Neiii_3869_g395m, x.Neiii_3869_g395m_p16_err,
                                                  x.Oii_3727_g395m, x.Oii_3727_g395m_p16_err), 
                            R23_ERR_84 = lambda x: r23_uncertainty(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p84_err, 
                                                                   x.Oiii_4959_g395m, x.Oiii_4959_g395m_p84_err, 
                                                                   x.Oii_3727_g395m, x.Oii_3727_g395m_p84_err,
                                                                   x.Hb_g395m, x.Hb_g395m_p84_err),
                            R23_ERR_16 = lambda x: r23_uncertainty(x.Oiii_5007_g395m, x.Oiii_5007_g395m_p16_err, 
                                                                   x.Oiii_4959_g395m, x.Oiii_4959_g395m_p16_err, 
                                                                   x.Oii_3727_g395m, x.Oii_3727_g395m_p16_err,
                                                                   x.Hb_g395m, x.Hb_g395m_p16_err),
                            L_Ha = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Ha_g395m)*10**-20),
                            L_Ha_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Ha_g395m_p84_err)*10**-20),
                            L_Ha_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Ha_g395m_p16_err)*10**-20),
                            L_Hb = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Hb_g395m)*10**-20),
                            L_Hb_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Hb_g395m_p84_err)*10**-20),
                            L_Hb_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Hb_g395m_p16_err)*10**-20),
                            L_OIII_5007 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Oiii_5007_g395m)*10**-20),
                            L_OIII_5007_ERR_84 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Oiii_5007_g395m_p84_err)*10**-20),
                            L_OIII_5007_ERR_16 = lambda x: pf.get_luminosity(np.array(x.z_g395m), np.array(x.Oiii_5007_g395m_p16_err)*10**-20)
                            )
    )

    return new_df

def signal_to_noise_5007_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Oiii_5007_g395m/df.Oiii_5007_g395m_p16_err > signal_to_noise) 
                           &(df.Oiii_5007_g395m/df.Oiii_5007_g395m_p84_err > signal_to_noise)
                           &(df.Oiii_5007_g395m < mcmc_runaway_threshold)\
                            )
    return signal_to_noise_cut

def signal_to_noise_5007_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_signal_to_noise_cut = (
                                  (df.Oiii_5007_prism/df.Oiii_5007_prism_p16_err > signal_to_noise)
                                  &(df.Oiii_5007_prism/df.Oiii_5007_prism_p84_err > signal_to_noise)
                                  &(df.Oiii_5007_prism < mcmc_runaway_threshold)
                                  )
    return signal_signal_to_noise_cut

def signal_to_noise_4959_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Oiii_4959_g395m/df.Oiii_4959_g395m_p16_err > signal_to_noise)
                           &(df.Oiii_4959_g395m/df.Oiii_4959_g395m_p84_err > signal_to_noise)
                           &(df.Oiii_4959_g395m < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_4959_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Oiii_4959_prism/df.Oiii_4959_prism_p16_err > signal_to_noise)
                           &(df.Oiii_4959_prism/df.Oiii_4959_prism_p84_err > signal_to_noise)
                           &(df.Oiii_4959_prism < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_3869_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Neiii_3869_g395m/df.Neiii_3869_g395m_p16_err > signal_to_noise)
                           &(df.Neiii_3869_g395m/df.Neiii_3869_g395m_p84_err > signal_to_noise)
                           &(df.Neiii_3869_g395m < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_3869_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Neiii_3869_prism/df.Neiii_3869_prism_p16_err > signal_to_noise)
                           &(df.Neiii_3869_prism/df.Neiii_3869_prism_p84_err > signal_to_noise)
                           &(df.Neiii_3869_prism < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_3727_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Oii_3727_g395m/df.Oii_3727_g395m_p16_err > signal_to_noise)
                           &(df.Oii_3727_g395m/df.Oii_3727_g395m_p84_err > signal_to_noise)
                           &(df.Oii_3727_g395m < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_3727_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Oii_3727_prism/df.Oii_3727_prism_p16_err > signal_to_noise)
                           &(df.Oii_3727_prism/df.Oii_3727_prism_p84_err > signal_to_noise)
                           &(df.Oii_3727_prism < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_Hb_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Hb_g395m/df.Hb_g395m_p16_err > signal_to_noise)
                           &(df.Hb_g395m/df.Hb_g395m_p84_err > signal_to_noise)
                           &(df.Hb_g395m < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_Hb_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Hb_prism/df.Hb_prism_p16_err > signal_to_noise)
                           &(df.Hb_prism/df.Hb_prism_p84_err > signal_to_noise)
                           &(df.Hb_prism < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_Ha_g395m(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Ha_g395m/df.Ha_g395m_p16_err > signal_to_noise)
                           &(df.Ha_g395m/df.Ha_g395m_p84_err > signal_to_noise)
                           &(df.Ha_g395m < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_Ha_prism(df, signal_to_noise, mcmc_runaway_threshold=10000):
    signal_to_noise_cut = (
                           (df.Ha_prism/df.Ha_prism_p16_err > signal_to_noise)
                           &(df.Ha_prism/df.Ha_prism_p84_err > signal_to_noise)
                           &(df.Ha_prism < mcmc_runaway_threshold)
                           )
    return signal_to_noise_cut

def signal_to_noise_O32_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise)
              ]

def signal_to_noise_O32_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise)
              ]

def signal_to_noise_O32Ha_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise) 
              & signal_to_noise_Ha_g395m(df, signal_to_noise)
              ]

def signal_to_noise_O32Ha_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise) 
              & signal_to_noise_Ha_prism(df, signal_to_noise)
              ]

def signal_to_noise_R23_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise) 
              & signal_to_noise_Hb_g395m(df, signal_to_noise) 
              & signal_to_noise_4959_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise)
              ]

def signal_to_noise_R23_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise) 
              & signal_to_noise_Hb_prism(df, signal_to_noise) 
              & signal_to_noise_4959_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise)
              ]

def signal_to_noise_R23Ha_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise) 
              & signal_to_noise_Hb_g395m(df, signal_to_noise) 
              & signal_to_noise_4959_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise) 
              & signal_to_noise_Ha_g395m(df, signal_to_noise)
              ]

def signal_to_noise_R23Ha_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise) 
              & signal_to_noise_Hb_prism(df, signal_to_noise) 
              & signal_to_noise_4959_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise) 
              & signal_to_noise_Ha_prism(df, signal_to_noise)
              ]

def signal_to_noise_O3Hb_prism(df, signal_to_noise):
    return df[
              signal_to_noise_Hb_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise) 
              ]

def signal_to_noise_O3Hb_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_Hb_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise) 
              ]

def signal_to_noise_O3HbHa_prism(df, signal_to_noise):
    return df[
              signal_to_noise_Hb_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise) 
              & signal_to_noise_Ha_prism(df, signal_to_noise)
              ]

def signal_to_noise_O3HbHa_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_Hb_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise) 
              & signal_to_noise_Ha_g395m(df, signal_to_noise)
              ]

def signal_to_noise_Ne3O2_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise)  
              & signal_to_noise_3869_prism(df, signal_to_noise) 
              ]

def signal_to_noise_Ne3O2_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise)  
              & signal_to_noise_3869_g395m(df, signal_to_noise) 
              ]

def signal_to_noise_Ne3O2Ha_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise)  
              & signal_to_noise_3869_prism(df, signal_to_noise) 
              & signal_to_noise_Ha_prism(df, signal_to_noise)
              ]

def signal_to_noise_Ne3O2Ha_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise)  
              & signal_to_noise_3869_g395m(df, signal_to_noise) 
              & signal_to_noise_Ha_g395m(df, signal_to_noise)
              ]

def signal_to_noise_Ne3O32_prism(df, signal_to_noise):
    return df[
              signal_to_noise_3727_prism(df, signal_to_noise)  
              & signal_to_noise_3869_prism(df, signal_to_noise) 
              & signal_to_noise_5007_prism(df, signal_to_noise)
              ]

def signal_to_noise_Ne3O32_g395m(df, signal_to_noise):
    return df[
              signal_to_noise_3727_g395m(df, signal_to_noise)  
              & signal_to_noise_3869_g395m(df, signal_to_noise) 
              & signal_to_noise_5007_g395m(df, signal_to_noise)
              ]

def make_sphinx_df(sphinx_data_path):
    data = Path(sphinx_data_path).resolve()
    df = (pd.read_csv(data.joinpath('all_basic_data.csv'))
                .assign(
                OIII_Hb = lambda x: x['O__3_5006.84A_int'] / x['H__1_4861.32A_int'],
                O32 = lambda x: x['O__3_5006.84A_int'] / (x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"]),
                R23 = lambda x: (x['O__3_5006.84A_int'] + x["O__3_4958.91A_int"] + x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"]) / x["H__1_4861.32A_int"],
                NeIII_OII = lambda x: x["Ne_3_3868.76A_int"] / (x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"]),
                log_OIII_Hb = lambda x: np.log10(x['O__3_5006.84A_int'] / x['H__1_4861.32A_int']),
                log_O32 = lambda x: np.log10(x['O__3_5006.84A_int'] / (x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"])),
                log_R23 = lambda x: np.log10((x['O__3_5006.84A_int'] + x["O__3_4958.91A_int"] + x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"]) / x["H__1_4861.32A_int"]),
                log_NeIII_OII = lambda x: np.log10(x["Ne_3_3868.76A_int"] / (x["O__2_3728.81A_int"] + x["O__2_3726.03A_int"])),
                log_LHa = lambda x: np.log10(x["H__1_6562.80A_int"]),
                )
    )
    return df

def make_sphinx_binned_df(sphinxdf):

    z_sphinx = sphinxdf['redshift'].unique()
    hden_OII = np.zeros(len(z_sphinx))
    hden_CIII = np.zeros(len(z_sphinx))

    log_LHa_sphinx = np.zeros(len(z_sphinx))
    log_LHa_sphinx_16 = np.zeros(len(z_sphinx))
    log_LHa_sphinx_84 = np.zeros(len(z_sphinx))

    log_O32_sphinx = np.zeros(len(z_sphinx))
    log_O32_sphinx_16 = np.zeros(len(z_sphinx))
    log_O32_sphinx_84 = np.zeros(len(z_sphinx))

    log_R23_sphinx = np.zeros(len(z_sphinx))
    log_R23_sphinx_16 = np.zeros(len(z_sphinx))
    log_R23_sphinx_84 = np.zeros(len(z_sphinx))

    log_OIII_Hb_sphinx = np.zeros(len(z_sphinx))
    log_OIII_Hb_sphinx_16 = np.zeros(len(z_sphinx))
    log_OIII_Hb_sphinx_84 = np.zeros(len(z_sphinx))

    log_NeIII_OII_sphinx = np.zeros(len(z_sphinx))
    log_NeIII_OII_sphinx_16 = np.zeros(len(z_sphinx))
    log_NeIII_OII_sphinx_84 = np.zeros(len(z_sphinx))

    for i,z in enumerate(z_sphinx):
        hden_OII[i] = sphinxdf[sphinxdf.redshift == z]['gas_density_3727'].median()
        hden_CIII[i] = sphinxdf[sphinxdf.redshift == z]['gas_density_1908'].median()

        log_LHa_sphinx[i] = sphinxdf[sphinxdf.redshift == z]['log_LHa'].median()
        log_LHa_sphinx_16[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_LHa'].quantile(q=0.16) - log_LHa_sphinx[i])
        log_LHa_sphinx_84[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_LHa'].quantile(q=0.84) - log_LHa_sphinx[i])

        log_O32_sphinx[i] = sphinxdf[sphinxdf.redshift == z]['log_O32'].median()
        log_O32_sphinx_16[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_O32'].quantile(q=0.16) - log_O32_sphinx[i])
        log_O32_sphinx_84[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_O32'].quantile(q=0.84) - log_O32_sphinx[i])

        log_R23_sphinx[i] = sphinxdf[sphinxdf.redshift == z]['log_R23'].median()
        log_R23_sphinx_16[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_R23'].quantile(q=0.16) - log_R23_sphinx[i])
        log_R23_sphinx_84[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_R23'].quantile(q=0.84) - log_R23_sphinx[i])

        log_OIII_Hb_sphinx[i] = sphinxdf[sphinxdf.redshift == z]['log_OIII_Hb'].median()
        log_OIII_Hb_sphinx_16[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_OIII_Hb'].quantile(q=0.16) - log_OIII_Hb_sphinx[i])
        log_OIII_Hb_sphinx_84[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_OIII_Hb'].quantile(q=0.84) - log_OIII_Hb_sphinx[i])

        log_NeIII_OII_sphinx[i] = sphinxdf[sphinxdf.redshift == z]['log_NeIII_OII'].median()
        log_NeIII_OII_sphinx_16[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_NeIII_OII'].quantile(q=0.16) - log_NeIII_OII_sphinx[i])
        log_NeIII_OII_sphinx_84[i] = np.abs(sphinxdf[sphinxdf.redshift == z]['log_NeIII_OII'].quantile(q=0.84) - log_NeIII_OII_sphinx[i])

    sphinx_binned_data = {
        'redshift':z_sphinx,
        'hden_OII':hden_OII,
        'hden_CIII':hden_CIII,
        'log_LHa_sphinx':log_LHa_sphinx,
        'log_LHa_sphinx_16':log_LHa_sphinx_16,
        'log_LHa_sphinx_84':log_LHa_sphinx_84,
        'log_O32_sphinx':log_O32_sphinx,
        'log_O32_sphinx_16':log_O32_sphinx_16,
        'log_O32_sphinx_84':log_O32_sphinx_84,
        'log_R23_sphinx':log_R23_sphinx,
        'log_R23_sphinx_16':log_R23_sphinx_16,
        'log_R23_sphinx_84':log_R23_sphinx_84,
        'log_OIII_Hb_sphinx':log_OIII_Hb_sphinx,
        'log_OIII_Hb_sphinx_16':log_OIII_Hb_sphinx_16,
        'log_OIII_Hb_sphinx_84':log_OIII_Hb_sphinx_84,
        'log_NeIII_OII_sphinx':log_NeIII_OII_sphinx,
        'log_NeIII_OII_sphinx_16':log_NeIII_OII_sphinx_16,
        'log_NeIII_OII_sphinx_84':log_NeIII_OII_sphinx_84
    }

    return pd.DataFrame(data=sphinx_binned_data)
