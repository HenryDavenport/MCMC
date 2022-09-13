"""
run_settings.py
Contains the Run_Settings class in which all global settings about the fits can be modified.
"""
from dataclasses import dataclass
from structures import Parameter

@dataclass
class Run_Settings:
    """
    Stores the information about all the fits being performed
    """

    cadence = 60  # in seconds the time between data point collection in the time series
    burnin = 100  # the number of samples used to initialise the walkers
    no_samples = 1000  # the total number of samples taken of the parameter space
    nwalkers = 20  # the number of walkers used by the emcee fit to explore the parameter space

    # the ratios of the heights of the m split components of the l=2 and l=3 modes
    l2_m_scale = 0.634
    l3_m_scale = 0.400
    # the mode visibilities
    l1_visibility = 1.505
    l2_visibility = 0.62
    l3_visibility = 0.075

    # the width of the fit window in micro-Hz.
    # This is the width above and below the expected frequencies of the peaks being fitted.
    window_width = 25

    lowest_freq_fitted = 2000  # lowest peak frequency (in micro-Hz) which will be fitted
    highest_freq_fitted = 4000  # lowest peak frequency (in micro-Hz) which will be fitted

    # the guess of the background
    background: Parameter
