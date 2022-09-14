""" model.py
Generates the model of the spectrum from asymmetric Lorentzians."""
import matplotlib.pyplot as plt
import numpy as np


def lorentz(freq, central_freq, width, pow, asymmetry):
    """
    function returns an asymmetric Lorenztian peak.
    :param freq: frequency array
    :param central_freq: the central frequency of the Lorenztian peak
    :param width: The width of the peak
    :param pow: the power of the peak
    :param asymmetry: the asymmetry of the peak
    :return: the Lorentzian peak array
    """
    height = pow / (0.5 * np.pi * width * 1e-6)
    xi = (2 / width) * (freq - central_freq)
    model = (height / (1 + xi ** 2)) * ((1 + asymmetry * xi) ** 2 + asymmetry ** 2)
    return model


def background(height):
    """
     returns the background function. At the moment uniform linear offset
    :param height: height of offset
    :return: uniform background
    """
    return height


def all_peaks_model(freq, fit_info):
    """
    returns the complete model for all peaks in fit_info across the frequency array freq.
    :param freq: frequency array
    :param fit_info: instance of class Fit_Info containing all the parameters for the fit.
    :return:
    """
    # create empty array to add Lorentzians to.
    model = np.zeros(np.shape(freq))

    for peak_pair in fit_info.peak_pair_list:
        for peak in peak_pair.peak_list:
            if peak.l == 0:
                # create 1 peak
                model += lorentz(freq, peak.freq.value, np.exp(peak.ln_linewidth.value),
                                      np.exp(peak.ln_power.value), peak.asymmetry.value)

            elif peak.l == 1:
                # create 2 peaks for m+-1:
                visibility = fit_info.l1_visibility
                model += lorentz(freq, peak.freq.value + peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value)/2, peak.asymmetry.value)
                model += lorentz(freq, peak.freq.value - peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / 2, peak.asymmetry.value)

            elif peak.l == 2:
                # create 3 peaks for m=+-2, 0

                # create m+-2 peaks:
                visibility = fit_info.l2_visibility
                scaling = 2 + peak.m_scale
                model += lorentz(freq, peak.freq.value + 2 * peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)
                model += lorentz(freq, peak.freq.value - 2 * peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)

                # m = 0 is smaller by factor "scale":
                model += peak.m_scale * lorentz(freq, peak.freq.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)

            elif peak.l == 3:
                # create 4 peaks for m+-1, m+-3
                visibility = fit_info.l3_visibility
                scaling = 2 + 2 * peak.m_scale

                # create m+-3 peaks:
                model += lorentz(freq, peak.freq.value + 3 * peak.splitting.value, np.exp(peak.ln_linewidth.value),
                        visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)
                model += lorentz(freq, peak.freq.value - 3 * peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)

                # m = +-1 is smaller by factor "scale":
                model += peak.m_scale * lorentz(freq, peak.freq.value + peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                 visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)
                model += peak.m_scale * lorentz(freq, peak.freq.value - peak.splitting.value, np.exp(peak.ln_linewidth.value),
                                                visibility * np.exp(peak.ln_power.value) / scaling, peak.asymmetry.value)

    # code for BiSON i.e. if gaps in data (this code hasn't been tested)
    # if PowWin_short is not None:
    #     n1 = len(model)
    #     n2 = len(PowWin_short)
    #     Temp = (np.fft.irfft(model) / (2.0 * n1)) * (np.fft.irfft(PowWin_short)) / (2.0 * n2)
    #     model = abs(np.fft.rfft(Temp))

    # add background
    model += fit_info.background.value
    return model