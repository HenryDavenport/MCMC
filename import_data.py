"""
Imports the time series data (i.e. from GOLF, BiSON etc.) and returns the FT of it
"""
from astropy.io import fits
import math
import numpy as np
import matplotlib.pyplot as plt


def import_data(filename, start, end, cad):
    """
    Imports data from file and takes range within the datapoint indices given by start and end parameters.
    Returns the FT of the time series data within that range.

    :param filename: name of file with extension .fit that contains data
    :param start: The starting index in the data you want to fit (i.e. corresponds to a starting time)
    :param end: The end index of the range in which you want to fit the data
    :param cad: The cadence of the data collection (in seconds) e.g. 60 seconds for GOLF
    :return: power and frequency arrays for fitting
    """
    with fits.open(filename) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

    print(repr(hdr))
    vel = data[math.floor(start):math.floor(end)]

    npts = len(vel)
    time = np.arange(0, npts * cad, cad)

    # load velocity data and then FFT to get frequency-power spectrum of the Sun
    fill = np.count_nonzero(vel) / (1.0 * len(vel))
    # apply normalisation so the power is per unit frequency.
    power = (2.0 * cad / (fill * npts)) * ((abs(np.fft.rfft(vel))) ** 2)
    freq = 1e6 * np.fft.rfftfreq(npts, d=cad)

    return power, freq

def import_data_window(filename, start, end, cad):
    """
    Modified version of import_data for use if the time series has gaps in it. (e.g. for BiSON).
    This code hasn't been tested so may not work!

    :param filename:
    :param start:
    :param end:
    :param cad:
    :return:
    """
    with fits.open(filename) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

    print(repr(hdr))

    all_data = data[math.floor(start):math.floor(end)]

    # NOTE this is if BiSON data is in second axis which the example I used did have.
    vel = [x[1] for x in all_data]

    npts = len(vel)
    time2 = np.arange(0, npts * cad, cad)

    window_fun = create_window_function(vel)
    powWin = abs(np.fft.rfft(window_fun)) ** 2
    # load velocity data and then FFT to get frequency-power spectrum of the Sun
    fill = np.count_nonzero(vel) / (1.0 * len(vel))
    # apply normalisation so the power is per unit frequency.
    power = (2.0 * cad / (fill * npts)) * ((abs(np.fft.rfft(vel))) ** 2)
    freq = 1e6 * np.fft.rfftfreq(npts, d=cad)
    return power, freq, powWin

def create_window_function(vel_data):
    """ creates function with 1.0 if data was measured and 0.0 if not  """
    window_function = [0.0 if i == 0.0 else 1.0 for i in vel_data]
    return window_function
