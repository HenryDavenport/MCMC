"""
import_priors.py
Contains functions to import the values from the priors file.
These consist of both expected values of parameters as well as the desired ranges that you want the MCMC
code to explore.
Function import_priors() is the main function which generates an instance of the Peak_Structure class
for each peak and fills in all the values from the priors.
"""
import numpy as np
from structures import Peak_Structure
from structures import Pair_Peaks


def import_priors(location, run_settings):
    """
    Load prior values from file location and identify each column using the column_titles.
    Returns the prior parameters in a list of the generated instances of the Peak_Structure class.
    With one instance of the class per peak which stores all the relevant information about the peak.
    :param location: file location for priors file
    :param run_settings: instance of class Run_Settings, con
    :return: list of instances of Peak_Structure class with prior parameters for each peak.
    """

    # import priors from file as a list of lists (one row for each peak)
    input_priors = np.loadtxt(location)

    # titles of columns for the prior file.
    column_titles = ["l", "n", "freq", "freq lower error", "freq upper error", "splitting",
                     "splitting lower error", "splitting upper error", "ln(width)", "ln(width) lower error",
                     "ln(width) upper error", "ln(power)", "ln(power) lower error", "ln(power) upper error",
                     "asymmetry", "asymmetry lower error", "asymmetry upper error"]

    # function converts the input_priors which is in a matrix to list of instances of Peak_Structure class.
    all_peaks = gen_peak_structs(input_priors, run_settings, column_titles)
    return all_peaks



def gen_peak_structs(input_data, run_settings, template):
    """
    takes in input data from file (as a list of lists/matrix)
    and a template labels each column of the matrix e.g "asymmetry" etc.
    Unpacks these values and puts them into one instance of the Peak_Structure class per peak.
    Returns a list of these classes.

    :param input_data: matrix of input data from priors file
    :param run_settings: instance of Run_Settings class containing peak information.
    :param template: the list of column titles for the priors file
    :return: returns list of instances of the Peak_Structure class with one per peak. Each contains all required information for a peak.
    """
    all_peaks = []
    for row in input_data:
        peak_dict = dict(zip(template, row))
        peak = Peak_Structure(peak_dict, run_settings)
        all_peaks.append(peak)
    return all_peaks


def find_pairs(all_peaks):
    """
    finds the pairs of peaks which are next to each other (e.g. the l=0/2 and l=1/3 pairs)
    e.g. l=0 n=N, l=2 n=N-1 and l=1 n=N, l=3 n=N-1
    :param all_peaks: a list of all peaks
    :return: a list of lists which contain each pair of peaks
    """
    pair_list = []
    for peak1 in all_peaks:
        if peak1.l == 0:
            for peak2 in all_peaks:
                if (peak2.l == 2) and (peak2.n == (peak1.n - 1)):
                    pair_peaks = Pair_Peaks([peak1, peak2])
                    pair_list.append(pair_peaks)
        if peak1.l == 1:
            for peak2 in all_peaks:
                if (peak2.l == 3) and (peak2.n == (peak1.n - 1)):
                    pair_peaks = Pair_Peaks([peak1, peak2])
                    pair_list.append(pair_peaks)

    return pair_list
