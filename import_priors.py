import numpy as np
from structures import Peak_Structure as peak_struc
from structures import Pair_Peaks as pair_peaks_struc

from typing import List

# functions used to process the input data and change its format (dictionary to list of input parameters etc)
def gen_peak_structs(input_data, run_info, template):
    """ takes in input data from file (as a list of lists) and a template which gives a label to each element.
    generates a dictionary of elements"""
    all_peaks = []
    for row in input_data:
        peak_dict = dict(zip(template, row))
        peak = peak_struc(peak_dict, run_info)
        all_peaks.append(peak)
    return all_peaks


def import_priors(location, run_info):
    """imports priors from file location from table with columns given by column_titles
    returns a list of the l and n numbers in the priors and a list of dictionaries of the data for each peak"""
    # import priors from file as a list of lists (one row for each peak)
    input_priors = np.loadtxt(location)
    column_titles = ["l", "n", "freq", "freq lower error", "freq upper error", "splitting",
                            "splitting lower error",
                            "splitting upper error", "ln(width)", "ln(width) lower error", "ln(width) upper error",
                            "ln(power)", "ln(power) lower error", "ln(power) upper error", "asymmetry",
                            "asymmetry lower error", "asymmetry upper error"]
    pk = peak_struc

    # add extra data for scale height of m split components
    # input_data_proc = []
    # for peak in input_priors:
    #     if peak[0] == 2.0:
    #         peak = np.append(peak, self.l2_m_scale)
    #     if peak[0] == 3.0:
    #         peak = np.append(peak, self.l3_m_scale)
    #     else:
    #         peak = np.append(peak, 0)
    #     input_data_proc.append(peak)
    # input_priors = input_data_proc

    all_peaks = gen_peak_structs(input_priors, run_info, column_titles)

    # MOVE TO generate priors?
    # # PROCESS INPUT PRIOR DATA, e.g. propagate any errors required
    # # need to convert errors in ln(width) and ln(power) to errors in width and power
    # for i in range(len(list_ln_values)):
    #     prior_peaks_list[i]["linewidth"] = np.exp(prior_peaks_list[i]["ln(width)"])
    #     prior_peaks_list[i]["linewidth error"] = abs(
    #         np.exp(prior_peaks_list[i]["ln(width)"] + prior_peaks_list[i]["ln(width) error"]) - np.exp(
    #             prior_peaks_list[i]["ln(width)"]))
    #
    #     # if virgo make it wider as linewidths a lot wider than for BiSON
    #     prior_peaks_list[i]["linewidth"] *= 5
    #
    #     # generate power errors
    #     prior_peaks_list[i]["power"] = np.exp(prior_peaks_list[i]["ln(power)"])
    #     prior_peaks_list[i]["power perc error"] = 100 * abs(
    #         np.exp(prior_peaks_list[i]["ln(power)"] + prior_peaks_list[i]["ln(power) error"])
    #         - np.exp(prior_peaks_list[i]["ln(power)"])) / prior_peaks_list[i]["power"]
    #
    #     # if virgo make it wider as linewidths a lot wider than for BiSON
    #     prior_peaks_list[i]["power"] *= 5

    return all_peaks


def find_pairs(all_peaks):
    """need to pair up the sets of l and n numbers to the ones close to each other. These pairs will be fitted together.
    e.g. l=0 n=N, l=2 n=N-1 and l=1 n=N, l=3 n=N-1"""
    pair_list = []
    for peak1 in all_peaks:
        if peak1.l == 0:
            for peak2 in all_peaks:
                if (peak2.l == 2) and (peak2.n == (peak1.n - 1)):
                    pair_peaks = pair_peaks_struc([peak1, peak2])
                    pair_list.append(pair_peaks)
        if peak1.l == 1:
            for peak2 in all_peaks:
                if (peak2.l == 3) and (peak2.n == (peak1.n - 1)):
                    pair_peaks = pair_peaks_struc([peak1, peak2])
                    pair_list.append(pair_peaks)

    return pair_list
