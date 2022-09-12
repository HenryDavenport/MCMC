import numpy as np
from import_data import import_data
from structures import Run_Info, Fit_Info, Parameter

Input_Priors_Columns = ["l", "n", "freq", "freq error", "splitting", "splitting error", "ln(width)",
                        "ln(width) error", "ln(power)", "ln(power) error", "asymmetry", "asymmetry error",
                        "background", "background error", "scale"]

# functions used to process the input data and change its format (dictionary to list of input parameters etc)
def gen_input_dict(input_data, template):
    """ takes in input data from file (as a list of lists) and a template which gives a label to each element.
    generates a dictionary of elements"""
    total_dict = []
    list_ln = []
    for peak in input_data:
        peak_dict = dict(zip(template, peak))
        list_ln.append([int(peak_dict["l"]), int(peak_dict["n"])])
        total_dict.append(peak_dict)
    return list_ln, total_dict

def import_priors(location, column_titles):
    """imports priors from file location from table with columns given by column_titles
    returns a list of the l and n numbers in the priors and a list of dictionaries of the data for each peak"""
    # import priors from file as a list of lists (one row for each peak)
    input_priors = np.loadtxt(location)


    list_ln_values, prior_peaks_list = gen_input_dict(input_priors, column_titles)

    # PROCESS INPUT PRIOR DATA, e.g. propagate any errors required
    # need to convert errors in ln(width) and ln(power) to errors in width and power
    for i in range(len(list_ln_values)):
        prior_peaks_list[i]["linewidth"] = np.exp(prior_peaks_list[i]["ln(width)"])
        prior_peaks_list[i]["linewidth error"] = abs(
            np.exp(prior_peaks_list[i]["ln(width)"] + prior_peaks_list[i]["ln(width) error"]) - np.exp(
                prior_peaks_list[i]["ln(width)"]))

        # if virgo make it wider as linewidths a lot wider than for BiSON
        prior_peaks_list[i]["linewidth"] *= 5

        # generate power errors
        prior_peaks_list[i]["power"] = np.exp(prior_peaks_list[i]["ln(power)"])
        prior_peaks_list[i]["power perc error"] = 100 * abs(
            np.exp(prior_peaks_list[i]["ln(power)"] + prior_peaks_list[i]["ln(power) error"])
            - np.exp(prior_peaks_list[i]["ln(power)"])) / prior_peaks_list[i]["power"]

        # if virgo make it wider as linewidths a lot wider than for BiSON
        prior_peaks_list[i]["power"] *= 5

    return list_ln_values, prior_peaks_list

location = "Main_Fits_BiSON_8640d_lbest_UseInSolarCycle.dat"
input_priors = np.loadtxt(location)
list_ln_values, prior_peaks_list = gen_input_dict(input_priors, Input_Priors_Columns)

background = Parameter(50000, 0, 100000, False, False)
run_info = Run_Info(background)
filename = "GOLF_bw_rw_pm3_960411_200830_Sync_RW_dt60F_v1.fits"
data_year = 31536000/run_info.cadence
power, freq = import_data(filename, 0, data_year, run_info.cadence)


def correct_powers(peak_dict, freq, power, run_info):
    """normalisation varies a lot between different data sets. This function finds the average height in a
    region around a particular peak to estimate the power. This is used as the initial power of the MCMC fit."""
    peaks_dict_update = []
    for peak in peak_dict:
        lower_peak_bound = np.argmax(freq > peak["freq"] - 3 * np.exp(peak["ln(width)"]))
        higher_peak_bound = np.argmax(freq > peak["freq"] + 3 * np.exp(peak["ln(width)"]))
        data_around_peak = power[lower_peak_bound: higher_peak_bound]
        average_height = np.mean(data_around_peak)
        # convert to power
        power_estimate = average_height * (0.5 * np.pi * np.exp(peak["ln(width)"]) * 1e-6)
        if peak["l"] == 1.0:
            power_estimate /= run_info.l1_visibility
            power_estimate /= 0.7

        elif peak["l"] == 2.0:
            power_estimate /= run_info.l2_visibility
            power_estimate /= 0.7

        elif peak["l"] == 3.0:
            power_estimate /= run_info.l3_visibility

        peak["ln(power)"] = np.log(power_estimate)
        peak["ln(power) error"] = abs(0.5*np.log(power_estimate))
        peaks_dict_update.append(peak)

    return peak_dict

# prior_peaks_list = correct_powers(prior_peaks_list, freq, power, run_info)
print(prior_peaks_list)
with open("priors.txt", "w") as f:
    for peak in prior_peaks_list:
        line = []
        multiplier1 = 40
        multiplier2 = 20
        multiplier3 = 20
        multiplier4 = 1
        multiplier5 = 20

        line.append(str(peak["l"])+"\t")
        line.append(str(peak["n"])+"\t")
        line.append(str(peak["freq"])+"\t")
        line.append(str(peak["freq"]-multiplier1*peak["freq error"])+"\t")
        line.append(str(peak["freq"]+multiplier1*peak["freq error"])+"\t")
        line.append(str(peak["splitting"]) + "\t")
        line.append(str(peak["splitting"] - multiplier2*peak["splitting error"]) + "\t")
        line.append(str(peak["splitting"] + multiplier2*peak["splitting error"]) + "\t")
        line.append(str(peak["ln(width)"]) + "\t")
        line.append(str(peak["ln(width)"] - multiplier3*peak["ln(width) error"]) + "\t")
        line.append(str(peak["ln(width)"] + multiplier3*peak["ln(width) error"]) + "\t")
        line.append(str(peak["ln(power)"]) + "\t")
        line.append(str(peak["ln(power)"] - multiplier4*peak["ln(power) error"]) + "\t")
        line.append(str(peak["ln(power)"] + multiplier4*peak["ln(power) error"]) + "\t")
        line.append(str(peak["asymmetry"]) + "\t")
        line.append(str(peak["asymmetry"] - multiplier5*peak["asymmetry error"]) + "\t")
        line.append(str(peak["asymmetry"] + multiplier5*peak["asymmetry error"]) + "\t")

        f.write("".join(line)+"\n")
