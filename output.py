import time
import numpy as np
import matplotlib.pyplot as plt
import os
import corner
import math
from dataclasses import fields

import model
from structures import Parameter

def create_directory(label):
    """Creates directory for output of fitting"""
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)
    folder_name = f"Output_{label}_{current_time}"
    # define the name of the directory to be created
    try:
        os.mkdir(folder_name)
    except OSError:
        print("Creation of the directory %s failed" % folder_name)
    else:
        print("All Output Stored in Folder: %s " % folder_name)
    return folder_name

#
def get_value_and_errors(samples):
    """uses percentiles to get errors on the sample best values."""
    lower_error = []
    upper_error = []
    values = []
    for i in range(len(samples[0])):
        percs = np.percentile(samples[:, i], [16, 50, 84])
        values.append(percs[1])
        q = np.diff(percs)
        lower_error.append(q[0])
        upper_error.append(q[1])
    return values, lower_error, upper_error

def add_parameter_units(par_list):
    """adds units to list of parameters this is so the plots have the right units.
        Also it changes the power and width labels to their natural logarithms."""
    labels_final = []
    for label in par_list:
        if "freq" in label or "splitting" in label:
            label += " [$\mu$Hz]"
            labels_final.append(label)

        elif "ln_power" in label or "background" in label:
            label += "[$m^2s^{-2}Hz^{-1}$]"
            labels_final.append(label)
        else:
            labels_final.append(label)
    return labels_final

def create_corner_plot(samples, labels, directory, identifier, current_time):
    labels_units = add_parameter_units(labels)
    fig2 = corner.corner(
        samples, labels=labels_units
    )
    fig2.savefig(f'{directory}/OUTPUT_corner_plot_{current_time}.jpg', dpi=100)
    plt.close(fig2)


def save_output(samples, labels, directory, fit_info, freq, power):
    current_time = time.time()
    identifier = "DATA"
    create_corner_plot(samples, labels, directory, identifier, current_time)
    values, lower_error, upper_error = get_value_and_errors(samples)
    fit_info.set_all_final_fit_variables(values, lower_error, upper_error)

    fig, ax = plt.subplots()

    ax.plot(freq, power)
    ax.plot(freq, model.all_peaks_model(freq, fit_info))
    fig.savefig(f'{directory}/OUTPUT_graph_{current_time}.jpg', dpi=100)
    plt.close(fig)

    write_to_file(fit_info, directory)

def write_to_file(fit_info, directory):
    lines = fit_info.output()
    with open(f'{directory}/fit_data.txt', 'a') as f:
        seperator = "\t"
        for line in lines:
            output_string = seperator.join(line)
            f.write(output_string)
            f.write("\n")
