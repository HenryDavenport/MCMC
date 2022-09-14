"""
output.py
Create all output files and save them.
Creates:
(a) a graph of each fit alongside the actual data.
(b) a corner plot showing the distribution of the samples collected.
(c) the fit paramaters which are saved to a .txt file called "fit_data.txt"
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import corner
import model

def create_directory(label):
    """
    Creates directory for output of fitting.
    :param string label: used to label the output folder e.g. with source of data "GOLF"
    :return: name of folder created
    """
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)
    # define the name of the directory to be created as combination of label and current time
    folder_name = f"Output_{label}_{current_time}"
    try:
        os.mkdir(folder_name)
    except OSError:
        print("Creation of the directory %s failed" % folder_name)
    else:
        print("All Output Stored in Folder: %s " % folder_name)
    return folder_name


def get_value_and_errors(samples):
    """
    Uses the median to get the value of a parameter and uses the 16th and 84th percentile to
    get the upper and lower errors (this is plus/minus 1 standard deviation).
    :param samples: The list of the sampled parameter values.
    :return: A list of the values of each parameter and their upper and lower errors.
    """
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
    """
    Adds units to list of parameters this is so the plots have the right units.
    Also it changes the power and width labels to their natural logarithms.
    :param par_list: list of parameter labels e.g. "asymmetry", "ln_power" etc.
    :return: The list of parameters now with the units appended to the parameter name.
    """
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
    """
    Creates a corner pllot based on the samples and saves it to file.
    :param samples: A list of all the samples of the fit parameters.
    :param labels: The labels used for each parameter
    :param directory: The directory the plot will be saved to
    :param identifier: The label for filename (not used currently)
    :param current_time: The time which is used to create the filename.
    :return:
    """
    labels_units = add_parameter_units(labels)
    fig2 = corner.corner(
        samples, labels=labels_units
    )
    fig2.savefig(f'{directory}/OUTPUT_corner_plot_{current_time}.jpg', dpi=100)
    plt.close(fig2)


def save_output(samples, labels, directory, fit_info, freq, power):
    """
    Function that is called by main.py to save all output - graphs and fit values.
    :param samples: The list of samples of the fit parameters returned by emcee
    :param labels: The list of the labels for each fit parameter e.g. "asymmetry" etc.
    :param directory: The name of the directory all output should be saved in.
    :param fit_info: Instance of class Fit_Info used to store all parameters
    :param freq: The frequency array
    :param power: The power array (e.g. from GOLF).
    :return:
    """
    current_time = time.time()
    identifier = "DATA"
    # create and save corner plots
    create_corner_plot(samples, labels, directory, identifier, current_time)
    # get the values and errors on the parameters using the samples
    values, lower_error, upper_error = get_value_and_errors(samples)

    # set all the final fit values into the fit_info class.
    # This makes it easy to write all the fit parameters to file and
    # generate the model in order to output a graph of the fit.
    fit_info.set_all_final_fit_variables(values, lower_error, upper_error)
    generate_graph(freq, power, fit_info, directory, current_time)
    # write the fit values to file in the .txt file "fit_data.txt"
    write_to_file(fit_info, directory)

def generate_graph(freq, power, fit_info, directory, current_time):
    """
    generates and saves a graph showing the final fit over the measured data for comparison.
    :param freq: frequency array
    :param power: power array (e.g. from GOLF)
    :param fit_info: the instance of class Fit_Info that contains all the parameter values and errors
    :param directory: directory name to save graph
    :param current_time: the time used to generate part of the output file name
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(freq, power)
    ax.plot(freq, model.all_peaks_model(freq, fit_info))
    fig.savefig(f'{directory}/OUTPUT_graph_{current_time}.jpg', dpi=100)
    plt.close(fig)

def write_to_file(fit_info, directory):
    """
    write all parameters to file in .txt file "fit_data.txt"
    :param fit_info: the instance of class Fit_Info that contains all the parameter values and errors
    :param directory:
    :return:
    """
    # function that returns the output in the correct order to write to file
    lines = fit_info.output()
    with open(f'{directory}/fit_data.txt', 'a') as f:
        seperator = "\t"
        for line in lines:
            output_string = seperator.join(line)
            f.write(output_string)
            f.write("\n")
