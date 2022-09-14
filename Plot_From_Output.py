"""
Plot_From_Output.py
Plots the data which is output from a fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from import_priors import gen_peak_structs
from run_settings import Run_Settings
from structures import Fit_Info, Parameter, Pair_Peaks
from model import all_peaks_model

# file location for fit parameters
location = "Output_GOLF_13_10_34/fit_data.txt"
# import data into matrix
all_data = np.loadtxt(location)

# column titles in output file
column_titles = ["l", "n", "freq", "freq lower error", "freq upper error", "splitting",
                 "splitting lower error", "splitting upper error", "ln(width)", "ln(width) lower error",
                 "ln(width) upper error", "ln(power)", "ln(power) lower error", "ln(power) upper error",
                 "asymmetry", "asymmetry lower error", "asymmetry upper error", "m_scale",
                 "background", "background lower error", "background upper error"]

# contains information about the m_scale splitting that is used in the fit
run_settings = Run_Settings

# convert the matrix into the right format (List[Pair_Peaks])
list_peaks = gen_peak_structs(all_data, run_settings, column_titles)

# plot only first 2 peaks in the list
pair_peak = Pair_Peaks(list_peaks[:2])

# add the background values to a Parameter in order to add to the Fit_Info class
lower_bound_background = all_data[0][-2]
expected_background_value = all_data[0][-3]  # the approximate expected value of the background in the fit
upper_bound_background = all_data[0][-1]
background = Parameter(expected_background_value, lower_bound_background, upper_bound_background,
                       constant=False, shared=False)

# contains all information required for the fit.
fit_info = Fit_Info(peak_pair_list=[pair_peak], background = background,
                    visibilities = run_settings.visibilities)

# frequency array
freq = np.linspace(2065, 2110, 10000)

fig, ax = plt.subplots()
ax.plot(freq, all_peaks_model(freq, fit_info))
ax.set_xlabel("Frequency [$\mu$ Hz]")
ax.set_ylabel("Power")
plt.show()
