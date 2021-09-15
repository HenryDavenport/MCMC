from mcmc import MCMC

import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mcmc = MCMC()
    mcmc.number_of_processes = 1
    data_to_be_fitted = "bison-allsites-alldata-waverage-fill.fits"

    # the ratios of the heights of the m split components of the l=2 and l=3 modes
    mcmc.l2_m_scale = 0.625
    mcmc.l3_m_scale = 0.445
    # the mode visibilities
    mcmc.l1_visibility = 1.895
    mcmc.l2_visibility = 1.296
    mcmc.l3_visibility = 0.467
    mcmc.cadence = 40

    # This is the number of seconds in a year divided by the cadence
    # Total number of measurements in a year (BiSON cadence is 40 seconds)
    year_npts = 788400
    # length of each fit in years: (i.e. one year)
    no_years_fit = 1.0
    # number of measurements in each fit:
    length_of_fit = year_npts * no_years_fit
    # total number of measurements in the whole file:
    total_npts = 34909918
    shift = math.floor(0.25 * year_npts)
    number_of_fits = math.floor(total_npts / shift)
    for i in range(number_of_fits):
        power, freq, Pow_Win = mcmc.import_data_window(data_to_be_fitted, i * shift, i * shift + length_of_fit)
        folder_name = mcmc.create_directory()

        # first run low frequency (2000 to 3300 micro-Hz) peaks with just the scale parameter constant
        mcmc.parameters_kept_constant = ["scale"]
        average_splitting = mcmc.fit_all_peaks(power, freq, folder_name, pars_constant=mcmc.parameters_kept_constant,
                               freq_range=(2500, 3400), pow_win=Pow_Win)

        # Run high frequency peaks with scale and splitting constant.
        # splitting is set to average of splittings for all low frequency fits.

        parameters_kept_constant = ["scale", "splitting"]
        mcmc.fit_all_peaks(power, freq, folder_name, pars_constant=mcmc.parameters_kept_constant,
                           freq_range=(3400, 4000), splitting=average_splitting, pow_win=Pow_Win)