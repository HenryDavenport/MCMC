from mcmc import MCMC

import math

if __name__ == '__main__':
    from mcmc import MCMC

    mcmc = MCMC()
    filename = "GOLF_bw_rw_pm3_960411_200830_Sync_RW_dt60F_v1.fits"
    mcmc.number_of_processes = 1

    # the ratios of the heights of the m split components of the l=2 and l=3 modes
    mcmc.l2_m_scale = 0.634
    mcmc.l3_m_scale = 0.400
    # the mode visibilities
    mcmc.l1_visibility = 1.505
    mcmc.l2_visibility = 0.62
    mcmc.l3_visibility = 0.075

    mcmc.cadence = 60

    mcmc.save_middle_peak_parameters_only = True

    # This is the number of seconds in a year divided by the cadence
    # Total number of measurements in a year
    year_npts = 525600
    # length of each fit in years: (i.e. one year)
    no_years_fit = 1.0
    # number of measurements in each fit:
    length_of_fit = year_npts * no_years_fit
    # total number of measurements in the whole file:
    total_npts = 12827520
    number_of_fits = math.floor(total_npts / length_of_fit)
    shift = 0.25 * year_npts
    for i in range(number_of_fits):
        power, freq = mcmc.import_data(filename, i * shift, i * shift + length_of_fit)
        folder_name = mcmc.create_directory()
        # first run low frequency (2000 to 3300 micro-Hz) peaks with just the scale parameter constant
        mcmc.parameters_kept_constant = ["scale"]
        average_splitting = mcmc.fit_all_peaks(power, freq, folder_name, pars_constant=mcmc.parameters_kept_constant,
                                               freq_range=(2000, 3400), modes_per_fit=3)
        # Run high frequency peaks with scale and splitting constant.
        # splitting is set to average of splittings for all low frequency fits.
        parameters_kept_constant = ["scale", "splitting"]
        mcmc.fit_all_peaks(power, freq, folder_name, pars_constant=mcmc.parameters_kept_constant,
                           freq_range=(3300, 4000), splitting=average_splitting, modes_per_fit=3)
