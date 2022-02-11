import math
import multiprocessing
import os
import time
import copy

import corner
import emcee
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


class MCMC:
    # useful parameters to change:
    # each l=0,2 or l=1,3 pair fitted with same or different linewidths
    Different_Widths = False
    # fit a simple linear background shift
    Fit_Background = True
    # Fit asymmetric lorentzians (not symmetric)
    Fit_Asymmetric = True
    # fit each peak in each l=0,2 and l=1,3 pair with different widths
    Different_Asymmetries = False

    parameters_kept_constant = ["scale"]

    priors_filename = "Priors.txt"
    Input_Priors_Columns = ["l", "n", "freq", "freq error", "splitting", "splitting error", "ln(width)",
                                 "ln(width) error", "ln(power)", "ln(power) error", "asymmetry", "asymmetry error",
                                 "background", "background error", "scale"]

    number_of_processes = 3
    cadence = 60
    burnin = 1000
    no_samples = 10000

    # the ratios of the heights of the m split components of the l=2 and l=3 modes
    l2_m_scale = 0.634
    l3_m_scale = 0.400

    # the mode visibilities
    l1_visibility = 1.505
    l2_visibility = 0.62
    l3_visibility = 0.075

    save_middle_peak_parameters_only = False

    # the guess of the background
    background_initial_guess = 500000

    # set to display the initial positions of all the walkers each peak run.
    check_priors = False

    # public functions
    def import_data(self, filename, start, end):
        """imports data and returns the power and frequency arrays"""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data

        print(repr(hdr))
        cad = self.cadence  # hdr['cadence'] #Spacing (in seconds) between data points
        vel = data[math.floor(start):math.floor(end)]

        npts = len(vel)
        time = np.arange(0, npts * cad, cad)

        # load velocity data and then FFT to get frequency-power spectrum of the Sun
        fill = np.count_nonzero(vel) / (1.0 * len(vel))
        # apply normalisation so the power is per unit frequency.
        power = (2.0 * cad / (fill * npts)) * ((abs(np.fft.rfft(vel))) ** 2)
        freq = 1e6 * np.fft.rfftfreq(npts, d=cad)
        return power, freq

    def import_data_window(self, filename, start, end):
        """imports data and returns the power and frequency arrays"""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data

        print(repr(hdr))

        cad = self.cadence  #Spacing (in seconds) between data points
        all_data = data[math.floor(start):math.floor(end)]

        # NOTE this is if BiSON data is in second axis which the example I used did have.
        vel = [x[1] for x in all_data]

        npts = len(vel)
        time2 = np.arange(0, npts * cad, cad)
        # plt.plot(time2, vel)
        # plt.show()

        window_fun = self.__create_window_function(vel)
        powWin = abs(np.fft.rfft(window_fun)) ** 2
        # load velocity data and then FFT to get frequency-power spectrum of the Sun
        fill = np.count_nonzero(vel) / (1.0 * len(vel))
        # apply normalisation so the power is per unit frequency.
        power = (2.0 * cad / (fill * npts)) * ((abs(np.fft.rfft(vel))) ** 2)
        freq = 1e6 * np.fft.rfftfreq(npts, d=cad)
        return power, freq, powWin

    def create_directory(self, label):
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

    def fit_all_peaks(self, power, freq, foldername, freq_range=(2000, 3500), pars_constant=parameters_kept_constant,
                      splitting=0.40, modes_per_fit=1, window_width=25, pow_win=None):
        """Sets up priors and frequency power spectrum and then runs the fitting over all pairs of peaks
        """
        # import the initial guess values from the priors filename
        prior_list_ln, priors_peak_dicts = self.import_priors(self.priors_filename,
                                                         self.Input_Priors_Columns)
        # find pairs of peaks in the priors data (e.g. adjacent l=0,2 peaks)
        paired_dict, unpaired_dict = self.__find_pairs(prior_list_ln, priors_peak_dicts)
        processes = []
        manager = multiprocessing.Manager()
        return_value_list = manager.list()
        lock = multiprocessing.Lock()
        # find all peaks in the frequency range requested:
        paired_freq_ordered = sorted(paired_dict, key=lambda x: x[0]["freq"])
        first_peak_index = next((i for i, item in enumerate(paired_freq_ordered) if freq_range[0] < item[0]["freq"]), None)
        last_peak_index = next((i-1 for i, item in enumerate(paired_freq_ordered) if freq_range[1] < item[0]["freq"]), None)

        # add the edge peaks below the lowest in the frequency range so all peaks in the range are fitted
        # only want to do this if number of modes per fit is larger than 1 and are only wanting to save middle
        if self.save_middle_peak_parameters_only and modes_per_fit > 1:
            first_peak_index -= math.floor(modes_per_fit/2)
            last_peak_index += math.floor(modes_per_fit/2)

        list_of_peaks_in_range = paired_freq_ordered[first_peak_index: last_peak_index+1]

        # this seperates the peaks into the groups that are to be fitted together, e.g. if want to fit the peaks either
        # side of peak of interest.
        peaks_fitted_together = []
        for i in range(0, len(list_of_peaks_in_range), 1):
            if i + modes_per_fit <= len(list_of_peaks_in_range):
                peaks_fitted_together.append(list_of_peaks_in_range[i:i + modes_per_fit])

        # cycle through the peaks and perform the fitting
        for peaks_to_be_fitted in peaks_fitted_together:
            peaks_in_range = []
            list_lns = []
            for [peak1, peak2] in peaks_to_be_fitted:
                peak = []
                l1 = peak1["l"]
                n1 = peak1["n"]
                l2 = peak2["l"]
                n2 = peak2["n"]
                list_lns.append([[l1, n1], [l2, n2]])
                peaks_in_range.append(peak1)
                peaks_in_range.append(peak2)
            # need to get right priors from list now:
            for i in range(len(peaks_in_range)):
                peaks_in_range[i]["asymmetry"] = 0.0
                peaks_in_range[i]["splitting"] = splitting

            # set up frequency range to be fitted
            LowerFreq1 = peaks_to_be_fitted[0][0]["freq"] - window_width
            UpperFreq1 = peaks_to_be_fitted[len(peaks_to_be_fitted)-1][1]["freq"] + window_width
            LowerPos = np.argmax(freq > LowerFreq1)
            UpperPos = np.argmax(freq > UpperFreq1)
            # make array length even (needed to use the window function for BiSON)
            if (UpperPos - LowerPos) % 2 != 0:
                UpperPos += 1

            # create freq and power arrays for fitting window
            freq_peaks = freq[LowerPos: UpperPos]
            power_peaks = power[LowerPos: UpperPos]
            # only needed for if BiSON fit make width of window even
            if pow_win is not None:
                N = len(freq_peaks)
                PowWin_short = []
                PowWin_short[0: math.floor(N / 2)] = pow_win[0: math.floor(N / 2)]
                PowWin_short[math.floor(N / 2):N] = pow_win[math.floor(N / 2): 0: -1]
            else:
                PowWin_short = None

            all_data = [freq_peaks, power_peaks]
            processes.append(multiprocessing.Process(target=self.run, args=(all_data, peaks_in_range, list_lns,
                                                                            peaks_in_range, foldername, lock,
                                                                            return_value_list, pars_constant, PowWin_short)))
            if len(processes) >= self.number_of_processes:
                for process in processes:
                    process.start()
                    time.sleep(1)
                for process in processes:
                    process.join()
                processes = []
                time.sleep(1)
        if len(processes) != 0:
            for process in processes:
                process.start()
                time.sleep(1)
            for process in processes:
                process.join()
            processes = []
            time.sleep(1)

        # calculate average splitting value in case needed for second run
        list_of_splitting = []
        for x in return_value_list:
            for y in x:
                if y < 1.0:
                    list_of_splitting.append(y)
        print(list_of_splitting)
        average_splitting = 0.40
        if len(list_of_splitting) != 0:
            average_splitting = sum(list_of_splitting) / len(list_of_splitting)
        return average_splitting

    def import_priors(self, location, column_titles):
        """imports priors from file location from table with columns given by column_titles
        returns a list of the l and n numbers in the priors and a list of dictionaries of the data for each peak"""
        # import priors from file as a list of lists (one row for each peak)
        input_priors = np.loadtxt(location)

        # add extra data for scale height of m split components
        input_data_proc = []
        for peak in input_priors:
            if peak[0] == 2.0:
                peak = np.append(peak, self.l2_m_scale)
            if peak[0] == 3.0:
                peak = np.append(peak, self.l3_m_scale)
            else:
                peak = np.append(peak, 0)
            input_data_proc.append(peak)
        input_priors = input_data_proc

        list_ln_values, prior_peaks_list = self.__gen_input_dict(input_priors, column_titles)

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

    def run(self, all_data, initial_guess_dicts, list_ln_numbers, prior_dict, folder_name, lock, return_dict,
            pars_constant, pow_Win):
        """runs fitting on pair of peaks in input"""
        freq_peaks1, power_peaks1 = all_data

        # run MCMC sampler and return samples and other parameters required for the save_all_output_data function
        Final_Values_Dict_non_log, Final_Fit_Values_log, log_samples, Par_List, Global_List, constants_dict \
            = self.__run_sampler(freq_peaks1, power_peaks1, folder_name, initial_guess_dicts, list_ln_numbers,
                                 lock, pars_constant, prior_dict, pow_Win)

        # generate all output graphs and save fit parameters to file.
        splitting_list = self.__save_all_output_data(Final_Fit_Values_log, Final_Values_Dict_non_log, Global_List,
                                                     Par_List, constants_dict, folder_name, freq_peaks1,
                                                     list_ln_numbers, lock, log_samples, power_peaks1)

        with lock:
            return_dict.append(splitting_list)
        return True


    # main control functions
    def __run_sampler(self, freq_1, power_1, folder_name, initial_guess_pars, list_ln_numbers, lock, pars_constant,
                      prior_dict, pow_Win):
        """
        runs the MCMC sampler

        :param freq_1: frequency array
        :param power_1: power spectrum array
        :param folder_name: folder for output
        :param initial_guess_pars: guess of parameters for fit read from file
        :param list_ln_numbers: list of the l and n numbers of the parameters to be fitted
        :param lock: prevents multiple processes writing to file simultaneously
        :param pars_constant: parameters that are kept constant in the fit
        :param prior_dict: gives errors on the fit values in the priors, used to define the width of the priors
        :param pow_Win: the window function used if the fit is for BiSON
        """
        initial_guess_pars_copy = [elem for elem in initial_guess_pars]
        list_ln_numbers_flat = [item for sublist in list_ln_numbers for item in sublist]

        # correct initial powers of the peaks
        if "power" not in pars_constant:
            varied_pars = self.__correct_powers(initial_guess_pars, freq_1, power_1)

        # correct background term:
        for i in range(len(initial_guess_pars)):
            initial_guess_pars[i]["background"] = self.background_initial_guess

        # separate into constants and parameters to be fitted.
        varied_pars, constants_dict = self.__find_constant_parameters(initial_guess_pars, pars_constant)
        # generates the list of parameter labels of the parameters that are going to be fitted.
        Par_List, Global_List = self.__gen_par_list(list_ln_numbers_flat, pars_constant)
        print("_________ LIST OF PARAMETERS TO BE FITTED _________")
        print(Par_List)
        print(Global_List)
        print("________PRIORS READ FROM FILE_________")
        prior_range_dicts = self.__generate_priors(prior_dict, initial_guess_pars_copy)
        print(prior_range_dicts)
        print("_________ LIST OF INITIAL PARAMETER VALUES _________")
        initial_guess_list = self.__gen_par_values(varied_pars, Par_List, Global_List)
        nwalkers = 100
        ndim = len(initial_guess_list)
        print(initial_guess_list)

        # add l and n number labels to prior dictionaries
        for i in range(len(prior_range_dicts)):
            l = list_ln_numbers_flat[i][0]
            n = list_ln_numbers_flat[i][1]
            prior_range_dicts[i]["l"] = l
            prior_range_dicts[i]["n"] = n

        # generate initial positions of walkers:
        prior_par_ranges_list = self.__gen_prior_ranges(prior_range_dicts, Par_List,
                                               Global_List)
        pos = initial_guess_list + prior_par_ranges_list * np.random.uniform(-1.0, 1.0, size=(nwalkers, ndim))

        # plot all initial walker positions if check priors True
        if self.check_priors:
            for i in range(len(initial_guess_list)):
                freqs = [listy[i] for listy in pos]
                plt.plot(freqs)
                plt.hlines(initial_guess_list[i], 0, len(freqs))
                plt.hlines(initial_guess_list[i]-prior_par_ranges_list[i], 0, len(freqs))
                plt.hlines(initial_guess_list[i]+prior_par_ranges_list[i], 0, len(freqs))
                plt.show()

        # initialise the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__log_probability,
                                        args=(
                                            freq_1, power_1, Par_List, Global_List, list_ln_numbers,
                                            constants_dict, initial_guess_list, prior_par_ranges_list, pow_Win))
        # repeats the sample procedure a maximum of x times
        x = 3
        for i in range(x):
            # Run burn-in phase
            state = sampler.run_mcmc(pos, self.burnin, progress=True)
            sampler.reset()
            # Actual run
            state2 = sampler.run_mcmc(state, self.no_samples, progress=True)

            # run 2 diagnostics to see if the sampling was successful, if not the sampling is re-run.
            acceptance_fracs = sampler.acceptance_fraction
            autocorrelationFail = True
            try:
                print(
                    "Mean autocorrelation time: {0:.3f} steps".format(
                        np.mean(sampler.get_autocorr_time())
                    )
                )
                lock.acquire()
                with open(f"{folder_name}/autocorrelation_times.txt", 'a') as f:
                    # f.write(identifier)
                    f.write(str(np.mean(sampler.get_autocorr_time())))
                    f.write("\n")
                    f.write(str(acceptance_fracs))
                    f.write("\n")
                lock.release()
                autocorrelationFail = False

            except:
                lock.acquire()
                with open(f"{folder_name}/autocorrelation_times.txt", 'a') as f:
                    # f.write(identifier)
                    f.write("fail")
                    f.write("\n")
                    f.write(str(acceptance_fracs))
                    f.write("\n")
                lock.release()
                print("Chain Not Converged")
                autocorrelationFail = True

            if (acceptance_fracs > 0.2).all() and not autocorrelationFail:
                print("Acceptance Fractions all over 0.2.")
                break
            else:
                if not (acceptance_fracs > 0.2).all():
                    print("Some Chains have very low acceptance fractions so rerun.")
                else:
                    pos = state

        # Get Samples
        samples = sampler.flatchain

        best_non_log_values = self.__get_errors(samples)[0]
        #samples[np.argmax(sampler.flatlnprobability)]
        Final_Values_Dict_non_log = self.__gen_dict_from_pars(best_non_log_values, constants_dict, Par_List, Global_List, list_ln_numbers)
        log_samples = []
        labels = [item for sublist in Par_List for item in sublist]
        labels.extend(Global_List)
        for sample in samples:
            single_log_sample = []
            for label, value in zip(labels, sample):
                if label == "linewidth" or label == "power":
                    single_log_sample.append(np.log(value))
                else:
                    single_log_sample.append(value)
            log_samples.append(single_log_sample)
        log_samples = np.array(log_samples)

        Final_Fit_Values_log = log_samples[np.argmax(sampler.flatlnprobability)]
        del sampler
        return Final_Values_Dict_non_log, Final_Fit_Values_log, log_samples, Par_List, Global_List, constants_dict

    def __save_all_output_data(self, Final_Fit_Values_log, Final_Values_Dict_non_log, Global_List, Par_List,
                               constants_dict, folder_name, freq_peaks1, list_ln_numbers, lock, log_samples,
                               power_peaks1):
        """Use final fit values and samples to generate graphs and save the fit parameters and errors to file""" 
        Par_List_Logs = []
        for peak in Par_List:
            temp = []
            for label in peak:
                if label == "linewidth" or label == "power":
                    temp.append("ln(" + label + ")")
                else:
                    temp.append(label)
            Par_List_Logs.append(temp)
        Global_List_Logs = []
        for label in Global_List:
            if "linewidth" in label or "power" in label:
                Global_List_Logs.append("ln(" + label + ")")
            else:
                Global_List_Logs.append(label)
        # convert power and width to log values:
        Final_Values_Dict_log = self.__gen_dict_from_pars(Final_Fit_Values_log, constants_dict, Par_List_Logs,
                                                          Global_List_Logs,
                                                          list_ln_numbers)
        print("_________ FINAL FIT VALUES _________")
        Par_List_Logs_errors = []
        for peak in Par_List_Logs:
            peak_temp = []
            for label in peak:
                peak_temp.append(label + " error")
            Par_List_Logs_errors.append(peak_temp)
        Final_Errors_Dict = self.__gen_dict_from_pars(self.__get_errors(log_samples)[1],
                                                      constants_dict,
                                                      Par_List_Logs_errors,
                                                      [s + " error" for s in Global_List_Logs], list_ln_numbers)
        # combine two dictionaries
        Final_Values_Errors = Final_Values_Dict_log
        for i in range(len(Final_Errors_Dict)):
            for key, value in Final_Errors_Dict[i].items():
                Final_Values_Errors[i][key] = value
        print(Final_Values_Errors)
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
        list_ln_numbers_flattened = [x for sublist in list_ln_numbers for x in sublist]
        identifier = "peaks_"
        for i in range(len(list_ln_numbers_flattened)):
            l1 = list_ln_numbers_flattened[i][0]
            n1 = list_ln_numbers_flattened[i][1]
            identifier += f"l{l1}n{n1}_"

        output_template = ["l", "n", "freq", "freq error", "ln(power)", "ln(power) error", "splitting",
                           "splitting error",
                           "scale",
                           "scale error", "ln(linewidth)", "ln(linewidth) error", "background", "background error",
                           "asymmetry",
                           "asymmetry error"]
        lock.acquire()
        # write fit parameters to file
        self.__write_to_file(f"{folder_name}/fit_parameters", output_template, Final_Values_Errors)
        lock.release()
        splitting_list = []
        for peak in Final_Values_Errors:
            if "splitting" in peak:
                splitting_list.append(peak["splitting"])

        # generate graph of data and fit and corner plots
        fig1, ax1 = plt.subplots()
        ax1.plot(freq_peaks1, power_peaks1, label="Simulated data")
        ax1.plot(freq_peaks1, self.all_peaks_model(freq_peaks1, Final_Values_Dict_non_log), label="MCMC Fit")
        ax1.legend()
        ax1.set_title("Simulated Data")
        ax1.set_ylabel("Power [$m^2s^{-2}Hz^{-1}$]")
        ax1.set_xlabel("Frequency [$\mu$Hz]")
        fig1.savefig(f"{folder_name}/OUTPUT_{identifier}_fit_plot_{current_time}.jpg", dpi=150)
        plt.close(fig1)
        # add peak labels and units to Par_List:
        # fix double use of template word (output_template and Template)
        print("_________ GENERATING CORNER PLOT _________")
        list_ln_numbers_flatten = [item for sublist in list_ln_numbers for item in sublist]
        final_labels = self.__add_parameter_units(Par_List, Global_List, list_ln_numbers_flatten)
        fig2 = corner.corner(
            log_samples[1::5], labels=final_labels, truths=Final_Fit_Values_log
        )
        fig2.savefig(f'{folder_name}/OUTPUT_{identifier}_corner_plot_{current_time}.jpg', dpi=150)
        plt.close(fig2)
        return splitting_list


    # functions used to process the input data and change its format (dictionary to list of input parameters etc)
    def __gen_input_dict(self, input_data, template):
        """ takes in input data from file (as a list of lists) and a template which gives a label to each element.
        generates a dictionary of elements"""
        total_dict = []
        list_ln = []
        for peak in input_data:
            peak_dict = dict(zip(template, peak))
            list_ln.append([int(peak_dict["l"]), int(peak_dict["n"])])
            total_dict.append(peak_dict)
        return list_ln, total_dict

    def __find_pairs(self, list_ln, info_dict):
        """need to pair up the sets of l and n numbers to the ones close to each other. These pairs will be fitted together.
        e.g. l=0 n=N, l=2 n=N-1 and l=1 n=N, l=3 n=N-1"""
        local_list = list_ln.copy()
        local_info_dict = info_dict.copy()
        i = 0
        unpaired_dict = []
        paired_dict = []
        while i < len(local_list):
            [l, n] = local_list[i]
            if l == 0:
                if [2, n - 1] in local_list:
                    index = local_list.index([2, n - 1])
                    paired_dict.append([local_info_dict[i], local_info_dict[index]])
                    del local_list[index]
                    del local_info_dict[index]
                else:
                    unpaired_dict.append(local_info_dict[i])
            elif l == 1:
                if [3, n - 1] in local_list:
                    index = local_list.index([3, n - 1])
                    paired_dict.append([local_info_dict[i], local_info_dict[index]])
                    del local_list[index]
                    del local_info_dict[index]
                else:
                    unpaired_dict.append(local_info_dict[i])
            elif l == 2:
                if [0, n + 1] in local_list:
                    index = local_list.index([0, n + 1])
                    paired_dict.append([local_info_dict[i], local_info_dict[index]])
                    del local_list[index]
                    del local_info_dict[index]
                else:
                    unpaired_dict.append(local_info_dict[i])
            elif l == 3:
                if [1, n + 1] in local_list:
                    index = local_list.index([1, n + 1])
                    paired_dict.append([local_info_dict[i], local_info_dict[index]])
                    del local_list[index]
                    del local_info_dict[index]
                else:
                    unpaired_dict.append(local_info_dict[i])
            else:
                unpaired_dict.append(local_info_dict[i])
            del local_list[i]
            del local_info_dict[i]
        return paired_dict, unpaired_dict

    def __correct_powers(self, peak_dict, freq, power):
        """normalisation varies a lot between different data sets. This function finds the average height in a
        region around a particular peak to estimate the power. This is used as the initial power of the MCMC fit."""
        peaks_dict_update = []
        for peak in peak_dict:
            lower_peak_bound = np.argmax(freq > peak["freq"] - 0.5 * peak["linewidth"])
            higher_peak_bound = np.argmax(freq > peak["freq"] + 0.5 * peak["linewidth"])
            data_around_peak = power[lower_peak_bound: higher_peak_bound]
            average_height = np.mean(data_around_peak)
            # convert to power
            power_estimate = average_height * (0.5 * np.pi * peak["linewidth"] * 1e-6)
            if peak["l"] == 1.0:
                power_estimate /= self.l1_visibility
                power_estimate /= 0.7

            elif peak["l"] == 2.0:
                power_estimate /= self.l2_visibility
                power_estimate /= 0.7

            elif peak["l"] == 3.0:
                power_estimate /= self.l3_visibility

            peak["power"] = power_estimate
            peaks_dict_update.append(peak)

        return peak_dict

    def __find_constant_parameters(self, peak_dict, par_constant):
        values_kept_constant = []
        values_to_be_varied = []
        for peak in peak_dict:
            single_peak_dict = {}
            for label in par_constant:
                if label in peak:
                    single_peak_dict[label] = peak[label]
                peak.pop(label, None)
            values_to_be_varied.append(peak)
            values_kept_constant.append(single_peak_dict)
        return values_to_be_varied, values_kept_constant

    def __gen_par_list(self, list1, parameters_constant):
        """function that takes the list of [l, n] numbers of the peaks to be fitted and works out what
    parameters need to be fitted. e.g. don't need to fit a splitting parameter for l=0 peak.
    also it looks at any global variables needed. These are variables which affect all peaks that are going to be fitted.
    e.g. background, asymmetry etc."""
        pars = []
        for [l, n] in list1:
            peak = []
            if l == 0:
                labels = ["freq", "power"]
                for label in [lab for lab in labels if lab not in parameters_constant]:
                    peak.extend([label])
            elif l == 1:
                labels = ["freq", "power", "splitting"]
                for label in [lab for lab in labels if lab not in parameters_constant]:
                    peak.extend([label])
            else:
                labels = ["freq", "power", "splitting", "scale"]
                for label in [lab for lab in labels if lab not in parameters_constant]:
                    peak.extend([label])
            if self.Different_Widths:
                peak.extend(["linewidth"])
            if self.Different_Asymmetries:
                peak.extend(["asymmetry"])
            pars.append(peak)
        # add global variables on end (these are the parameters which affect all the peaks e.g. if they all have
        # the same linewidth fitted and the background and asymmetry)
        global_list = []
        if not self.Different_Widths and "linewidth" not in parameters_constant:
            for [l, n] in list1:
                if l==1 or l==0:
                    global_list.extend(["linewidth " + str(l)+"_" + str(n)])
        if self.Fit_Background and "background" not in parameters_constant:
            global_list.extend(["background"])
        if self.Fit_Asymmetric and not self.Different_Asymmetries and "asymmetry" not in parameters_constant:
            for [l, n] in list1:
                if l == 1 or l == 0:
                    global_list.extend(["asymmetry " + str(l) + "_" + str(n)])
        return pars, global_list

    def __generate_priors(self, prior_dict, peak_dict):
        """function that takes as an input the dictionary of prior error values and the actual initial guess values
        outputs the list of priors for each peak with appropriate scaling."""
        scaling = 150
        priors = []
        for prior, peak in zip(prior_dict, peak_dict):
            if int(peak["l"]) == 0:
                Priors_l0 = {"freq": 3, "power": peak["power"],
                             "linewidth": peak["linewidth"],
                             "background": self.background_initial_guess, "asymmetry": 0.5}
                priors.append(Priors_l0)
            elif int(prior["l"]) == 1:
                Priors_l1 = {"freq": 3, "splitting": 0.4,
                             "power": peak["power"],
                             "linewidth": peak["linewidth"], "background": self.background_initial_guess,
                             "asymmetry": 0.5}
                priors.append(Priors_l1)
            elif int(prior["l"]) == 2:
                Priors_l2 = {"freq": 3, "splitting": 0.4,
                             "power": peak["power"],
                             "linewidth": peak["linewidth"], "scale": scaling * 0.01,
                             "background": self.background_initial_guess, "asymmetry": 0.5}
                priors.append(Priors_l2)

            elif int(prior["l"]) == 3:
                Priors_l3 = {"freq": 3, "splitting": 0.4,
                             "power": peak["power"],
                             "linewidth": peak["linewidth"], "scale": scaling * 0.01,
                             "background": self.background_initial_guess, "asymmetry": 0.5}
                priors.append(Priors_l3)

        # check and correct if the prior range goes negative (for all but asymmetry which can be negative)
        # if these are preventing your range going high enough then your initial priors need to be changed
        # this can be a problem for linewidths which are quite different between VIRGO and BiSON
        for prior, peak in zip(priors, peak_dict):
            for key in prior:
                if key in peak and "asymmetry" not in key:
                    if prior[key] > peak[key]:
                        prior[key] = copy.deepcopy(peak[key])
        return priors

    def __gen_prior_ranges(self, dict_list, listPar, list_Global):
        """ takes in list of names of parameters to be fitted
                 and returns an array with the value (from dict_list) for each one."""
        list_par_values = []
        for (keys, peak) in zip(listPar, dict_list):
            for key in keys:
                list_par_values.append(peak[key])
        # cycle through global variables
        if len(list_Global) != 0:
            for key in list_Global:
                if "linewidth" in key:
                    numbers = key[10:]
                    l, n = [float(s) for s in numbers.split("_")]
                    for peak in dict_list:
                        if peak["l"] == l and peak["n"] == n:
                            list_par_values.append(peak["linewidth"])
                            break

                if "asymmetry" in key:
                    numbers = key[10:]
                    l, n = [float(s) for s in numbers.split("_")]
                    for peak in dict_list:
                        if peak["l"] == l and peak["n"] == n:
                            list_par_values.append(peak["asymmetry"])
                            break

                if "background" in key:
                    # picks initial value from first input peak
                    # list_par_values.append(dict_list[0]["background"])
                    list_par_values.append(self.background_initial_guess)

        return np.array(list_par_values)


    def __gen_par_values(self, dict_list, listPar, list_Global):
        """ takes in list of names of parameters to be fitted
         and returns an array with the value (from dict_list) for each one."""
        list_par_values = []
        for (keys, peak) in zip(listPar, dict_list):
            for key in keys:
                list_par_values.append(peak[key])
        # cycle through global variables
        if len(list_Global) != 0:
            for key in list_Global:
                if "asymmetry" in key:
                    for peak in dict_list:
                        if "l" in peak:
                            peak_identifier = str(peak["l"])+"_"+str(peak["n"])
                            if peak_identifier in key:
                                list_par_values.append(peak["asymmetry"])
                                break
                        else:
                            # for if looking at prior ranges
                            list_par_values.append(peak["asymmetry"])
                            break
                if "linewidth" in key:
                    for peak in dict_list:
                        if "l" in peak:
                            peak_identifier = str(peak["l"]) + "_" + str(peak["n"])
                            if peak_identifier in key:
                                list_par_values.append(peak["linewidth"])
                                break
                        else:
                            # for if looking at prior ranges
                            list_par_values.append(peak["linewidth"])
                            break

                if "background" in key:
                    # picks initial value from first input peak
                    # list_par_values.append(dict_list[0]["background"])
                    list_par_values.append(self.background_initial_guess)

        return list_par_values

    def __gen_dict_from_pars(self, theta, constant_dict, Par_List, Global_List, list_ln):
        """ does opposite of __gen_par_values(): converts the input list of parameters and set of parameters
         back into a dictionary for easy recall for later functions"""
        Global_dict = {}
        if len(Global_List) != 0:
            Global_bit = theta[-len(Global_List):]
            for (key, value) in zip(Global_List, Global_bit):
                Global_dict[key] = value
            theta = theta[:len(theta)-len(Global_List)]

        # make
        counter = 0
        all_peaks_dicts = []
        list_ln_flatten = [item for sublist in list_ln for item in sublist]
        for (peak, ln) in zip(Par_List, list_ln_flatten):
            peak_dict = {}
            peak_dict["l"] = ln[0]
            peak_dict["n"] = ln[1]
            for key in peak:
                peak_dict[key] = theta[counter]
                counter += 1

            all_peaks_dicts.append(peak_dict)

        # add parameters that were kept constant back into dictionary
        for i in range(len(all_peaks_dicts)):
            for key, value in constant_dict[i].items():
                all_peaks_dicts[i][key] = value
            for key, value in Global_dict.items():
                if "linewidth" in key or "asymmetry" in key:
                    if all_peaks_dicts[i]["l"] == 1.0 or all_peaks_dicts[i]["l"] == 0.0:
                        peak_identifier = str(all_peaks_dicts[i]["l"])+"_"+str(all_peaks_dicts[i]["n"])
                    else:
                        peak_identifier = str(all_peaks_dicts[i]["l"]-2)+"_"+str(all_peaks_dicts[i]["n"]+1)

                    if peak_identifier in key:
                        if "linewidth" in key:
                            if "error" in key:
                                all_peaks_dicts[i]["ln(linewidth) error"] = value
                            elif "ln(" in key:
                                all_peaks_dicts[i]["ln(linewidth)"] = value
                            else:
                                all_peaks_dicts[i]["linewidth"] = value
                        if "asymmetry" in key:
                            if "error" in key:
                                all_peaks_dicts[i]["asymmetry error"] = value
                            else:
                                all_peaks_dicts[i]["asymmetry"] = value
                else:
                    all_peaks_dicts[i][key] = value

        return all_peaks_dicts

    def __create_window_function(self, vel_data):
        window_function = [0.0 if i == 0.0 else 1.0 for i in vel_data]
        return window_function

    # functions used by the MCMC sampler
    def __log_probability(self, theta, *args):
        """the probability function. This is directly called by emcee, the parameters that are varied are in theta."""
        freq, power, Par_List, Global_List, list_ln, constant_dict, initial_guess_list, prior_par_ranges_list, PowWin_short = args

        # convert input parameters to dictionary
        theta_dict = self.__gen_dict_from_pars(theta, constant_dict, Par_List, Global_List, list_ln)
        lp = self.__log_prior(theta, initial_guess_list, prior_par_ranges_list)
        # if log of prior is neg infinity then full log probability function must be too
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__log_likelihood(theta_dict, freq, power, PowWin_short)

    def __log_prior(self, thetas, inputs, priors):
        """ prior function called by log_probability().
        Returns negative infinity if any of the parameter values
            are outside the priors. Uses uniform priors currently."""

        for (theta, input, prior) in zip(thetas, inputs, priors):
            if abs(theta - input) > prior:
                return -np.inf
        return 0

    def __log_likelihood(self, dict1, freq, measured, PowWin_short = None ):
        model = self.all_peaks_model(freq, dict1, PowWin_short)
        return -np.sum(np.log(model) + measured / model)

    def lorentz(self, freq, central_freq, width, pow, asymmetry):
        """function returns the values of the self.lorentzian peak across the freq range inputted"""
        height = pow / (0.5 * np.pi * width * 1e-6)
        xi = (2 / width) * (freq - central_freq)
        model = (height / (1 + xi ** 2)) * ((1 + asymmetry * xi) ** 2 + asymmetry ** 2)
        return model

    def background(self, height):
        """ returns the background function. At the moment uniform linear offset"""
        return height

    def all_peaks_model(self, freq, theta_dict, PowWin_short=None):
        """returns the complete model with parameters given by input theta_dict"""
        model = 0
        # add background
        if self.Fit_Background:
            if "background" in theta_dict[0]:
                shift = self.background(theta_dict[0]["background"])
        else:
            shift = 0

        for peak_Dict in theta_dict:
            if not self.Fit_Asymmetric:
                asymmetry = 0
            if self.Fit_Asymmetric:
                asymmetry = peak_Dict["asymmetry"]
            if peak_Dict["l"] == 0:
                # create 1 peak
                model += self.lorentz(freq, peak_Dict["freq"], peak_Dict["linewidth"],
                                 peak_Dict["power"], asymmetry)

            elif peak_Dict["l"] == 1:
                # create 2 peaks for m+-1:
                visibility = self.l1_visibility
                model += self.lorentz(freq, peak_Dict["freq"] + peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / 2, asymmetry)
                model += self.lorentz(freq, peak_Dict["freq"] - peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / 2, asymmetry)

            elif peak_Dict["l"] == 2:
                # create 3 peaks for m=+-2, 0
                # create m+-2 peaks:
                visibility = self.l2_visibility
                scaling = 2 + peak_Dict["scale"]
                model += self.lorentz(freq, peak_Dict["freq"] + 2 * peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / scaling, asymmetry)
                model += self.lorentz(freq, peak_Dict["freq"] - 2 * peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / scaling, asymmetry)
                # m = 0 is smaller by factor "scale":
                model += peak_Dict["scale"] * self.lorentz(freq, peak_Dict["freq"], peak_Dict["linewidth"],
                                                      visibility * peak_Dict["power"] / scaling, asymmetry)

            elif peak_Dict["l"] == 3:
                # create 4 peaks for m+-1, m+-3
                # create m+-3 peaks:
                visibility = self.l3_visibility
                scaling = 2 + 2 * peak_Dict["scale"]
                model += self.lorentz(freq, peak_Dict["freq"] + 3 * peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / scaling, asymmetry)
                model += self.lorentz(freq, peak_Dict["freq"] - 3 * peak_Dict["splitting"], peak_Dict["linewidth"],
                                 visibility * peak_Dict["power"] / scaling, asymmetry)

                # m = +-1 is smaller by factor "scale":
                model += peak_Dict["scale"] * self.lorentz(freq, peak_Dict["freq"] + peak_Dict["splitting"],
                                                      peak_Dict["linewidth"],
                                                      visibility * peak_Dict["power"] / scaling, asymmetry)
                model += peak_Dict["scale"] * self.lorentz(freq, peak_Dict["freq"] - peak_Dict["splitting"],
                                                      peak_Dict["linewidth"],
                                                      visibility * peak_Dict["power"] / scaling, asymmetry)

        if PowWin_short is not None:
            n1 = len(model)
            n2 = len(PowWin_short)
            Temp = (np.fft.irfft(model) / (2.0 * n1)) * (np.fft.irfft(PowWin_short)) / (2.0 * n2)
            model = abs(np.fft.rfft(Temp))

        # add background
        model += shift
        return model

    # functions used to generate the output graphs and write fit parameters to file
    def __write_to_file(self, filename, output_template, fit_values):
        """ writes the data in fit_values to filename with order of columns given by output template.
        background information is stored in position of dictionary."""
        if self.save_middle_peak_parameters_only:
            pos_centre_peak1 = math.floor(len(fit_values)/2 - 1)
            fit_values = fit_values[pos_centre_peak1: pos_centre_peak1+2]

        if "background" in fit_values[0]:
            background = fit_values[0]["background"]
            if "background error" in fit_values[0]:
                background_error = fit_values[0]["background error"]
            else:
                background_error = [0, 0]
        else:
            background = 0
            background_error = [0, 0]

        for peak_values in fit_values:
            output_list = []
            for key in output_template:
                if key in peak_values.keys():
                    if key.endswith("error"):
                        for error in peak_values[key]:
                            output_list.append(str(error))
                    else:
                        output_list.append(str(peak_values[key]))
                elif key == "background":
                    output_list.append(str(background))
                elif key == "background error":
                    for error in background_error:
                        output_list.append(str(error))
                else:
                    if key.endswith("error"):
                        output_list.append("0\t0")
                    else:
                        output_list.append("0")
            seperator = "\t"
            output_string = seperator.join(output_list)
            with open(f'{filename}.txt', 'a') as f:
                f.write(output_string)
                f.write("\n")
        return True

    def __get_errors(self, samples):
        """uses percentiles to get errors on the sample best values."""
        errors = []
        values = []
        for i in range(len(samples[0])):
            percs= np.percentile(samples[:, i], [16, 50, 84])
            values.append(percs[1])
            q = np.diff(percs)
            errors.append([q[0], q[1]])
        return values, errors

    def __add_parameter_units(self, par_list, global_list, list_ln_numbers):
        """adds units to list of parameters this is so the plots have the right units.
            Also it changes the power and width labels to their natural logarithms."""
        labels_final = []
        for [l, n], labels in zip(list_ln_numbers, par_list):
            for label in labels:
                if label == "linewidth" or label == "power":
                    label = "ln(" + label + ")"
                if label == "freq" or label == "splitting" or label == "ln(linewidth)":
                    label += " [$\mu$Hz]"
                elif label == "ln(power)":
                    label += "[$m^2s^{-2}Hz^{-1}$]"
                label += f" (l={l})"
                labels_final.append(label)
        if len(global_list) != 0:
            for label in global_list:
                if "background" in label:
                    label += " [$m^2s^{-2}Hz^{-1}$]"
                if "linewidth" in label:
                    label = "ln("+label+")"
                    label += " [$\mu$Hz]"
                labels_final.append(label)
        return labels_final
