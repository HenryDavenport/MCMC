""" structures.py
Contains all the classes which define the way in which the fit information is stored and retrieved.
"""

from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class Parameter:
    """ Class to define all variables for each fit parameter
    e.g. one for Asymmetry, frequency, ln(power) etc"""
    value: float  # expected value of parameter from priors
    lower: float  # lower bound of value - sets lower bound on range explored by MCMC walker
    upper: float  # upper bound of value - sets upper bound on range explored by MCMC walker
    constant: bool  # boolean sets if this parameter is kept constant e.g. not fitted but set to value given as input
    shared: bool  # boolean sets if same parameter value is fitted for neighbouring peak (e.g. within each l=0/2 and l=1/3 pair)


@dataclass
class Peak_Structure:
    """ Class contains all the data about a particular peak.
    Contains all parameters for a peak and the upper and lower bounds on the parameter values"""
    l: int
    n: int
    freq: Parameter
    splitting: Parameter
    ln_linewidth: Parameter
    ln_power: Parameter
    asymmetry: Parameter
    m_scale: float

    def __init__(self, peak_dict, run_info):
        """
        Unpacks the prior values from the peak_dict and run_info and populates the structure above.
        Also sets which parameter values are:
        (a) constant/not fitted. These are just set to the prior value and not modified during the fit
        (b) shared between pairs of peaks. For example the same asymmetry value is often fitted for pairs of peaks
                in each l=0/2 and l=1/3 pair.

        :param peak_dict: dictionary with all values and errors for the given peak (see below for field names)
        :param Run_Info run_info: class containing global information about fit - e.g. scaling of m-split components
        """

        # set the l and n number of the peaks in the structure
        self.l = int(peak_dict["l"])
        self.n = int(peak_dict["n"])

        # set the frequency parameter
        freq = Parameter(peak_dict["freq"], peak_dict["freq lower error"],
                         peak_dict["freq upper error"], constant=False, shared=False)
        self.freq = freq

        # set splitting parameter
        if peak_dict["splitting"] == 0:
            # if splitting = 0 in input then this parameter is not fitted so set to constant
            kept_constant = True
        else:
            # else splitting is fitted so not kept constant
            kept_constant = False
        splitting = Parameter(peak_dict["splitting"], peak_dict["splitting lower error"],
                              peak_dict["splitting upper error"], constant=kept_constant, shared=False)
        self.splitting = splitting

        # set ln(linewidth) parameter
        ln_linewidth = Parameter(peak_dict["ln(width)"], peak_dict["ln(width) lower error"],
                                 peak_dict["ln(width) upper error"], constant=False, shared=False)
        self.ln_linewidth = ln_linewidth

        # set ln(power) parameter
        ln_power = Parameter(peak_dict["ln(power)"], peak_dict["ln(power) lower error"],
                              peak_dict["ln(power) upper error"], constant=False, shared=False)
        self.ln_power = ln_power

        # set asymmetry parameter, note that the value of asymmetry is shared by both peaks in the pair
        asymmetry = Parameter(peak_dict["asymmetry"], peak_dict["asymmetry lower error"],
                              peak_dict["asymmetry upper error"], constant=False, shared=True)
        self.asymmetry = asymmetry

        # set the m component scaling depending on l value.
        if self.l == 2:
            self.m_scale = run_info.l2_m_scale
        else:
            self.m_scale = run_info.l3_m_scale

    def output_data(self):
        """
        Returns a list of all parameters values, upper and low bounds that are held in structure.
        The list is in the order given by the order of the parameter fields shown above.

        :return: list of all parameters values, upper and low bounds that are held in structure.
        """
        output_list = []
        for parameter_field in fields(self):
            parameter = getattr(self, parameter_field.name)
            if isinstance(parameter, Parameter):
                output_list.append(str(parameter.value))
                output_list.append(str(parameter.lower))
                output_list.append(str(parameter.upper))
            else:
                output_list.append(str(parameter))
        return output_list


@dataclass
class Pair_Peaks:
    """ Data structure contains all fit parameter values for a pair of peaks.
    Contains methods used to get and set fit parameters during the fitting."""
    peak_list: List[Peak_Structure] # list of peak data, one Peak_Structure type for each peak.

    def get_fit_variables(self):
        """ Returns an array of parameter values for all parameters
        that are to be fitted in the pair of peaks.
        The parameters in the array are in the following order:
            (a) First the parameters for each peak in turn which are both not constant and not shared between the peaks.
                (shared means that the same parameter value is used for both peaks in the peak pair)
            (b) Secondly all parameters which are both not constant and are shared between peaks in the peak pair.
        This array is used to pass to the emcee library which requires the fit parameters to be in an array.
        This order makes it easy to convert between this array and all the parameter values in the structure. """

        par_list = []  # list of parameter values
        lower_list = []  # list of lower bounds on parameter values
        upper_list = []  # list of upper bounds on parameter values
        par_labels = []  # list of parameter labels associated with the parameter values (used for graphs)

        # first add parameters which are not shared between the peaks to the list
        for peak in self.peak_list:
            for parameter_field in fields(peak):
                parameter = getattr(peak, parameter_field.name)
                if isinstance(parameter, Parameter):
                    if not parameter.constant and not parameter.shared:
                        par_labels.append(str(parameter_field.name)+" l={}, n={}".format(peak.l, peak.n))
                        par_list.append(parameter.value)
                        lower_list.append(parameter.lower)
                        upper_list.append(parameter.upper)

        # add variables which are shared by both peaks
        for parameter_field in fields(self.peak_list[0]):
            parameter = getattr(self.peak_list[0], parameter_field.name)
            if isinstance(parameter, Parameter):
                if not parameter.constant and parameter.shared:
                    par_labels.append(str(parameter_field.name) + " l={}, n={}".format(peak.l, peak.n))
                    par_list.append(parameter.value)
                    lower_list.append(parameter.lower)
                    upper_list.append(parameter.upper)

        return par_list, lower_list, upper_list, par_labels

    def set_fit_variables(self, par_list):
        """
        Opposite to get_fit_variables.
        Input is a list of parameter values in the order that is returned in get_fit_variables().
        Sets all the parameter values in the peaks in the peak_list.
        Returns the remaining parameter values in the par_list.
        This is for two reasons:
        Firstly it means multiple pairs of peaks can be fitted at once.
        Secondly the background parameter value is stored in the Fit_Info type so must be returned to be stored there.

        :param List[float] par_list: list of fit parameters
        :return: Remaining parameter values in the par_list after setting all values for this pair of peaks.

        """
        i = 0
        # par_list is a list of parameters in a specific order
        # add variables that are only peak specific.
        for peak in self.peak_list:
            for parameter_field in fields(peak):
                parameter = getattr(peak, parameter_field.name)
                if isinstance(parameter, Parameter):
                    if not parameter.constant and not parameter.shared:
                        parameter.value = par_list[i]
                        i += 1

        initial_i = i
        # add variables which are shared by both peaks
        for peak in self.peak_list:
            i = initial_i
            for parameter_field in fields(peak):
                parameter = getattr(peak, parameter_field.name)
                if isinstance(parameter, Parameter):
                    if not parameter.constant and parameter.shared:
                        parameter.value = par_list[i]
                        i += 1
        return par_list[i:]

    def set_final_fit_variables(self, par_list, lower_list, upper_list):
        """
        Similar to set_fit_variables but now sets the lower and upper errors on all the fit parameters as well.
        Used to store all the final fit parameters before the values and errors are output to file.

        :param List[float] par_list: list of all fit parameters
        :param List[float] lower_list: list of all fit parameter lower errors
        :param List[float] upper_list: list of all fit parameter upper errors
        :return: remaining fit information after setting all values in the pair of peaks
        """
        i = 0
        for peak in self.peak_list:
            for parameter_field in fields(peak):
                parameter = getattr(peak, parameter_field.name)
                if isinstance(parameter, Parameter):
                    if not parameter.constant and not parameter.shared:
                        parameter.value = par_list[i]
                        parameter.lower = lower_list[i]
                        parameter.upper = upper_list[i]
                        i += 1

        initial_i = i
        # add variables which are shared by both peaks
        for peak in self.peak_list:
            i = initial_i
            for parameter_field in fields(peak):
                parameter = getattr(peak, parameter_field.name)
                if isinstance(parameter, Parameter):
                    if not parameter.constant and parameter.shared:
                        parameter.value = par_list[i]
                        parameter.lower = lower_list[i]
                        parameter.upper = upper_list[i]
                        i += 1
        return par_list[i:], lower_list[i:], upper_list[i:]

    def output_data_peak(self):
        """
        returns all fit data - values, upper and lower errors in a list of lists.

        :return: list of lists of all output data for each peak in the pair of peaks
        """
        output_data = []
        for peak in self.peak_list:
            output_data.append(peak.output_data())
        return output_data

@dataclass
class Fit_Info:
    """ class for storing all information for a particular fit.
    Used to generate the list of parameters which are varied in the fit.
    """
    # parameters that are set when the class is initialised
    peak_pair_list: List[Pair_Peaks]  # list of peak pairs involved in fit (if only fitting one pair per window then this list has one item)
    background: Parameter  # background value for fit
    l1_visibility: float
    l2_visibility: float
    l3_visibility: float

    # parameters that can be left without values when the class is initialised
    fit_pars: List[float] = field(default_factory=list)  # list of all fit parameter values to pass to emcee
    lower_prior: List[float] = field(default_factory=list)  # list of all lower bounds on par values - used to generate prior prob distribution
    upper_prior: List[float] = field(default_factory=list)  # list of all upper bounds on par values - used to generate prior prob distribution
    par_labels: List[str] = field(default_factory=list)  # list of all parameter labels

    def __init__(self, peak_pair_list, background, visibilities):
        self.peak_pair_list = peak_pair_list
        self.background = background
        self.l1_visibility = visibilities[0]
        self.l2_visibility = visibilities[1]
        self.l3_visibility = visibilities[2]

        self.get_all_fit_variables()

    def get_all_fit_variables(self):
        """ generates the list of the values of the parameters for all parameters that need to be fitted.
        This list is passed to the emcee code.
        Generates list of parameter names (in order to generate corner plots) stored in par_labels
        Generates lists of upper and lower bounds of the fit values, these are used to generate the uniform priors for
        the emcee walkers as well the initial parameter values for the walkers.
        :return list of fit parameters """
        self.fit_pars = []
        self.lower_prior = []
        self.upper_prior = []
        self.par_labels = []
        for peak_pair in self.peak_pair_list:
            value_list, lower_list, upper_list, label_list = peak_pair.get_fit_variables()
            self.fit_pars += value_list
            self.lower_prior += lower_list
            self.upper_prior += upper_list
            self.par_labels += label_list

        if not self.background.constant:
            self.fit_pars.append(self.background.value)
            self.lower_prior.append(self.background.lower)
            self.upper_prior.append(self.background.upper)
            self.par_labels += ["background"]
        return self.fit_pars

    def set_all_fit_variables(self, par_list):
        """
        given new list of parameter values (from the emcee walkers) sets all the appropriate values in
        the peaks in peak_pair_list
        :param List[float] par_list: list of all fit parameter values
        :return:
        """
        for peak_pair in self.peak_pair_list:
            par_list = peak_pair.set_fit_variables(par_list)
        if not self.background.constant:
            self.background.value = par_list[0]

    def set_all_final_fit_variables(self, par_list, lower_list, upper_list):
        """
        Used after emcee code has run and the final values and lower and upper errors have been found.
        This populates the peaks in peak_pair_list with all the final values in order to make it easy to
        generate the output final with all output values in.

        :param List[float] par_list: list of final fit parameter values
        :param List[float] lower_list: list of all lower errors on fit parameter values
        :param List[float] upper_list: list of all upper errors on fit parameter values
        :return:
        """
        for peak_pair in self.peak_pair_list:
            par_list, lower_list, upper_list = peak_pair.set_final_fit_variables(par_list, lower_list, upper_list)
        if not self.background.constant:
            self.background.value = par_list[0]
            self.background.lower = lower_list[0]
            self.background.upper = upper_list[0]

    def output(self):
        """
        Generate a list of the parameter values and errors for each peak in peak_pair_list
        which are written to a .txt file.
        :return: list of parameter values and errors
        """
        lines = []
        for peak_pair in self.peak_pair_list:
            output_lines = peak_pair.output_data_peak()
            for line in output_lines:
                if self.background.constant:
                    line.append(str(self.background.value))
                    line.append(str(0.0))
                    line.append(str(0.0))
                else:
                    line.append(str(self.background.value))
                    line.append(str(self.background.lower))
                    line.append(str(self.background.upper))
                lines.append(line)
        return lines

    def freq_range(self):
        """
        :return: the frequency in the priors for each peak in peak_pair_list
        """
        freqs = []
        for peak_pair in self.peak_pair_list:
            for peak in peak_pair.peak_list:
                freqs.append(peak.freq.value)
        freqs.sort()
        return freqs
