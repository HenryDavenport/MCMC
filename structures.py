from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class Parameter:
    value: float
    lower: float
    upper: float
    constant: bool
    shared: bool

@dataclass
class Parameter_Fit_Info:
    constant: bool
    shared: bool

@dataclass
class Parameters_Fit_Info:
    freq: Parameter_Fit_Info
    splitting: Parameter_Fit_Info
    ln_linewidth: Parameter_Fit_Info
    ln_power: Parameter_Fit_Info
    asymmetry: Parameter_Fit_Info

@dataclass
class Peak_Structure:
    l: int
    n: int
    freq: Parameter
    splitting: Parameter
    ln_linewidth: Parameter
    ln_power: Parameter
    asymmetry: Parameter
    m_scale: float

    def __init__(self, peak_dict, run_info):
        self.l = int(peak_dict["l"])
        self.n = int(peak_dict["n"])
        freq = Parameter(peak_dict["freq"], peak_dict["freq lower error"],
                         peak_dict["freq upper error"], False, False)
        self.freq = freq

        if peak_dict["splitting"] == 0:
            splitting = Parameter(peak_dict["splitting"], peak_dict["splitting lower error"],
                                  peak_dict["splitting upper error"], True, False)
            self.splitting = splitting
        else:
            splitting = Parameter(peak_dict["splitting"], peak_dict["splitting lower error"],
                                  peak_dict["splitting upper error"], False, False)
            self.splitting = splitting

        ln_linewidth = Parameter(peak_dict["ln(width)"], peak_dict["ln(width) lower error"],
                                 peak_dict["ln(width) upper error"], False, False)
        self.ln_linewidth = ln_linewidth

        ln_power = Parameter(peak_dict["ln(power)"], peak_dict["ln(power) lower error"],
                              peak_dict["ln(power) upper error"], False, False)
        self.ln_power = ln_power

        asymmetry = Parameter(peak_dict["asymmetry"], peak_dict["asymmetry lower error"],
                              peak_dict["asymmetry upper error"], False, True)
        self.asymmetry = asymmetry

        if self.l == 2:
            self.m_scale = run_info.l2_m_scale
        else:
            self.m_scale = run_info.l3_m_scale

    def output_data(self):
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
    peak_list: List[Peak_Structure]

    def get_fit_variables(self):
        par_list = []
        lower_list = []
        upper_list = []
        par_labels = []
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
        i = 0
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
        output_data = []
        for peak in self.peak_list:
            output_data.append(peak.output_data())
        return output_data

@dataclass
class Fit_Info:
    peak_pair_list: List[Pair_Peaks]
    background: Parameter
    fit_pars: List[float] = field(default_factory=list)
    lower_prior: List[float] = field(default_factory=list)
    upper_prior: List[float] = field(default_factory=list)
    par_labels: List[str] = field(default_factory=list)

    l1_visibility: float = 1.505
    l2_visibility:  float = 0.62
    l3_visibility:  float = 0.075


    def get_all_fit_variables(self):
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
        for peak_pair in self.peak_pair_list:
            par_list = peak_pair.set_fit_variables(par_list)
        if not self.background.constant:
            self.background.value = par_list[0]

    def set_all_final_fit_variables(self, par_list, lower_list, upper_list):
        for peak_pair in self.peak_pair_list:
            par_list, lower_list, upper_list = peak_pair.set_final_fit_variables(par_list, lower_list, upper_list)
        if not self.background.constant:
            print(par_list)
            print(lower_list)
            print(upper_list)
            self.background.value = par_list[0]
            self.background.lower = lower_list[0]
            self.background.upper = upper_list[0]

    def output(self):
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
        freqs = []
        for peak_pair in self.peak_pair_list:
            for peak in peak_pair.peak_list:
                freqs.append(peak.freq.value)
        return freqs


@dataclass
class Run_Info:
    number_of_processes = 3
    cadence = 60
    burnin = 1000
    no_samples = 10000
    nwalkers = 20

    # the ratios of the heights of the m split components of the l=2 and l=3 modes
    l2_m_scale = 0.634
    l3_m_scale = 0.400
    # the mode visibilities
    l1_visibility = 1.505
    l2_visibility = 0.62
    l3_visibility = 0.075
    window_width = 25
    lower_range = 2000
    upper_range = 4000

    save_middle_peak_parameters_only = False
    # the guess of the background
    background: Parameter
    # set to display the initial positions of all the walkers each peak run.
    check_priors = False