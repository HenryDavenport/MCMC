""" initialise_walker_positions.py
The walkers for the MCMC are given positions in parameter space which are selected from a uniform
distribution between the inputted lower and upper prior values for each parameter. """
import numpy as np


def walker_pos(nwalkers, lower_priors, upper_priors):
    """
    The initial walker positions are selected from a uniform distribution between the lower and upper priors.
    :param int nwalkers: number of walkers being used
    :param lower_priors: The list of the lower bound on values of each parameter
    :param upper_priors: The list of the upper bound on values of each parameter
    :return: The positions in parameter space of all the walkers in a list of lists.
    """
    walker_positions = []
    for i in range(nwalkers):
        single_walker = []
        for lower, upper in zip(lower_priors, upper_priors):
            # pick values from a uniform distribution between the lower and upper bounds
            single_walker.append(np.random.uniform(low=lower, high=upper))
        walker_positions.append(single_walker)
    return walker_positions

