import numpy as np

def walker_pos(nwalkers, lower_priors, upper_priors):
    walker_positions = []
    for i in range(nwalkers):
        single_walker = []
        for lower, upper in zip(lower_priors, upper_priors):
            single_walker.append(np.random.uniform(low=lower, high=upper))
        walker_positions.append(single_walker)
    return walker_positions

