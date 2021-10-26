import numbers
import numpy as np
from numpy.lib.arraysetops import isin

def check_random_state(seed):
    '''
    Turn seed into a np.random.RandomState instance

    Parameters:
        seed -- Return randomstate instance, None | int | instance of Randomstate
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)