
import sys
from numpy.lib.arraysetops import isin
import torch
import numpy as np

def get_factor(unit):
    fac_dict = {'b': 1, 'kb':1e-3, 'mb': 1e-6, 'gb': 1e-9}
    if not unit in fac_dict:
        raise ValueError(f'\'{unit}\' is not a valid storage unit, options are \'b\', \'kb\', \'mb\', \'gb\'.')
    return fac_dict[unit.lower()]

def get_mem_size(x, unit='gb'):
    ''' Get memory size
    Get size of an object in memory.

    Parameters
    ----------
    x : Object
        The object to get the size of.
    unit : str (optional, default='gb')
        The unit to report the size in. Options are 'b', 'kb', 'mb', 'gb'.
    
    Returns
    -------
    float
        The size of the object in the specified unit.
    '''
    factor = get_factor(unit)
    if isinstance(x, torch.Tensor) or \
        isinstance(x, torch.nn.parameter.Parameter):
        return sys.getsizeof(x.storage())*factor
    elif isinstance(x, torch.nn.Module):
        return sum([get_mem_size(y, unit=unit) for y in x.parameters()])
    elif isinstance(x, np.ndarray) and x.dtype == object:
        return sum([get_mem_size(y, unit) for y in x.flatten()])
    elif isinstance(x, (tuple, list)):
        return sum([get_mem_size(y, unit) for y in x])
    else:
        return sys.getsizeof(x)*factor

def free_gpu_memory(i=0, unit='gb'):
    r = torch.cuda.memory_reserved(i)
    a = torch.cuda.memory_allocated(i)
    return (r-a)*get_factor(unit)