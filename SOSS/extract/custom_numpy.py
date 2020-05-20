import numpy as np

def is_sorted(x, no_dup=True):
    
    if no_dup:
        return (np.diff(x) > 0).all()
    else:
        return (np.diff(x) >= 0).all()

def fill_list(x, fill_value=np.nan, **kwargs):
    '''
    Fill a list `x` (N, non-constant M)
    to make an array (N, M) with it 
    '''
    
    n1 = len(x)
    n2 = np.max([len(x_i) for x_i in x])
    
    out = np.ones((n1, n2), **kwargs) * fill_value
    for i, x_i in enumerate(x):
        out[i,:len(x_i)] = x_i
    
    return out

def first_change(cond, axis=None):
    '''
    Returns the position before the first change
    in array value, along axi`. If no change is found,
    an empty array will be returned.
    '''
    # Find first change
    cond = np.diff(cond, axis=axis)
    # Return position
    return np.where(cond)

def vrange(*args, return_where=False, dtype=None):

    if len(args)==1:
        starts = 0
    elif len(args)==2:
        starts = args[0]
    else:
        raise TypeError('vrange() takes at most 2 non-kw args')
    stops = args[-1]
    
    if return_where:
        l = (stops - starts).astype(int)
        ind1 = np.repeat(np.arange(len(l)), l)
        ind2 = _vrange(0, l)
        return _vrange(starts, stops, dtype), (ind1, ind2)
    else:
        return _vrange(starts, stops, dtype)
    
def _vrange(starts, stops, dtype=None):
    """
    Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    if dtype==int or dtype==None:
        stops = np.asarray(stops, dtype=dtype)
        l = (stops - starts).astype(int) # Lengths of each range.
        return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    else:
        return NotImplemented

def arange_2d(*args, dtype=None, return_mask=False):
    
    if len(args)==1:
        starts = 0
    elif len(args)==2:
        starts = args[0]
    else:
        raise TypeError('vrange() takes at most 2 non-kw args')
    stops = args[-1]
    
    l1 = len(stops)
    l2 = int((stops - starts).max())
    
    out = np.ones((l1,l2), dtype=dtype)
    
    values, ind = vrange(starts, stops,
                         dtype=dtype, return_where=True)
    out[ind[0], ind[1]] = values
    
    if return_mask:
        mask = np.ones((l1,l2), dtype=bool)
        mask[ind[0], ind[1]] = False
        return out, mask
    else:
        return out