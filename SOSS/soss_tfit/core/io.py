#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-12

@author: cook
"""
import numpy as np
from typing import List, Dict

# =============================================================================
# Define variables
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def stack_multi_spec(multi_spec, use_orders: List[int],
                     quantities: List[str]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Convert a jwst multi_spec into a dictionary

    all_spec[order][quantity]

    where order is [1, 2, 3] etc
    quantity is ['WAVELENGTH', 'FLUX', 'FLUX_ERROR', 'TIME', 'BIN_LIMITS']

    :param multi_spec:
    :param use_orders:
    :param quantities:
    :return:
    """
    # define a container for the stack
    all_spec = dict()
    # -------------------------------------------------------------------------
    # loop around orders
    for sp_ord in use_orders:
        # set up dictionary per order for each quantity
        all_spec[sp_ord] = dict()
        # loop around quantities
        for quantity in quantities:
            all_spec[sp_ord][quantity] = []
    # -------------------------------------------------------------------------
    # loop around orders and get data
    for spec in multi_spec.spec:
        # get the spectral order number
        sp_ord = spec.spectral_order
        # skip orders not wanted
        if sp_ord not in use_orders:
            continue
        # load values for each quantity
        for quantity in quantities:
            # only add quantities
            if quantity in spec.spec_table.columns.names:
                all_spec[sp_ord][quantity].append(spec.spec_table[quantity])
    # -------------------------------------------------------------------------
    # convert list to array
    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])
    # -------------------------------------------------------------------------
    # return the all spectrum dictionary
    return all_spec


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================