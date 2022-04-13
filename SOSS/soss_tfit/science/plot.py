#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-04-13

@author: cook
"""
import matplotlib.pyplot as plt

from soss_tfit.science import general

# =============================================================================
# Define variables
# =============================================================================
# get input data class
InputData = general.InputData


# =============================================================================
# Define functions
# =============================================================================
def plot_flux(data: InputData):
    # Show a plot of the data. Each colour is a different wavelength.
    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.set(xlabel='Time (days)', ylabel='Flux')
    # get photometric data dictionary
    phot = data.phot
    # loop around photometric bandpasses
    for i_phot in range(data.n_phot):
        frame.plot(phot['TIME'][i_phot], phot['FLUX'][i_phot])
    # show and close
    plt.show()
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
