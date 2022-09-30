#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2022-09-29 at 14:55

@author: cook
"""
from astraeus import xarrayIO
from astropy.table import Table
from astropy.visualization import LinearStretch
from astropy.visualization import MinMaxInterval, ZScaleInterval
from astropy.visualization import ImageNormalize
import glob
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Tuple
import warnings

# =============================================================================
# Define variables
# =============================================================================
# path to directories contain "EA", "MR", "LPC"
PATH = '/scratch3/jwst-soss/data/jwst-data/oliva_t1'
# file path within PATH (with wildcards) to select the visits we want
FILES = '*/V1_*.h5'
# the SED file within the "EA" directory to apply to EA residual spectra data
SED_FILE = 'V1_sed_trappist-1b_ord1.csv'
# The comparison operation (either "-" or "/")
COMPARISON = '/'
# A user readable name for the comparison operator ("diff" or "ratio")
COMP_NAME = 'ratio'
# the spline keyword arguments
splinekwargs = dict(ext=1, k=3)
# storage for the data so we don't open it more times than we need
DATA_STORAGE = dict()


# =============================================================================
# Define classes
# =============================================================================
class DifferencePlot:
    """
    This class makes a clickable plot
    """

    def __init__(self, diffimage: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                 vector1: np.ndarray, vector2: np.ndarray,
                 name1: str, name2: str):
        """
        Construct the difference plot class

        :param diffimage: numpy array the 2D comparison image
                          (be if diff or ratio)
        :param x1: x-axis data for 1D view plot for name1 (i.e. wavelengths)
        :param x2: x-axis data for 1D view plot for name2 (i.e. wavelengths)
        :param vector1: y-axis data for 1D view plot for name1 (i.e. flux)
        :param vector2: y-axis data for 1D view plot for name2 (i.e. flux)
        :param name1: string name for legend (data set 1)
        :param name2: string name for legend (data set 2)
        """
        # store the comparison array (2D)
        self.diffimage = diffimage
        # work out the gradients in both directions
        with warnings.catch_warnings(record=True) as _:
            self.dx_diffimage = np.gradient(self.diffimage, axis=1)
            self.dy_diffimage = np.gradient(self.diffimage, axis=0)
        # store the x-axis data
        self.x1 = x1
        self.x2 = x2
        # store the y-axis data
        self.vector1 = vector1
        self.vector2 = vector2
        # store the names
        self.name1 = name1
        self.name2 = name2
        # graph properties blank at start
        self.fig = None
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.frame4 = None
        self.frame5 = None
        # a counter for when graph is clicked
        self.counter = 0

    def plot(self):
        """
        Plot the difference plot
        """
        plt.close()
        self.fig = plt.figure()
        self.frame1 = plt.subplot2grid((3, 3), loc=(0, 0))
        self.frame2 = plt.subplot2grid((3, 3), loc=(0, 1))
        self.frame3 = plt.subplot2grid((3, 3), loc=(0, 2))
        self.frame4 = plt.subplot2grid((3, 3), loc=(1, 0), colspan=3)
        self.frame5 = plt.subplot2grid((3, 3), loc=(2, 0), colspan=3)
        # define the extent of the images
        extent = [np.nanmin(self.x1), np.nanmax(self.x1), 0, len(self.diffimage)]
        # ---------------------------------------------------------------------
        # Diff image
        # ---------------------------------------------------------------------
        norm1 = ImageNormalize(self.diffimage, interval=MinMaxInterval(),
                               stretch=LinearStretch())
        im1 = self.frame1.imshow(self.diffimage, norm=norm1,
                                 aspect='auto', origin='lower', extent=extent)
        self.frame1.set(title=f'{COMP_NAME} image', xlabel='wavelength',
                        ylabel='integration')
        divider1 = make_axes_locatable(self.frame1)
        cax1 = divider1.append_axes('bottom', size='5%', pad=0.5)
        self.fig.colorbar(im1, cax=cax1, orientation='horizontal')
        # ---------------------------------------------------------------------
        # Diff image
        # ---------------------------------------------------------------------
        norm2 = ImageNormalize(self.dx_diffimage, interval=ZScaleInterval(),
                               stretch=LinearStretch())
        im2 = self.frame2.imshow(self.dx_diffimage, norm=norm2,
                                 aspect='auto', origin='lower', extent=extent)
        self.frame2.set(title=f'dx {COMP_NAME} image', xlabel='wavelength',
                        ylabel='integration')
        divider2 = make_axes_locatable(self.frame2)
        cax2 = divider2.append_axes('bottom', size='5%', pad=0.5)
        self.fig.colorbar(im2, cax=cax2, orientation='horizontal')
        # ---------------------------------------------------------------------
        # Diff image
        # ---------------------------------------------------------------------
        norm3 = ImageNormalize(self.dy_diffimage, interval=ZScaleInterval(),
                               stretch=LinearStretch())
        im3 = self.frame3.imshow(self.dy_diffimage, norm=norm3,
                                 aspect='auto', origin='lower', extent=extent)
        self.frame3.set(title=f'dy {COMP_NAME} image', xlabel='wavelength',
                        ylabel='integration')

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.plot_graphs()
        divider3 = make_axes_locatable(self.frame3)
        cax3 = divider3.append_axes('bottom', size='5%', pad=0.5)
        self.fig.colorbar(im3, cax=cax3, orientation='horizontal')
        # ---------------------------------------------------------------------
        plt.suptitle(f'{self.name1} vs {self.name2}  comparison={COMP_NAME}')
        plt.show()

    def plot_graphs(self):
        """
        Plot the changes that need to update when clicked

        we have to replot the axis completely but this is also used in
        self.plot to do it the first time
        """
        self.frame4.plot(self.x1, self.vector1[self.counter], label=self.name1)
        self.frame4.plot(self.x2, self.vector2[self.counter], label=self.name2)

        self.frame4.legend(loc=0, title=f'Integration {self.counter}')
        self.frame4.set(xlabel='wavelength', ylabel='flux')

        self.frame5.plot(self.x1, self.diffimage[self.counter],
                         label=f'{COMP_NAME}')
        self.frame5.legend(title=f'Integration {self.counter}', loc=0)
        self.frame5.set(xlabel='wavelength', ylabel='flux')

    def onclick(self, event):
        """
        The on click event handled by matplotlib, must have the event argument

        :param event: event passed and handled by maplotlib
        """
        if event.inaxes is not self.frame4 and event.inaxes is not self.frame5:
            if event.button == 1:
                try:
                    self.counter = int(event.ydata)
                except:
                    return
        # clear current data
        self.frame4.clear()
        self.frame5.clear()
        # redraw
        self.plot_graphs()
        plt.draw()


# =============================================================================
# Define functions
# =============================================================================
def read_file(filename: str, kind: str
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Basic read file that handles doing weird stuff for certain peoples
    reduction

    :param filename: the absolute path to the file to read
    :param kind: str, either "EA", "MR" or "LPC", controls how files are read

    :returns: tuple, 1. the wave vector, 2. the time vector, 3. the flux array
              4. the flux error array
    """
    global DATA_STORAGE
    if filename in DATA_STORAGE:
        # read from data storage
        spec = DATA_STORAGE[filename]
    else:
        # read the .h5 data using xarrayIO
        spec = xarrayIO.readXR(filename, verbose=False)
        # add to data storage (avoids opening it again)
        DATA_STORAGE[filename] = spec

    wave = np.array(spec['wave_1d'].data)
    time = np.array(spec['time'].data)
    flux = np.array(spec['optspec'].data)
    flux_err = np.array(spec['opterr'].data)

    if 'EA' in kind:
        # need to get the SED for EA residual spectrum
        path = os.path.dirname(filename)
        sed_table = Table.read(os.path.join(path, SED_FILE))
        # its the wrong way round for splining
        sed_wave = sed_table['wavelength'][::-1]
        sed_rawflux = sed_table['raw flux'][::-1]
        # get rid of NaN values
        sed_mask = np.isfinite(sed_wave) & np.isfinite(sed_rawflux)
        # make a spline
        sed_spline = InterpolatedUnivariateSpline(sed_wave[sed_mask],
                                                  sed_rawflux[sed_mask],
                                                  **splinekwargs)
        # get the SED flux to scale the residual spectrum by
        sed_flux = sed_spline(wave)
        # loop around each row and multiple out by the spectrum
        for _row in range(len(flux)):
            flux[_row] *= sed_flux

    return wave, time, flux, flux_err


def comparison(vector1: np.ndarray, vector2: np.ndarray,
               kind: str = '-') -> np.ndarray:
    """
    Comparison operator, depending on "kind" does a diff or a ratio of vectors

    :param vector1: the first 1D vector
    :param vector2: the second 1D vector
    :param kind: str, comparison operator, either '-' or '/'

    :returns: the comparison vector (same shape as vector 1 and 2)
    """
    if kind == '-':
        return vector1 - vector2
    elif kind == '/':
        with warnings.catch_warnings(record=True) as _:
            return vector1 / vector2


def innerlimit(vector1: np.ndarray, vector2: np.ndarray) -> Tuple[float, float]:
    """
    Works out the inner limit i.e. the inner bounds of down vectors

    :param vector1: the first 1D vector
    :param vector2: the second 1D vector

    :returns: tuple, 1. inner minimum of the two vectors, 2. inner maximum of
              of the two vectors
    """
    min1 = np.nanmin(vector1)
    max1 = np.nanmax(vector1)
    min2 = np.nanmin(vector2)
    max2 = np.nanmax(vector2)

    return np.max([min1, min2]), np.min([max1, max2])


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # get all files using glob
    files = glob.glob(os.path.join(PATH, FILES))
    # make all combinations of the files
    combinations = list(itertools.product(files, files))

    # combinations = [[PATH + '/LPC/V1_Vis1Order1NewTRAPPIST_1_b_Table_Save_bjdtdb.h5',
    #                  PATH + '/MR/V1_jw02589001001_04101_00001-segall_nis_extract1dstep_box_ord1_bjdtdb.h5']]

    # store combinations used
    used = []
    # loop around all combinations
    for it, (file1, file2) in enumerate(combinations):
        # ---------------------------------------------------------------------
        # get names using directory names)
        compname1 = os.path.basename(os.path.dirname(file1))
        compname2 = os.path.basename(os.path.dirname(file2))
        # ---------------------------------------------------------------------
        # print the current progress
        pargs = [it + 1, len(combinations), compname1, COMP_NAME, compname2]
        print('Processing {0} of {1} ({2} {3} {4})'.format(*pargs))
        # ---------------------------------------------------------------------
        # deal with A-A
        if file1 == file2:
            print('\tSkipping combination')
            continue
        # deal with B-A when we already have A-B
        if (file2, file1) in used:
            print('\tSkipping combination')
            continue
        # # add A-B to used list
        # used.append((file1, file2))

        # read data 1
        wave1, time1, flux1, flux_err1 = read_file(file1, kind=compname1)
        # read data 2
        wave2, time2, flux2, flux_err2 = read_file(file2, kind=compname2)
        # ---------------------------------------------------------------------
        # create arrays to hold our interpolated + masked fluxes
        flux1a_arr = np.full_like(flux1, np.nan)
        flux2a_arr = np.full_like(flux1, np.nan)
        # get the lengths in each direction
        length_time = time1.shape[0]
        length_wave = wave1.shape[0]

        # ---------------------------------------------------------------------
        # push wave2 onto wave1
        # ---------------------------------------------------------------------
        for row in range(length_time):
            # get inner bounds
            inner_min, inner_max = innerlimit(wave1, wave2)
            # work out inner limit for data 1
            mask1 = (wave1 > inner_min) & (wave1 < inner_max)
            # mask out NaN values
            mask1 &= np.isfinite(flux1[row])
            # work out inner limit for data 2
            mask2 = (wave2 > inner_min) & (wave2 < inner_max)
            mask2 &= np.isfinite(flux2[row])
            # -----------------------------------------------------------------
            # create spline for data 2
            spline = InterpolatedUnivariateSpline(wave2[mask2],
                                                  flux2[row][mask2],
                                                  **splinekwargs)
            # -----------------------------------------------------------------
            # apply inner limit and spline onto data 1's wave grid
            flux1a = flux1[row][mask1]
            flux2a = spline(wave1[mask1])
            # push flux into array
            flux1a_arr[row][mask1] = flux1a
            flux2a_arr[row][mask1] = flux2a

        # ---------------------------------------------------------------------
        # comparison
        # ---------------------------------------------------------------------
        comp_arr = np.full_like(flux1, np.nan)

        for row in range(length_time):
            comp_arr[row] = comparison(flux1a_arr[row], flux2a_arr[row],
                                       COMPARISON)

        # ---------------------------------------------------------------------
        # plot
        # ---------------------------------------------------------------------
        # make difference plot class
        dplot = DifferencePlot(comp_arr, wave1, wave2, flux1, flux2,
                               compname1, compname2)
        # run the plot
        dplot.plot()

        # pause
        _ = input('\nCtrl+C to stop. Enter to continue\n')

# =============================================================================
# End of code
# =============================================================================
