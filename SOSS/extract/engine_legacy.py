#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports.
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

# Astronoomy imports.
from astropy.io import fits

# Plotting.
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

###############################################
# Hack to get the path of module. To be changed.
from os.path import abspath, dirname


def get_module_path(file):

    dir_path = abspath(file)
    dir_path = dirname(dir_path) + '/'

    return dir_path
###############################################


# Default file parameters
DEF_PATH = get_module_path(__file__) + "Ref_files/"

FILE_SOSS = "NIRISS_Throughput_STScI.fits"
DEF_FILE_FRAME = "spectral_kernel_matrix/spectral_kernel_matrix_os_{}_width_{}pixels.fits"


class ThroughputSOSS(interp1d):
    """
    Callable Throughput of SOSS mode for a given order.
    Function oof wavelength in microns.
    """
    path = DEF_PATH
    filename = FILE_SOSS

    def __init__(self, order=1):
        """
        Parameter:
        order: int
            which order do you want? Default is the first order (1)
        """
        # Open file
        hdu = fits.open(self.path + self.filename)

        # Get transmission
        key = 'SOSS_order{}'.format(order)
        tr = hdu[1].data[key].squeeze()

        # Get wavelength
        wv = hdu[1].data['LAMBDA'].squeeze()
        # nm to microns
        wv /= 1000.

        # Interpolate
        super().__init__(wv, tr, kind='cubic',
                         fill_value=0, bounds_error=False)


class WebbKer:
    """
    Class to load Webb convolution kernel. Once instanciated,
    the object act as a callable (function)
    of wavelength and center wavelength.
    It is also possible to have a look at the kernels with
    the `show` method.
    """
    path = DEF_PATH
    file_frame = DEF_FILE_FRAME

    def __init__(self, wave_map, n_os=10, n_pix=21,
                 bounds_error=False, fill_value="extrapolate"):
        """
        Parameters
        ----------
        wave_map: 2d array
            Wavelength map of the detector. Since WebbPSF returns
            kernels in the pixel space, we need a wv_map to convert
            to wavelength space.
        n_os: int, optional
            oversampling of the kernel. Default is 10
        n_pix: int, optional
            Length of the kernel in pixels. Default is 21.
        bounds_error: bool, optional
            If True, raise an error when trying to call the
            function out of the interpolation range. If False,
            the values will be extrapolated. Default is False
        fill_value: str, opotional
            How to extrapolate when needed. Default is "extrapolate"
            and it is the oonly option so far. There is the
            possibility to implement other ways like in
            scipy.interp1d, but it is not done yet.
        """

        # Mask where wv_map is equal to 0
        wave_map = np.ma.array(wave_map, mask=(wave_map == 0))

        # Force wv_map to have the red wavelengths
        # at the end of the detector
        if np.diff(wave_map, axis=-1).mean() < 0:
            wave_map = np.flip(wave_map, axis=-1)

        # Number of columns
        ncols = wave_map.shape[-1]

        # Create filename  # TODO change to CRDS/manual input.
        file = self.file_frame.format(n_os, n_pix)

        # Read file
        hdu = fits.open(self.path + file)
        header = hdu[0].header
        kernel, wave_kernel = hdu[0].data

        # Where is the blue and red end of the kernel
        i_blue, i_red = header["BLUINDEX"], header["REDINDEX"]

        # Flip `ker` to put the red part of the kernel at the end.
        if i_blue > i_red:
            kernel = np.flip(kernel, axis=0)

        # Create oversampled pixel position array  # TODO easier to read form?
        pixels = np.arange(-(n_pix//2), n_pix//2 + 1/n_os, 1/n_os)

        # `wave_kernel` has only the value of the central wavelength
        # of the kernel at each points because it's a function
        # of the pixels (so depends on wv solution).
        wave_center = wave_kernel[0, :]

        # Use the wavelength solution to create a mapping.
        # First find the kernels that fall on the detector.
        wave_min = np.amin(wave_map[wave_map > 0])
        wave_max = np.amax(wave_map[wave_map > 0])
        i_min = np.searchsorted(wave_center, wave_min)  # TODO searchsorted has offsets?
        i_max = np.searchsorted(wave_center, wave_max) - 1

        # SAVE FOR LATER ###########
        # Use the next kernels at each extremities to define the
        # boundaries of the interpolation to use in the class
        # RectBivariateSpline (at the end)
        # bbox = [min pixel, max pixel, min wv_center, max wv_center]
        bbox = [None, None,
                wave_center[np.maximum(i_min-1, 0)],
                wave_center[np.minimum(i_max+1, len(wave_center)-1)]]
        #######################

        # Keep only kernels that fall on the detector.
        kernel, wave_kernel = kernel[:, i_min:i_max+1], wave_kernel[:, i_min:i_max+1]
        wave_center = np.array(wave_kernel[0, :])

        # Then find the pixel closest to each kernel center
        # and use the surrounding pixels (columns)
        # to get the wavelength. At the boundaries,
        # wavelenght might not be defined or falls out of
        # the detector, so fit a 1-order polynomial to
        # extrapolate. The polynomial is also used to interpolate
        # for oversampling.
        i_surround = np.arange(-(n_pix//2), n_pix//2 + 1)
        poly = []
        for i_cen, wv_c in enumerate(wave_center):
            wv = np.ma.masked_all(i_surround.shape)

            # Closest pixel wv
            i_row, i_col = np.unravel_index(
                np.argmin(np.abs(wave_map - wv_c)), wave_map.shape
            )
            # Update wavelength center value
            # (take the nearest pixel center value)
            wave_center[i_cen] = wave_map[i_row, i_col]

            # Surrounding columns
            index = i_col + i_surround

            # Make sure it's on the detector
            i_good = (index >= 0) & (index < ncols)

            # Assign wv values
            wv[i_good] = wave_map[i_row, index[i_good]]

            # Fit n=1 polynomial
            poly_i = np.polyfit(i_surround[~wv.mask], wv[~wv.mask], 1)

            # Project on os pixel grid
            wave_kernel[:, i_cen] = np.poly1d(poly_i)(pixels)

            # Save coeffs
            poly.append(poly_i)

        # Save attributes
        self.n_pix = n_pix
        self.n_os = n_os
        self.wave_kernel = wave_kernel
        self.kernel = kernel
        self.pixels = pixels
        self.wave_center = wave_center
        self.poly = np.array(poly)
        self.fill_value = fill_value
        self.bounds_error = bounds_error

        # 2d Interpolate
        self.f_ker = RectBivariateSpline(pixels, wave_center, kernel, bbox=bbox)

    def __call__(self, wave, wave_c):
        """
        Returns the kernel value, given the wavelength
        and the kernel center wavelength.

        Parameters
        ----------
        wave: 1d array
            wavelenght where the kernel is projected.
        wave_c: 1d array (same shape as `wv`)
            center wavelength of the kernel
        """

        wave_center = self.wave_center
        poly = self.poly
        fill_value = self.fill_value
        bounds_error = self.bounds_error
        n_wv_c = len(wave_center)
        f_ker = self.f_ker
        n_pix = self.n_pix

        # #################################
        # First, convert wv value in pixels
        # using a linear interpolation
        # #################################

        # Find corresponding interval
        i_wv_c = np.searchsorted(wave_center, wave_c) - 1

        # Deal with values out of bounds
        if bounds_error:
            message = "Value of wv center out of interpolation range"
            raise ValueError(message)
        elif fill_value == "extrapolate":
            i_wv_c[i_wv_c < 0] = 0
            i_wv_c[i_wv_c >= (n_wv_c - 1)] = n_wv_c - 2
        else:
            message = "`fill_value`={} is not an valid option."
            raise ValueError(message.format(fill_value))

        # Compute coefficients that interpolate along wv_centers
        d_wv_c = wave_center[i_wv_c + 1] - wave_center[i_wv_c]
        a_c = (wave_center[i_wv_c + 1] - wave_c) / d_wv_c
        b_c = (wave_c - wave_center[i_wv_c]) / d_wv_c

        # Compute a_pix and b_pix from the equation:
        # pix = a_pix * lambda + b_pix
        a_pix = 1 / (a_c * poly[i_wv_c, 0] + b_c * poly[i_wv_c+1, 0])
        b_pix = -(a_c * poly[i_wv_c, 1] + b_c * poly[i_wv_c+1, 1])
        b_pix /= (a_c * poly[i_wv_c, 0] + b_c * poly[i_wv_c+1, 0])

        # Compute pixel values
        pix = a_pix * wave + b_pix

        # ######################################
        # Second, compute kernel value on the
        # interpolation grid (pixel x wv_center)
        # ######################################

        out = f_ker(pix, wave_c, grid=False)
        # Make sure it's not negative
        out[out < 0] = 0

        # and put values out of pixel range
        # to zero
        out[pix > n_pix//2] = 0
        out[pix < -(n_pix//2)] = 0

        return out

    def show(self):
        """
        Plot kernels.
        The first figure is a 2d image of the kernels.
        The second figure is a 1d image of the kernels
        in the wavelength space.
        """

        # 2D figure of the kernels
        fig1 = plt.figure()

        # Log plot, so clip values <= 0
        image = np.clip(self.kernel, np.min(self.kernel[self.kernel > 0]), np.inf)

        # plot
        plt.pcolormesh(self.wave_center, self.pixels,  image, norm=LogNorm())

        # Labels and others
        plt.colorbar(label="Kernel")
        plt.ylabel("Position relative to center [pixel]")
        plt.xlabel(r"Center wavelength [$\mu m$]")
        plt.tight_layout()

        # 1D figure of all kernels
        fig2 = plt.figure()
        plt.plot(self.wave_kernel, self.kernel)

        # Labels and others
        plt.ylabel("Kernel")
        plt.xlabel(r"Wavelength [$\mu m$]")
        plt.tight_layout()

        return fig1, fig2
