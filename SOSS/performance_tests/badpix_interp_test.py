"""
# badpix_interp_test.py

# This script is used to test the performance of the bad pixel interpolation.
# It uses the valid pixels of a time series and randomly selects a fraction of
# them to be masked. The masked pixels are then interpolated using the
# interpolation function that is tested.

Created on 2024-05-02

@author: Antoine Darveau-Bernier
"""

# # Import os to make the output directory
# from pathlib import Path
import numpy as np

from astropy.convolution import convolve, TrapezoidDisk2DKernel

# from jwst import datamodels
# from jwst.datamodels import dqflags
# from astropy.nddata.bitmask import bitfield_to_boolean_mask

# # Only import the extraction step
# from jwst.extract_1d import Extract1dStep

# # Import utilities to analyse the outputs
# # from analysis_tools import 
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm #for better display of FITS images

# import jwst

# Import the random number generator
rng = np.random.default_rng()


# -----------------------------------------------------------------------------
# Functions taken from SupremeSPOON (Micheal Radical)
# Meant to identify remaining bad pixels (not picked up by the DQ flags)
# -----------------------------------------------------------------------------
def outlier_resistant_variance(data, axis=0):
    """Calculate the varaince of some data along the 0th axis in an outlier
    resistant manner.
    """

    var = (np.nanmedian(np.abs(data - np.nanmedian(data, axis=axis)), axis=axis) / 0.6745)**2
    return var


def get_interp_box(data, box_size, i, j, dimx):
    """Get median and standard deviation of a box centered on a specified
    pixel.

    Parameters
    ----------
    data : array-like[float]
        Data frame.
    box_size : int
        Size of box to consider.
    i : int
        X pixel.
    j : int
        Y pixel.
    dimx : int
        Size of x dimension.

    Returns
    -------
    box_properties : array-like
        Median and standard deviation of pixels in the box.
    """

    # Get the box limits.
    low_x = np.max([i - box_size, 0])
    up_x = np.min([i + box_size, dimx - 1])

    # Calculate median and std deviation of box - excluding central pixel.
    box = np.concatenate([data[j, low_x:i], data[j, (i+1):up_x]])
    median = np.nanmedian(box)
    stddev = np.sqrt(outlier_resistant_variance(box))

    # Pack into array.
    box_properties = np.array([median, stddev])

    return box_properties


def find_remaining_bad_pix(median_stack, space_thresh=15, box_size=5):
    """Flag remaining bad pixels in the median stack.

    Parameters
    ----------
    median_stack : array-like[float]
        Median stack of the data.
    space_thresh : float
        Threshold for flagging bad pixels.
    box_size : int
        Size of the box to consider.

    Returns
    -------
    missed_bad_pix : array-like[bool]
    """

    dimy, dimx = np.shape(median_stack)
    missed_bad_pix = np.zeros((dimy, dimx), dtype=bool)
    # Loop over whole deepstack and flag deviant pixels.
    for i in range(5, dimx - 5):
        print(f'{i+1} / {dimx - 8}', end='\r')
        for j in range(dimy - 5):
            # If the pixel is known to be hot, add it to list to interpolate.
            if np.isnan(median_stack[j, i]):
                pass
            # If not already flagged, double check that the pixel isn't
            # deviant in some other manner.
            else:
                box_size_i = box_size
                box_prop = get_interp_box(median_stack, box_size_i,
                                                i, j, dimx)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = get_interp_box(median_stack, box_size_i,
                                                    i, j, dimx)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant flag it.
                if np.abs(median_stack[j, i] - med) >= (space_thresh * std):
                    missed_bad_pix[j, i] = 1
    print()

    return missed_bad_pix


# -----------------------------------------------------------------------------
# Custom functions
# -----------------------------------------------------------------------------

def spread_bad_pixels(im, kernel=None, radius=1., frac_spread=0.7):
    """Spread bad pixels in an image.

    Parameters
    ----------
    bad_pix_im : array-like[bool]
        Image of bad pixels.
    kernel : array-like[float]
        Kernel to spread the bad pixels.
    radius : float
        Radius of the kernel.
    frac_spread : float
        Fraction of bad pixels to spread.

    Returns
    -------
    mask : array-like[bool]
        Mask of bad pixels.
    """

    if kernel is None:
        kernel = TrapezoidDisk2DKernel(radius)
    
    im_conv = convolve(im, kernel, boundary='fill', fill_value=0.)
    im_conv_rav = im_conv.ravel()
    mask_rav = (rng.random(im_conv_rav.size) * im_conv_rav.max()) < im_conv_rav * frac_spread
    mask = mask_rav.reshape(im_conv.shape)
    
    return mask


def create_test_data(data_valid_rows, mask_valid_rows, n_test, spread_frac=0.3, n_cosmic=0):
    """Create test data.

    Parameters
    ----------
    data_valid_rows : array-like[float]
        Valid data.
    mask_valid_rows : array-like[bool]
        Mask of valid data.
    n_test : int
        Number of pixels to test.
    spread_frac : float
        Fraction of bad pixels to spread.
    n_cosmic : int
        Number of cosmic rays.

    Returns
    -------
    test_data : array-like[float]
        Test data.
    pix_test_mask : array-like[bool]
        Mask of test data.
    """

    pix_id = np.arange(data_valid_rows.shape[1] * data_valid_rows.shape[2])

    pix_test_mask = np.zeros_like(mask_valid_rows)
    for i_frame in range(data_valid_rows.shape[0]):
        rnd_int = rng.integers(0, mask_valid_rows[i_frame].sum(), n_test)
        rnd_id = pix_id[mask_valid_rows[i_frame].ravel()][rnd_int]

        # Some bad pix are spreaded
        n_spreaded = int(spread_frac * n_test)

        rnd_id_2d = np.unravel_index(rnd_id[:n_spreaded], mask_valid_rows[i_frame].shape)
        temp_mask = np.zeros_like(pix_test_mask[i_frame])
        temp_mask[rnd_id_2d] = True
        temp_mask = spread_bad_pixels(temp_mask, radius=1.5)
        pix_test_mask[i_frame] |= temp_mask

        # Some bad pix are cosmic rays
        # but not yet implemented

        # Add the rest
        rnd_id_2d = np.unravel_index(rnd_id[n_spreaded+n_cosmic:], mask_valid_rows[i_frame].shape)
        temp_mask = np.zeros_like(pix_test_mask[i_frame])
        temp_mask[rnd_id_2d] = True
        pix_test_mask[i_frame] |= temp_mask
    
    # Make sure to remove real bad pixels after spreading
    pix_test_mask &= ~mask_valid_rows

    return pix_test_mask


def run_pix_interp_test(file_list, obsdir=None,)


# # Path and name to the data
# obsdir = Path.home() / Path('projects/def-dlafre/kmorel/WASP_80/Observations')

# # Create output directory
# output_dir = Path('atoca_results')
# output_dir.mkdir(exist_ok=True)

# !ls $obsdir

# file_list = [
#     'jw01201007001_04101_00001-seg001_nis_rateints.fits',
#     'jw01201007001_04101_00001-seg002_nis_rateints.fits',
#     'jw01201007001_04101_00001-seg003_nis_rateints.fits',
#     'jw01201007001_04101_00001-seg004_nis_rateints.fits',
#             ]

# mask, valid_data, valid_err, dq = [], [], [], []
# n = 0
# for filename in file_list:
#     datacube = datamodels.open(obsdir / filename)
#     # Get data and put nans for invalid pixels
#     mask.append(bitfield_to_boolean_mask(datacube.dq))
#     valid_data.append(np.where(mask[-1], np.nan, datacube.data))
#     valid_err.append(np.where(mask[-1], np.nan, datacube.err))
#     dq.append(datacube.dq)
#     print(valid_data[-1].shape)
#     wl_curve = np.nansum(valid_data[-1], axis=(1, 2))
#     plt.plot(np.arange(n, n + len(wl_curve)), wl_curve)
#     n += len(wl_curve)
    

# mask = np.concatenate(mask)
# valid_data = np.concatenate(valid_data)
# valid_err = np.concatenate(valid_err)
# dq = np.concatenate(dq)

# n_int, n_row, n_col = valid_data.shape

# median_stack = np.nanmedian(valid_data, axis=0) 

# def outlier_resistant_variance(data, axis=0):
#     """Calculate the varaince of some data along the 0th axis in an outlier
#     resistant manner.
#     """

#     var = (np.nanmedian(np.abs(data - np.nanmedian(data, axis=axis)), axis=axis) / 0.6745)**2
#     return var


# def get_interp_box(data, box_size, i, j, dimx):
#     """Get median and standard deviation of a box centered on a specified
#     pixel.

#     Parameters
#     ----------
#     data : array-like[float]
#         Data frame.
#     box_size : int
#         Size of box to consider.
#     i : int
#         X pixel.
#     j : int
#         Y pixel.
#     dimx : int
#         Size of x dimension.

#     Returns
#     -------
#     box_properties : array-like
#         Median and standard deviation of pixels in the box.
#     """

#     # Get the box limits.
#     low_x = np.max([i - box_size, 0])
#     up_x = np.min([i + box_size, dimx - 1])

#     # Calculate median and std deviation of box - excluding central pixel.
#     box = np.concatenate([data[j, low_x:i], data[j, (i+1):up_x]])
#     median = np.nanmedian(box)
#     stddev = np.sqrt(outlier_resistant_variance(box))

#     # Pack into array.
#     box_properties = np.array([median, stddev])

#     return box_properties

# box_size = 5
# space_thresh = 15

# dimy, dimx = np.shape(median_stack)
# missed_bad_pix = np.zeros((dimy, dimx), dtype=bool)
# # Loop over whole deepstack and flag deviant pixels.
# for i in range(5, dimx - 5):
#     print(f'{i+1} / {dimx - 8}', end='\r')
#     for j in range(dimy - 5):
#         # If the pixel is known to be hot, add it to list to interpolate.
#         if np.isnan(median_stack[j, i]):
#             pass
#         # If not already flagged, double check that the pixel isn't
#         # deviant in some other manner.
#         else:
#             box_size_i = box_size
#             box_prop = get_interp_box(median_stack, box_size_i,
#                                             i, j, dimx)
#             # Ensure that the median and std dev extracted are good.
#             # If not, increase the box size until they are.
#             while np.any(np.isnan(box_prop)):
#                 box_size_i += 1
#                 box_prop = get_interp_box(median_stack, box_size_i,
#                                                 i, j, dimx)
#             med, std = box_prop[0], box_prop[1]

#             # If central pixel is too deviant flag it.
#             if np.abs(median_stack[j, i] - med) >= (space_thresh * std):
#                 missed_bad_pix[j, i] = 1
# print()

# missed_bad_pix.sum(), dimx * dimy

# mask |= missed_bad_pix
# valid_data = np.where(missed_bad_pix, np.nan, valid_data)
# valid_err = np.where(missed_bad_pix, np.nan, valid_err)
# dq[:, missed_bad_pix] += 16  # Add the OUTLIER flag

# n_int, n_row, n_col = valid_data.shape

# median_stack = np.nanmedian(valid_data, axis=0)

# n_test = 50

# is_signal = median_stack > 30  # Use pixels with science signal

# cond = (~mask & is_signal)
# is_valid_row = [cond[:,i_row, :].any() for i_row in range(valid_data.shape[1])]

# data_valid_rows = valid_data[:, is_valid_row, :]
# err_valid_rows = valid_err[:, is_valid_row, :]
# mask_valid_rows = mask[:, is_valid_row, :]
# cond = cond[:, is_valid_row, :]

# # Randomly pick pixels
# pix_id = np.arange(data_valid_rows.shape[1] * data_valid_rows.shape[2])

# from astropy.convolution import convolve, TrapezoidDisk2DKernel

# def spread_bad_pixels(bad_pix_im, kernel=None, radius=1., frac_spread=0.7):
    
#     if kernel is None:
#         kernel = TrapezoidDisk2DKernel(radius)
    
#     im = np.zeros_like(bad_pix_im, dtype=float)
#     im[rnd_id_2d] = 1
    
#     im_conv = convolve(im, kernel, boundary='fill', fill_value=0.)
#     im_conv_rav = im_conv.ravel()
#     mask_rav = (rng.random(im_conv_rav.size) * im_conv_rav.max()) < im_conv_rav * frac_spread
#     mask = mask_rav.reshape(im_conv.shape)
    
#     return mask

# %matplotlib notebook

# spread_frac = 0.3
# n_cosmic = 0  # not yet implemented, keep to 0

# pix_test_mask = np.zeros_like(mask_valid_rows)
# for i_frame in range(data_valid_rows.shape[0]):
#     rnd_int = rng.integers(0, cond[i_frame].sum(), n_test)
#     rnd_id = pix_id[cond[i_frame].ravel()][rnd_int]
    
    
#     # Some bad pix are spreaded
#     n_spreaded = int(spread_frac * n_test)
    
#     rnd_id_2d = np.unravel_index(rnd_id[:n_spreaded], cond[i_frame].shape)
#     temp_mask = np.zeros_like(pix_test_mask[i_frame])
#     temp_mask[rnd_id_2d] = True
#     temp_mask = spread_bad_pixels(pix_test_mask[i_frame], radius=1)
#     pix_test_mask[i_frame] |= temp_mask
    
# #   # Some bad pix are cosmic rays
# #   # but not yet implemented
    
#     # Add the rest
#     rnd_id_2d = np.unravel_index(rnd_id[n_spreaded+n_cosmic:], cond[i_frame].shape)
#     temp_mask = np.zeros_like(pix_test_mask[i_frame])
#     temp_mask[rnd_id_2d] = True
#     pix_test_mask[i_frame] |= temp_mask
    
# # Make sure to remove real bad pixels from the test sample after spreading
# pix_test_mask &= ~mask_valid_rows

# # Create test timeseries
# test_data = np.where(pix_test_mask, np.nan, data_valid_rows)

# # Replace bad pixels with the  function  specified
# filled_data = replace_fct(test_data, *fct_args, **fct_kwargs)


# # Diagnostic plots

# # This is not a good name, but it takes only the pixels that are tested
# test_results = ((data_valid_rows - filled_data) / err_valid_rows)[pix_test_mask]

# # This is the same as above, but with the same shape as the data, the pixels that are not tested are set to nan
# test_results_col = np.where(pix_test_mask, (data_valid_rows - filled_data) / err_valid_rows, np.nan)


# # Many figures to show the results
# plt.figure(figsize=(12,4))
# plt.errorbar(np.arange(test_results_col.shape[1]), np.nanmean(test_results_col, axis=(0, 2)),
#              yerr=np.nanstd(test_results_col, axis=(0, 2)), alpha=0.2)
# plt.plot(np.arange(test_results_col.shape[1]), np.nanmean(test_results_col, axis=(0, 2)), ".")

# plt.figure(figsize=(12,4))
# plt.errorbar(np.arange(test_results_col.shape[2]), np.nanmean(test_results_col, axis=(0, 1)),
#              yerr=np.nanstd(test_results_col, axis=(0, 1)), alpha=0.2)
# plt.plot(np.arange(test_results_col.shape[2]), np.nanmean(test_results_col, axis=(0, 1)), ".")

# plt.figure(figsize=(12,4))
# plt.errorbar(np.arange(test_results_col.shape[0]), np.nanmean(test_results_col, axis=(1, 2)),
#              yerr=np.nanstd(test_results_col, axis=(1, 2)), alpha=0.2)
# plt.plot(np.arange(test_results_col.shape[0]), np.nanmean(test_results_col, axis=(1, 2)), ".")

# print('chi2_red =', 0.5 * np.mean(test_results**2))

# plt.plot(data_valid_rows[pix_test_mask], test_results, ".")
# idx_sort = np.argsort(data_valid_rows[pix_test_mask])
# plt.plot(data_valid_rows[pix_test_mask][idx_sort],
#          np.convolve(test_results[idx_sort], np.ones(100)/100, mode='same'))
# plt.ylabel('Reconstruction Error [sigma]')
# plt.xlabel('Pixel count')

# frame, rows, cols = np.indices(data_valid_rows.shape)

# plt.plot(rows[pix_test_mask], test_results, ".")
# plt.ylabel('Reconstruction Error [sigma]')
# plt.xlabel('Rows')

# %matplotlib notebook

# plt.plot(cols[pix_test_mask], test_results, ".")
# plt.ylabel('Reconstruction Error [sigma]')
# plt.xlabel('Columns')

# plt.plot(frame[pix_test_mask], test_results, ".")
# plt.ylabel('Reconstruction Error [sigma]')
# plt.xlabel('Integration number')

# plt.hist(test_results.ravel(), bins=80)
# plt.axvline(0, color="k")


