import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.lib.suffix import remove_suffix
from matplotlib import rcParams

rcParams["image.origin"] = "lower"


def nanmediandev(x, axis=None):
    med = np.nanmedian(x, axis=axis)

    return np.nanmedian(np.abs(x - med), axis=axis) / 0.67449

def mediandev(x, axis=None):
    med = np.median(x, axis=axis)

    return np.median(np.abs(x - med), axis=axis) / 0.67449


def stack_ramp(ramp: np.ndarray) -> np.ndarray:
    """
    Stack the ramp along the integration axis (axis=0).

    :param ramp: Ramp (uncal) observation to be stacked (nints, ngroups, npix1, npix2)
    :type ramp: np.ndarray
    :return: Ramp array stacked along integration axis. Shape (ngroups, npix1, npix2)
    :rtype: np.ndarray
    """
    # TODO: Try with etienne's odd_ratio_mean
    # TODO: This handles outliers along int, but not along group or spatially (i.e. same in all int but outlier vs others like hot pixel)
    # Check for nans because nanmedian can be slow and memory-intensive for large files
    if np.isnan(ramp).any():
        deepstack = np.nanmedian(ramp, axis=0)
        #rms = np.nanstd(ramp, axis=0)
        rms = nanmediandev(ramp, axis=0)
    else:
        deepstack = np.median(ramp, axis=0)
        #rms = np.std(ramp, axis=0)
        rms = mediandev(ramp, axis=0)

    return deepstack, rms


def compute_oof(ramp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute 1/f noise for each column in the rampself.

    This function computes the average in each column. The ramp should have
    been subtracted from astrophysical scene and other artifacts (e.g.
    superbias). Data should be:
        - 2D (dimy, dimx)
        - 3D (ngroup or nint, dimy, dimx)
        - >3D (nint, ngroup, dimy, dimx, *)
    For >3D, extra dimensions (e.g. amplificators in the FULL array) should be
    at the end.

    The weighted average is computed per column and broadcast to the same shape
    as ramp

    Parameters
    ----------
    ramp : np.ndarray
        The data from which 1/f noise should be computed
    weights : np.ndarray
        Weight of each pixel in the array, with shape (ngroup, dimy, dimx)

    Returns
    -------
    np.ndarray
        1/f map with the same shape as ramp
    """
    # Get axis that is along columns (with length nrows)
    if ramp.ndim > 3:
        # If more than 3 dim: (nint, ngroup, n, ncol, *others)
        col_axis = 2
    else:
        # If 3 or less: (ngroup?, nrow, ncol) -> -2 is safest
        col_axis = -2
    nrows = ramp.shape[col_axis]

    # Get index shape when we repeat along columns at then end
    # TODO: Should we use -2 for weights, to allow 2D map as well?
    if col_axis >= 0:
        # Positive col_axis, index will be, e.g. [:, :, new, ...] For col = 2
        repeat_ind = np.s_[(slice(None),) * col_axis + (np.newaxis,) + (Ellipsis,)]
    else:
        # Negative col_axis, index will be [..., new, :] For col = -2
        repeat_ind = np.s_[
            (Ellipsis,) + (np.newaxis,) + (slice(None),) * -(col_axis + 1)
        ]

    # Sum along columns to get DC value in each column
    dc = np.nansum(weights * ramp, axis=col_axis) / np.nansum(weights, axis=1)
    # Make sure non nan
    dc = np.where(np.isfinite(dc), dc, 0)
    # Expand along column axis
    return np.repeat(dc[repeat_ind], nrows, axis=col_axis)


def generate_noise_map(
    sub: np.ndarray, pixel_weights: np.ndarray, subarray: str
) -> np.ndarray:
    """
    Generate 1/f noise map for each individual group.

    This function uses numpy broadcasting to generate 1/f noise map and handle
    per-amplificator correction in the FULL array case.

    Parameters
    ----------
    sub : np.ndarray
        Ramp data with stacked integration subtracted from each group
    pixel_weights : np.ndarray
        Weight associated with each pixel when calculing mean over columns.
    subarray : str
        Subarray used to acquire the data. When FULL, correction is done per amplificator.

    Returns
    -------
    np.ndarray
        1/f noise map with same shape as sub
    """

    nint, ngroup, nrow, ncol = sub.shape

    # TODO: Account for outliers if used
    # TODO: Shouldn't weight be set by outliers?, not value
    # sub = sub * outliers
    # Median on all pixels in each frame, keep int and group dims
    # and broadcast to match subarray
    # TODO: Remove this. Very low freq noise could ~offset each group.
    #sub = sub - np.nanmedian(sub, axis=(2, 3))[:, :, None, None]

    if subarray == "FULL":
        # For full array, each amplificator has its own noise
        # All columsn are split in 4 (2048/512)
        amp_nrows = 512
        namps = nrow // amp_nrows
        # I could not get numpy to reshape directly, but moving axes does the trick
        per_amp_shape = (nint, ngroup, namps, amp_nrows, ncol)
        sub_per_amp = np.moveaxis(sub.reshape(per_amp_shape), -3, -1)
        pixel_weights_per_amp = np.moveaxis(
            pixel_weights.reshape(per_amp_shape[1:]), -3, -1
        )
        # Important that namps on last axis.
        # Function below has assumptions about first 4 axes (int, group, y, x)
        dcmap_per_amp = compute_oof(sub_per_amp, pixel_weights_per_amp)
        # Back to original shape. Again because of reshape need to move namp axis before
        dcmap = np.moveaxis(dcmap_per_amp, -1, -3).reshape(sub.shape)
    else:
        # If only one amp (like in all subarrays), vectorized correction works directly
        dcmap = compute_oof(sub, pixel_weights)

    return dcmap


def generate_noise_map_iter(
    sub: np.ndarray, pixel_weights: np.ndarray, subarray: str
) -> np.ndarray:
    """
    Generate the 1/f noise map by looping over integrations.

    This function is kept only in case `generate_noise_map` (the vectorized version)
    causes memory issues. Another reason to use this would be to attempt optimizations
    with something like numba.

    Parameters
    ----------
    sub : np.ndarray
        Ramp data with stacked integration subtracted from each group
    pixel_weights : np.ndarray
        Weight associated with each pixel when calculing mean over columns.
    subarray : str
        Subarray used to acquire the data. When FULL, correction is done per amplificator.

    Returns
    -------
    np.ndarray
        1/f noise map with same shape as sub
    """

    nint, ngroup, _, _ = sub.shape
    dcmap = np.empty_like(sub)
    for i in range(nint):
        # Get ith integration in actual data, subtract median frame from it for each group and each pixel
        #for g in range(ngroup):

            # TODO: Account for outliers if used
            # TODO: Shouldn't weight be set by outliers?, not value
            # sub[i, g, :, :] = sub[i, g, :, :] * outliers[i]
            # TODO: Test with and without this. Just subtracts a mean value for the group before doing 1/f
            #sub[i, g] = sub[i, g] - np.nanmedian(sub[i, g])

        if subarray == "FULL":
            amp_nrows = 512
            for iamp in range(4):
                amp_first_row = iamp * amp_nrows
                int_ind = np.s_[i, :, amp_first_row : amp_first_row + amp_nrows, :]
                dcmap[int_ind] = compute_oof(sub[int_ind], pixel_weights[int_ind[1:]])
        else:
            dcmap[i] = compute_oof(sub[i], pixel_weights)

    return dcmap


def _save_intermediate_fits(
    data: np.ndarray, output_path: Optional[Union[Path, str]], overwrite: bool = True
):
    """
    Helper function to save intermediate products to FITS

    Parameters
    ----------
    data : np.ndarray
        Dataset to save
    output_path : Optional[Union[Path, str]]
        Full Path of the output
    overwrite : bool
        Whether existing files should be overwritten (passed to astropy)
    """
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(output_path, overwrite=overwrite)


def correct_oof(
    input_file: str,
    save_results: bool = False,
    output_dir: Optional[Union[Path, str]] = None,
    outlier_map: Optional[Union[Path, str]] = None,
    iterative: bool = False,
    save_intermediate: bool = False,
    intermediate_output_subdir: Optional[Union[Path, str]] = None,
) -> datamodels.RampModel:
    """
    Correct 1/f noise from JWST ramp data

    Correct 1/f noise by subtracting a stacked image over integration for each group
    and then computing a mean value for each column. For the FULL subarray, each
    amplificator is handled separately.

    Parameters
    ----------
    input_file : str
        Path to the input file
    save_results : bool
        Whether result should be saved or not
    output_dir : Optional[Union[Path, str]]
        Directory where result is saved (defaults to CWD)
    outlier_map : Optional[Union[Path, str]]
        Outler map with dimensions (nint, ny, nx). Outliers are ignored in the correction
    iterative : bool
        Whether or not the 1/f should be calculated by looping over ints.
        Default is to use numpy broadcasting.
        Kept only in case of memory issues for large observations
    save_intermediate : bool
        Whether intermediate outputs should be saved (in separate files).
        Intermediate outputs include the noise map, the stacked ramp along integrations, etc.
    intermediate_output_subdir : Optional[Union[Path, str]]
        The directory where intermedaite outputs should be saved. Default is in output_dir.

    Returns
    -------
    datamodels.RampModel
        JWST data model with the `data` attribute replaced by 1/f-corrected data
    """
    input_model = datamodels.RampModel(input_file)

    # Get stacked ramp (keep group dimenion, but stack along integration)
    stacked_ramp, rms = stack_ramp(input_model.data)
    # TODO: Fix the fact that outliers with very high values have high weight
    # Using outlier map or some sort of thresholding is probalby best.
    pixel_weights = rms**-2
    # Some outliers (probably max value) are fix so have 0 rms and inf weight
    pixel_weights[~np.isfinite(pixel_weights)] = 0.0

    # This automatically subtracts stacked ramp from each int
    sub = input_model.data - stacked_ramp

    # TODO: Without running separate outlier script, could flag some directly here using ramps and stack
    if outlier_map is not None:
        outliers = fits.getdata(outlier_map)

        # outliers == 1 is an outlier. Convert those to NaN, otherwise set to 1
        outliers = np.where(outliers == 0, 1, np.nan)

        # One outlier map per integration. Broadcast to all groups for each
        sub = sub * outliers[:, np.newaxis, ...]

    subarray = input_model.meta.subarray.name
    if iterative:
        dcmap = generate_noise_map_iter(sub, pixel_weights, subarray)
    else:
        dcmap = generate_noise_map(sub, pixel_weights, subarray)

    subcorr = sub - dcmap

    dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

    output_model = input_model.copy()
    output_model.data = input_model.data - dcmap

    if save_results or save_intermediate:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = Path(".")

    if save_intermediate:
        if intermediate_output_subdir is not None:
            output_subdir = output_dir / intermediate_output_subdir
            output_subdir.mkdir(exist_ok=True, parents=True)
        else:
            output_subdir = output_dir
        _save_intermediate_fits(stacked_ramp, output_subdir / "deepstack.fits")
        _save_intermediate_fits(rms, output_subdir / "deepstack_rms.fits")
        _save_intermediate_fits(sub, output_subdir / "sub.fits")
        _save_intermediate_fits(subcorr, output_subdir / "subcorr.fits")
        _save_intermediate_fits(dcmap, output_subdir / "noisemap.fits")

    if save_results:
        input_path = Path(input_file)
        file_id = input_path.stem
        file_id, sep = remove_suffix(file_id)
        output_path = file_id + sep + "oneoverf.fits"

        output_path = output_dir / output_path

        output_model.write(output_path)

    return output_model


if __name__ == "__main__":
    # Open the uncal time series that needs 1/f correction
    exposurename = (
        "scratch/results/stage1/jw01189017001_06101_00001_nis_saturation.fits"
    )
    outdir = "scratch/results/oof_test"

    # Run the 1/f correction step
    # map = applycorrection(uncal_datamodel, exposurename)
    correct_oof(exposurename, save_results=True, output_dir=outdir)
