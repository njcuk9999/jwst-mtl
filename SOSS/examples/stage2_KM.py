#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MCR, modifications by KM

Custom JWST DMS pipeline steps for Stage 2 (Spectroscopic processing).
"""

from astropy.io import fits
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_boxextract
from jwst.pipeline import calwebb_spec2

from supreme_spoon import plotting, utils


class AssignWCSStep:
    """Wrapper around default calwebb_spec2 Assign WCS step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'assignwcsstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Assign WCS Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.assign_wcs_step.AssignWcsStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class SourceTypeStep:
    """Wrapper around default calwebb_spec2 Source Type Determination step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'sourcetypestep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Source Type Determination Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.srctype_step.SourceTypeStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class FlatFieldStep:
    """Wrapper around default calwebb_spec2 Flat Field Correction step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'flatfieldstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Flat Field Correction Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.flat_field_step.FlatFieldStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class BadPixStep:
    """Wrapper around custom Bad Pixel Correction Step.
    """

    def __init__(self, input_data, smoothed_wlc, baseline_ints,
                 output_dir='./', occultation_type='transit'):
        """Step initializer.
        """

        self.tag = 'badpixstep.fits'
        self.output_dir = output_dir
        self.smoothed_wlc = smoothed_wlc
        self.baseline_ints = baseline_ints
        self.occultation_type = occultation_type
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, thresh=3, box_size=5, max_iter=2, save_results=True,
            force_redo=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_deep = self.output_dir + self.fileroot_noseg + 'deepframe.fits'
            if expected_file not in all_files:
                do_step *= 0
            else:
                results.append(datamodels.open(expected_file))
        if do_step == 1 and force_redo is False:
            print('Output files already exist.')
            print('Skipping Bad Pixel Correction Step.\n')
            deepframe = fits.getdata(expected_deep)
        # If no output files are detected, run the step.
        else:
            step_results = badpixstep(self.datafiles,
                                      baseline_ints=self.baseline_ints,
                                      smoothed_wlc=self.smoothed_wlc,
                                      output_dir=self.output_dir,
                                      save_results=save_results,
                                      fileroots=self.fileroots,
                                      fileroot_noseg=self.fileroot_noseg,
                                      occultation_type=self.occultation_type,
                                      max_iter=max_iter, thresh=thresh,
                                      box_size=box_size)
            results, deepframe = step_results

        return results, deepframe


class TracingStep:
    """Wrapper around custom Tracing Step.
    """

    def __init__(self, input_data, deepframe, output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.deepframe = deepframe
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, mask_width, save_results=True, force_redo=False,
            show_plots=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        suffix = 'tracemask_width{}.fits'.format(mask_width)
        expected_file = self.output_dir + self.fileroot_noseg + suffix
        expected_cen = self.output_dir + self.fileroot_noseg + 'centroids.csv'
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Tracing Step.\n')
            tracemask = fits.getdata(expected_file)
            centroids = pd.read_csv(expected_cen, comment='#')
        # If no output files are detected, run the step.
        else:
            step_results = tracingstep(self.datafiles, self.deepframe,
                                       output_dir=self.output_dir,
                                       mask_width=mask_width,
                                       save_results=save_results,
                                       show_plots=show_plots,
                                       fileroot_noseg=self.fileroot_noseg)
            tracemask, centroids = step_results

        return tracemask, centroids


class LightCurveEstimateStep:
    """Wrapper around custom Light Curve Estimation Step.
    """

    def __init__(self, input_data, baseline_ints, output_dir='./',
                 occultation_type='transit'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.occultation_type = occultation_type
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, smoothing_scale=None, save_results=True, force_redo=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        suffix = 'lcestimate.npy'
        expected_file = self.output_dir + self.fileroot_noseg + suffix
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Light Curve Estimation Step.\n')
            smoothed_lc = np.load(expected_file)
        # If no output files are detected, run the step.
        else:
            smoothed_lc = lcestimatestep(self.datafiles,
                                         baseline_ints=self.baseline_ints,
                                         save_results=save_results,
                                         output_dir=self.output_dir,
                                         occultation_type=self.occultation_type,
                                         fileroot_noseg=self.fileroot_noseg,
                                         smoothing_scale=smoothing_scale)

        return smoothed_lc


def badpixstep(datafiles, baseline_ints, smoothed_wlc=None, thresh=3,
               box_size=5, max_iter=2, output_dir='./', save_results=True,
               fileroots=None, fileroot_noseg='', occultation_type='transit'):
    """Identify and correct hot pixels remaining in the dataset. Find outlier
    pixels in the median stack and correct them via the median of a box of
    surrounding pixels in each integration.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    smoothed_wlc : array-like[float]
        Estimate of the normalized light curve.
    thresh : int
        Sigma threshold for a deviant pixel to be flagged.
    box_size : int
        Size of box around each pixel to test for deviations.
    max_iter : int
        Maximum number of outlier flagging iterations.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.
    fileroots : array-like[str], None
        Root names for output files.
    fileroot_noseg : str
        Root file name with no segment information.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    data : list[CubeModel]
        Input datamodels for each segment, corrected for outlier pixels.
    deepframe : array-like[float]
        Final median stack of all outlier corrected integrations.
    """

    print('Starting custom hot pixel interpolation step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)

    data = []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        data.append(currentfile)

        # To create the deepstack, join all segments together.
        # Also stack all the dq arrays from each segement.
        if i == 0:
            cube = currentfile.data
            dq_cube = currentfile.dq
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
            dq_cube = np.concatenate([dq_cube, currentfile.dq], axis=0)

    # Initialize starting loop variables.
    newdata = np.copy(cube)
    newdq = np.copy(dq_cube)
    it = 0

    while it < max_iter:
        print('Starting iteration {0} of {1}.'.format(it + 1, max_iter))

        # Generate the deepstack.
        print(' Generating a deep stack using all integrations...')
        deepframe = utils.make_deepstack(newdata[baseline_ints])
        badpix = np.zeros_like(deepframe)
        count = 0
        nint, dimy, dimx = np.shape(newdata)

        # Loop over whole deepstack and flag deviant pixels.
        for i in tqdm(range(dimx)):
            for j in range(dimy):
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe, box_size_i, i, j,
                                                dimx, dimy)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i,
                                                    j, dimx, dimy)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant (or nan) flag it.
                if np.abs(deepframe[j, i] - med) > thresh * std or np.isnan(deepframe[j, i]):
                    mini, maxi = np.max([0, i - 1]), np.min([dimx - 1, i + 1])
                    minj, maxj = np.max([0, j - 1]), np.min([dimy - 1, j + 1])
                    badpix[j, i] = 1
                    # Also flag cross around the central pixel.
                    badpix[maxj, i] = 1
                    badpix[minj, i] = 1
                    badpix[j, maxi] = 1
                    badpix[j, mini] = 1
                    count += 1

        print(' {} bad pixels identified this iteration.'.format(count))
        # End if no bad pixels are found.
        if count == 0:
            break
        # Replace the flagged pixels in the median integration.
        newdeep, deepdq = utils.do_replacement(deepframe, badpix,
                                               dq=np.ones_like(deepframe),
                                               box_size=box_size)

        # If no lightcurve is provided, estimate it from the current data.
        if smoothed_wlc is None:
            postage = cube[:, 20:60, 1500:1550]
            timeseries = np.sum(postage, axis=(1, 2))
            timeseries = timeseries / np.median(timeseries[baseline_ints])
            # Smooth the time series on a timescale of roughly 2%.
            smoothed_wlc = median_filter(timeseries,
                                         int(0.02 * np.shape(cube)[0]))
        # Replace hot pixels in each integration using a scaled median.
        newdeep = np.repeat(newdeep, nint).reshape(dimy, dimx, nint)
        newdeep = newdeep.transpose(2, 0, 1) * smoothed_wlc[:, None, None]
        mask = badpix.astype(bool)
        newdata[:, mask] = newdeep[:, mask]
        # Set DQ flags for these pixels to zero (use the pixel).
        deepdq = ~deepdq.astype(bool)
        newdq[:, deepdq] = 0

        it += 1

    # Generate a final corrected deep frame for the baseline integrations.
    deepframe = utils.make_deepstack(newdata[baseline_ints])

    current_int = 0
    # Save interpolated data.
    for n, file in enumerate(data):
        currentdata = file.data
        nints = np.shape(currentdata)[0]
        file.data = newdata[current_int:(current_int + nints)]
        file.dq = newdq[current_int:(current_int + nints)]
        current_int += nints
        if save_results is True:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                file.write(output_dir + fileroots[n] + 'badpixstep.fits')

    if save_results is True:
        # Save deep frame.
        hdu = fits.PrimaryHDU(deepframe)
        hdu.writeto(output_dir + fileroot_noseg + 'deepframe.fits',
                    overwrite=True)

    return data, deepframe


def lcestimatestep(datafiles, baseline_ints, save_results=True,
                   output_dir='./', occultation_type='transit',
                   fileroot_noseg='', smoothing_scale=None):
    """Construct a rough estimate of the TSO photometric light curve to use as
    a scaling factor in 1/f noise correction.

    Parameters
    ----------
    datafiles : array-like[str], array-like[CubeModel]
        Input data files.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    save_results : bool
        If True, save results to file.
    output_dir : str
        Directory to which to save outputs.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.
    fileroot_noseg : str
        Root file name with no segment information.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve.

    Returns
    -------
    smoothed_wlc : array-like[float]
        Estimate of the TSO photometric light curve.
    """

    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)

    # Open datafiles and pack into datacube.
    datafiles = np.atleast_1d(datafiles)
    for i, file in enumerate(datafiles):
        current_data = utils.open_filetype(file)
        if i == 0:
            cube = current_data.data
        else:
            cube = np.concatenate([cube, current_data.data], axis=0)

    # Use an area centered on the peak of the order 1 blaze to estimate the
    # photometric light curve.
    postage = cube[:, 20:60, 1500:1550]
    timeseries = np.sum(postage, axis=(1, 2))
    # Normalize by the baseline flux level.
    timeseries = timeseries / np.median(timeseries[baseline_ints])
    # If not smoothing scale is provided, smooth the time series on a
    # timescale of roughly 2% of the total length.
    if smoothing_scale is None:
        smoothing_scale = int(0.02 * np.shape(cube)[0])
    smoothed_lc = median_filter(timeseries, smoothing_scale)

    if save_results is True:
        outfile = output_dir + fileroot_noseg + 'lcestimate.npy'
        np.save(outfile, smoothed_lc)

    return smoothed_lc


def tracingstep(datafiles, deepframe, output_dir='./', mask_width=30,
                save_results=True, show_plots=False, fileroot_noseg=''):
    """Locate the centroids of all three SOSS orders via the edgetrigger
    algorithm. Then create a mask of a given width around the centroids.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    deepframe : str, array-like[float]
        Path to median stack file, or the median stack itself. Should be 2D
        (dimy, dimx).
    output_dir : str
        Directory to which to save outputs.
    mask_width : int
        Mask width, in pixels, around the trace centroids.
    save_results : bool
        If Tre, save results to file.
    show_plots : bool
        If True, display plots.
    fileroot_noseg : str
        Root file name with no segment information.

    Returns
    -------
    tracemask : array-like[bool]
        3D (order, dimy, dimx) trace mask.
    centroids : array-like[float]
        Trace centroids for all three orders.
    """

    print('Starting Tracing Step.')

    # Get centroids for orders one to three.
    if isinstance(deepframe, str):
        deepframe = fits.getdata(deepframe)
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 96:
        subarray = 'SUBSTRIP96'
    else:
        raise NotImplementedError

    # Get the most up to date trace table file
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    tracetable = step.get_reference_file(datafiles[0], 'spectrace')
    # Get centroids via the edgetrigger method
    save_filename = output_dir + fileroot_noseg
    centroids = utils.get_trace_centroids(deepframe, tracetable, subarray,
                                          save_results=save_results,
                                          save_filename=save_filename)
    x1, y1 = centroids[0][0], centroids[0][1]
    x2, y2 = centroids[1][0], centroids[1][1]
    x3, y3 = centroids[2][0], centroids[2][1]
    # Show the extracted centroids over the deepframe is requested.
    if show_plots is True:
        plotting.do_centroid_plot(deepframe, x1, y1, x2, y2, x3, y3)

    # Create the masks for each order.
    weights1 = soss_boxextract.get_box_weights(y1, mask_width, (dimy, dimx),
                                               cols=x1.astype(int))
    weights1 = np.where(weights1 == 0, 0, 1)
    weights2 = soss_boxextract.get_box_weights(y2, mask_width, (dimy, dimx),
                                               cols=x2.astype(int))
    weights2 = np.where(weights2 == 0, 0, 1)
    weights3 = soss_boxextract.get_box_weights(y3, mask_width, (dimy, dimx),
                                               cols=x3.astype(int))
    weights3 = np.where(weights3 == 0, 0, 1)

    # Pack the masks into an array.
    tracemask = np.zeros((3, dimy, dimx))
    tracemask[0] = weights1
    tracemask[1] = weights2
    tracemask[2] = weights3
    # Plot the mask if requested.
    if show_plots is True:
        plotting.do_tracemask_plot(weights1 | weights2 | weights3)

    # Save the trace mask to file if requested.
    if save_results is True:
        hdu = fits.PrimaryHDU(tracemask)
        suffix = 'tracemask_width{}.fits'.format(mask_width)
        hdu.writeto(output_dir + fileroot_noseg + suffix, overwrite=True)

    return tracemask, centroids


def run_stage2(results, baseline_ints, smoothed_wlc=None, save_results=True,
               force_redo=False, mask_width=30, root_dir='./', output_tag='',
               occultation_type='transit', smoothing_scale=None):
    """Run the supreme-SPOON Stage 2 pipeline: spectroscopic processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html

    Parameters
    ----------
    results : array-like[str], array-like[CubeModel]
        supreme-SPOON Stage 1 output files.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    smoothed_wlc : array-like[float], None
        Estimate of the normalized light curve.
    save_results : bool
        If True, save results of each step to file.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    mask_width : int
        Width, in pixels, of trace mask to generate
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    occultation_type : str
        Type of occultation: transit or eclipse.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve.

    Returns
    -------
    results : array-like[CubeModel]
        Datafiles for each segment processed through Stage 2.
    deepframe : array-like[float]
        Median stack of the baseline flux level integrations (i.e.,
        out-of-transit or in-eclipse).
    tracemask : array-like[float]
        Trace mask.
    centroids : array-like[float]
        Trace centroids for all three orders.
    smoothed_wlc : array-like[float]
        Estimate of the photometric light curve.
    """

    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    print('\n\n**Starting supreme-SPOON Stage 2**')
    print('Spectroscopic processing\n\n')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2/'

    # ===== Assign WCS Step =====
    # Default DMS step.
    step = AssignWCSStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo)

    # ===== Source Type Determination Step =====
    # Default DMS step.
    step = SourceTypeStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo)

    # ===== Flat Field Correction Step =====
    # Default DMS step.
    step = FlatFieldStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo)

    # ===== Bad Pixel Correction Step =====
    # Custom DMS step.
    step = BadPixStep(results, baseline_ints=baseline_ints,
                      smoothed_wlc=smoothed_wlc, output_dir=outdir,
                      occultation_type=occultation_type)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results, deepframe = step_results

    # ===== Tracing Step =====
    # Custom DMS step.
    step = TracingStep(results, deepframe=deepframe, output_dir=outdir)
    step_results = step.run(mask_width=mask_width, save_results=save_results,
                            force_redo=force_redo)
    tracemask, centroids = step_results

    # ===== Light Curve Estimation Step =====
    # Custom DMS step.
    step = LightCurveEstimateStep(results, baseline_ints=baseline_ints,
                                  output_dir=outdir,
                                  occultation_type=occultation_type)
    smoothed_wlc = step.run(smoothing_scale=smoothing_scale,
                            save_results=save_results, force_redo=force_redo)

    return results, deepframe, tracemask, centroids, smoothed_wlc