import bottleneck as bn
import numpy as np
import os
import warnings

def format_out_frames(baseline_ints, occultation_type='transit'):
    """From supreme_spoon
    Create a mask of baseline flux frames for lightcurve normalization.
    Either out-of-transit integrations for transits or in-eclipse integrations
    for eclipses.

    Parameters
    ----------
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    out_frames : array-like[int]
        Array of out-of-transit, or in-eclipse frames for transits and
        eclipses respectively.

    Raises
    ------
    ValueError
        If an unknown occultation type is passed.
    """

    if occultation_type == 'transit':
        # Format the out-of-transit integration numbers.
        baseline_ints = np.abs(baseline_ints)
        out_frames = np.concatenate([np.arange(baseline_ints[0]),
                                     np.arange(baseline_ints[1]) - baseline_ints[1]])
    elif occultation_type == 'eclipse':
        # Format the in-eclpse integration numbers.
        out_frames = np.linspace(baseline_ints[0], baseline_ints[1],
                                 baseline_ints[1] - baseline_ints[0] + 1).astype(int)
    else:
        msg = 'Unknown Occultaton Type: {}'.format(occultation_type)
        raise ValueError(msg)

    return out_frames

def format_in_frames(transit_ints, occultation_type='transit'):
    """
    Either in-transit integrations for transits or in-eclipse integrations
    for eclipses.

    Parameters
    ----------
    transit_ints : array-like[int]
        Integration numbers t2 and t3.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    in_frames : array-like[int]
        Array of in-transit, or in-eclipse frames for transits and
        eclipses respectively.

    Raises
    ------
    ValueError
        If an unknown occultation type is passed.
    """

    if occultation_type == 'transit':
        # Format the in-transit integration numbers.
        in_frames = np.arange(transit_ints[0], transit_ints[1])
    elif occultation_type == 'eclipse':
        msg = 'Not implemented yet for occulation type : {}'.format(occultation_type)
        raise ValueError(msg)
    else:
        msg = 'Unknown Occultaton Type: {}'.format(occultation_type)
        raise ValueError(msg)

    return in_frames

def normalization(lightcurve, baseline_ints, occultation_type='transit'):
    """Normalizes light curve with out of frames data.

    Parameters
    ----------
    lightcurve : array-like[float]
        Extracted flux of a TSO. [ints, wl]
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    occultation_type : str
        Type of occultation, either 'transit' or ''eclipse'.

    Returns
    ----------
    lc_norm : array_like[float]
        Normalized light curve.
    """

    # Normalizes light curve with median of out of frames data.
    out_frames = format_out_frames(baseline_ints, occultation_type)
    lc_norm = lightcurve / np.nanmedian(lightcurve[out_frames], axis=0)

    return lc_norm

def transit_depth(f_lambda, baseline_ints, transit_ints, occultation_type='transit'):
    """
    Parameters
    ----------
    f_lambda : array-like[float]
        Flux array
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress. (t1, t4)
    transit_ints : array-like[int]
        Integration numbers of complete transit (t2, t3)

    Returns
    ----------
    depth : array_like[float]
        Transit light curve.
    """
    # To make sure we measure the deep in transit mean
    transit_ints[0] = transit_ints[0]+10
    transit_ints[1] = transit_ints[1]-10

    # Median flux value during transit for all wavelengths
    in_frames = format_in_frames(transit_ints, occultation_type)
    in_transit_med = np.nanmedian(f_lambda[in_frames], axis=0)
    # Median flux value during out of transit for all wavelengths
    out_frames = format_out_frames(baseline_ints, occultation_type)
    out_transit_med = np.nanmedian(f_lambda[out_frames], axis=0)

    depth = out_transit_med - in_transit_med
    return depth


