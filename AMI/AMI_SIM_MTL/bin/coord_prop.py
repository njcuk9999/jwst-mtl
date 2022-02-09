#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coord_prop.py

Get astrometric parameters at a different point in time

usage: coord_prop.py [-h] [--ra RA] [--dec DEC] [--pmra PMRA] [--pmde PMDE]
                     [--plx PLX] [--coordtime COORDTIME]
                     [--coordtimefmt {decimalyear,jd,mjd,iso}]
                     [--obstime OBSTIME] [--obstimefmt {decimalyear,jd,mjd,iso}]

Apply space motion to coordinates.

Note if ra/dec/pmra/pmde/plx/cordtime/obstime are not given, all values come
from python code.

optional arguments:
  -h, --help            show this help message and exit
  --ra RA               RA in degrees
  --dec DEC             Dec in degrees
  --pmra PMRA           PMRA in mas/yr
  --pmde PMDE           PMDE in mas/yr
  --plx PLX             plx in mas
  --coordtime COORDTIME
                        coordinate time of the ra/dec/pmra/pmde/plx
  --coordtimefmt {decimalyear,jd,mjd,iso}
                        coordinate time format
  --obstime OBSTIME     Time and date to propagate ra/dec/pmra/pmde/plx to
  --obstimefmt {decimalyear,jd,mjd,iso}
                        observation time format

Created on 2022-02-09

@author: cook
"""
import argparse
from astropy.coordinates import SkyCoord, Distance
from astropy import units as uu
from astropy.time import Time
import warnings


# =============================================================================
# Define variables
# =============================================================================
# If not inputting from the command line
#  Note RA/DEC/PMRA/PMDE/PLX must have units
#    (i.e. uu.deg, uu.mas/uu.yr, uu.mas)
INPUT_RA = 269.4520769586187 * uu.deg
INPUT_DEC = 4.6933649665767 * uu.deg
INPUT_PMRA = -801.551 * uu.mas / uu.yr
INPUT_PMDE = 10362.394 * uu.mas / uu.yr
INPUT_PLX = 546.9759 * uu.mas
# Note COORD_TIME and OBS_TIME must be Time objects
#    (formats = 'jd', 'mjd', 'decimalyear', 'iso')

# coord time is the time of the ra/dec/pmra/pmde/plx
INPUT_COORD_TIME = Time(2015.5, format='decimalyear')
# obs time is the time with
INPUT_OBS_TIME = Time('2025-07-02 11:59:59.500', format='iso')


# =============================================================================
# Define functions
# =============================================================================
def get_args() -> dict:
    """
    Get arguments using argparser (or use defaults it not defined)

    Returns
    -------

    props: dict
        The property dictionary
    """
    description = 'Apply space motion to coordinates.'
    description += (' Note if ra/dec/pmra/pmde/plx/cordtime/obstime are '
                    'not given, all values come from python code.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--ra', action='store', default=None,
                        dest='ra', help='RA in degrees')
    parser.add_argument('--dec', action='store', default=None,
                        dest='dec', help='Dec in degrees')
    parser.add_argument('--pmra', action='store', default=None,
                        dest='pmra', help='PMRA in mas/yr')
    parser.add_argument('--pmde', action='store', default=None,
                        dest='pmde', help='PMDE in mas/yr')
    parser.add_argument('--plx', action='store', default=None,
                        dest='plx', help='plx in mas')
    parser.add_argument('--coordtime', action='store', default=None,
                        dest='coordtime',
                        help='coordinate time of the ra/dec/pmra/pmde/plx')
    parser.add_argument('--coordtimefmt', default='decimalyear',
                        dest='coordtimefmt',
                        choices=['decimalyear', 'jd', 'mjd', 'iso'],
                        help='coordinate time format')
    parser.add_argument('--obstime', action='store', default=None,
                        dest='obstime',
                        help='Time and date to propagate ra/dec/pmra/pmde/plx '
                             'to')
    parser.add_argument('--obstimefmt', default='iso',
                        dest='obstimefmt',
                        choices=['decimalyear', 'jd', 'mjd', 'iso'],
                        help='observation time format')
    # parse arguments
    args = parser.parse_args()
    # check for Nones
    cond1 = args.ra is None
    cond2 = args.dec is None
    cond3 = args.pmra is None
    cond4 = args.pmde is None
    cond5 = args.plx is None
    cond6 = args.coordtime is None
    cond7 = args.obstime is None
    # load into props
    props = dict()
    # deal with no inputs
    if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7:
        # define colours
        yellow = '\033[1;93;1m'
        end = '\033[0;0m'
        print(yellow + 'Using values from python code' + end)
        props['ra'] = INPUT_RA
        props['dec'] = INPUT_DEC
        props['pmra'] = INPUT_PMRA
        props['pmde'] = INPUT_PMDE
        props['plx'] = INPUT_PLX
        props['coordtime'] = INPUT_COORD_TIME
        props['obstime'] = INPUT_OBS_TIME
    else:
        props['ra'] = args.ra * uu.deg
        props['dec'] = args.dec * uu.deg
        props['pmra'] = args.pmra * uu.deg
        props['pmde'] = args.pmde * uu.deg
        props['plx'] = args.plx * uu.deg
        props['coordtime'] = Time(args.coordtime, format=args.coordtimefmt)
        props['obstime'] = Time(args.obstime, format=args.obstimefmt)
    # return props
    return props


def propagate_coords(ra: uu.Quantity, dec: uu.Quantity, pmra: uu.Quantity,
                     pmde: uu.Quantity, plx: uu.Quantity, coordtime: Time,
                     obstime: Time) -> dict:
    """
    Propagate coordiantes

    Parameters
    ----------

    ra : astropy quantity
        The Right Ascension with units (e.g. uu.deg)
    dec : astropy quantity
        The declination with units (e.g. uu.deg)
    pmra : astropy quantity
        The proper motion right ascension (cos dec) component with units
        (e.g. uu.mas / uu.yr)
    pmde : astropy quantity
        The proper motion declination component with units
        (e.g. uu.mas / uu.yr)
    plx : astropy quantity
        The parallax with units (e.g. uu.mas)
    coordtime: astropy Time
        The time of the coordinates
    obstime: astropy Time
        The required output coordinate time

    Returns
    -------
    props: dict
        The updated dictionary of parameters
    """
    # deal with distance
    if plx == 0:
        distance = None
    else:
        distance = Distance(parallax=plx)
    # need to propagate ra and dec to J2000
    coords = SkyCoord(ra=ra, dec=dec, distance=distance,
                      pm_ra_cosdec=pmra, pm_dec=pmde,
                      obstime=coordtime)
    # work out the delta time
    delta_time = obstime - coordtime
    # get the coordinates in J2000
    with warnings.catch_warnings(record=True) as _:
        new_coords = coords.apply_space_motion(dt=delta_time.to(uu.day))
    # new_props
    new_props = dict()
    new_props['ra'] = new_coords.ra
    new_props['dec'] = new_coords.dec
    new_props['pmra'] = new_coords.pm_ra_cosdec
    new_props['pmde'] = new_coords.pm_dec
    new_props['plx'] = new_coords.distance.parallax.to(uu.mas)
    new_props['coordtime'] = obstime
    new_props['obstime'] = obstime

    # return the ra and dec quantity
    return new_props


def print_coords(params: dict, kind: str):
    """
    Print coordiantes in friendly way

    Parameters
    ----------
    params: dict, the coordinate dictionary
    kind: str, the coordinate kind

    Returns
    -------
    None, prints to screen
    """
    # define colours
    green = '\033[92;1m'
    end = '\033[0;0m'
    blue = '\033[94;1m'
    # construct print string
    print_str = blue + '\n' + '='*50
    print_str += '\n\t{0}'.format(kind)
    print_str += '\n' + '='*50
    # add ra/dec
    ra_str = 'RA: {0} {1}'.format(params['ra'].value, params['ra'].unit)
    dec_str = 'DEC: {0} {1}'.format(params['dec'].value, params['dec'].unit)
    print_str += green + '\n\t {0:40s} {1}'.format(ra_str, dec_str)
    # add pmra/pmde
    pmra_str = 'PMRA: {0} {1}'.format(params['pmra'].value, params['pmra'].unit)
    pmde_str = 'PMDE: {0} {1}'.format(params['pmde'].value, params['pmde'].unit)
    print_str += green + '\n\t {0:40s} {1}'.format(pmra_str, pmde_str)
    # add plx
    print_str += '\n\t PLX: {plx} '.format(**params)
    # add times
    coordtimestr = 'COORDTIME: {0}'.format(params['coordtime'].iso)
    obstimestr = 'OBSTIME: {0}'.format(params['obstime'].iso)
    print_str += '\n\t {0:40s} {1}'.format(coordtimestr, obstimestr)
    print_str += end
    # print print_str
    print(print_str)


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # get args
    in_params = get_args()
    # propagate coordinates
    out_params = propagate_coords(**in_params)
    # print input parameters
    print_coords(in_params, 'input')
    # print output parameters
    print_coords(out_params, 'output')


# =============================================================================
# End of code
# =============================================================================
