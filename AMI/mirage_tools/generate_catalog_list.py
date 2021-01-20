#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    use function:

    make_catalog(ra, dec, radius, year, outfile)

    to generate the catalog

    Make a catalogue of Gaia/2MASS point sources based on a circle of center
    "ra" and "dec" of "radius" [arcsec]

    Output table has following columns
    index, ra, dec, kmag, kmag, kmag

    and is saved in "outfile"

Created on 2021-01-2021-01-20 16:10

@author: cook
"""
from astropy.table import Table
from astropy.time import Time
from astropy import units as uu
from astropy.coordinates import SkyCoord, Distance
from astroquery.utils.tap.core import TapPlus
import numpy as np
import os
from typing import Union
import warnings


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'generate_catalog_list.py'
# center of field
INPUT_RA = 86.8211987087
INPUT_DEC = -51.0665114264
# crossmatch radius (in arcsec)
CROSSMATCH = 120.0
# decimal year of observation
INPUT_YEAR = 2025.0
# Output table name
OUTPUT_TABLE = 'catalog.list'

# -----------------------------------------------------------------------------
# probably don't need to touch these constants
# -----------------------------------------------------------------------------
# Define the Gaia URL
GAIA_URL = 'https://gaia.obspm.fr/tap-server/tap'
# Define the Gaia Query
GAIA_QUERY = """
SELECT
   a.ra as ra, a.dec as dec, a.source_id as gaia_id, a.parallax as plx, 
   a.pmdec as pmde, a.pmra as pmra, 
   a.phot_g_mean_mag as gmag, a.phot_bp_mean_mag as bpmag, 
   a.phot_rp_mean_mag as rpmag, b.source_id as gaia_id_tmass,
   b.tmass_oid as {GAIA_TWOMASS_ID}
FROM 
   gaiadr2.gaia_source as a, gaiadr2.tmass_best_neighbour as b
WHERE 
   1=CONTAINS(POINT('ICRS', ra, dec),
              CIRCLE('ICRS', {RA}, {DEC}, {RADIUS}))
   AND (a.source_id = b.source_id)
"""
# Define some columns
GAIA_TWOMASS_ID = 'tmass_id_gaia'
# Define 2MASS URL
TWOMASS_URL = 'https://gaia.obspm.fr/tap-server/tap'
# Define 2MASS query
TWOMASS_QUERY = """
SELECT 
   j_m as jmag, h_m as hmag, ks_m as kmag, 
   tmass_oid as {TWOMASS_ID} 
FROM
   extcat.tmass
WHERE
   (tmass_oid = {ID})

"""
# Define some columns
TWOMASS_ID = 'tmass_id_tmass'
# header of output file
header = """#
# VEGAMAG
#
#
"""


# =============================================================================
# Define functions
# =============================================================================
def tap_query(url: str, query: str) -> Union[Table, None]:
    """
    Query via a TapPlus query

    :param url: str, the URL to the SQL database
    :param query: str, the SQL query to run

    :return: astropy.table.Table or None - the results of the gaia TAP query
    """
    # ------------------------------------------------------------------
    # try running gaia query
    with warnings.catch_warnings(record=True) as _:
        # construct TapPlus instance
        tap = TapPlus(url=url, verbose=False)
        # launch tap job
        job = tap.launch_job(query=query)
        # get tap table
        table = job.get_results()
    # ------------------------------------------------------------------
    # if we have no entries we did not find object
    if len(table) == 0:
        # return None
        return None
    # else we return result
    return table


def make_catalog(ra: float, dec: float, radius: float, year: float,
                 outfile = None):
    """
    Make a catalogue of Gaia/2MASS point sources based on a circle of center
    "ra" and "dec" of "radius" [arcsec]

    Output table has following columns
    index, ra, dec, kmag, kmag, kmag

    and is saved in "outfile"

    :param ra: float, the Right Ascension in degrees
    :param dec: float, the Declination in degrees
    :param radius: float, the field radius (in arc seconds)
    :param year: float, the decimal year (to propagate ra/dec with proper
                 motion/plx)

    :return: None - makes file "outfile"
    """
    print('='*50)
    print('Field Catalog Generator')
    print('='*50)

    # log input parameters
    print('\nMaking catalog for field centered on:')
    print('\t RA: {0}'.format(ra))
    print('\t DEC: {0}'.format(ra))
    print('\n\tradius = {0} arcsec'.format(radius))
    print('\tObservation date: {0}'.format(year))

    # get observation time
    with warnings.catch_warnings(record=True) as _:
        obs_time = Time(year, format='decimalyear')

    # -------------------------------------------------------------------------
    # Query Gaia - need proper motion etc
    # -------------------------------------------------------------------------
    # construct gaia query
    gaia_query = GAIA_QUERY.format(RA=ra, DEC=dec, RADIUS=radius/3600.0,
                                   GAIA_TWOMASS_ID=GAIA_TWOMASS_ID)
    # define gaia time
    gaia_time = Time('2015.5', format='decimalyear')
    # run Gaia query
    print('\nQuerying Gaia field\n')
    gaia_table = tap_query(GAIA_URL, gaia_query)
    # -------------------------------------------------------------------------
    # Query 2MASS - need to get J, H and Ks mag
    # -------------------------------------------------------------------------
    jmag, hmag, kmag, tmass_id = [], [], [], []
    # now get 2mass magnitudes for each entry
    for row in range(len(gaia_table)):
        # log progress
        pargs = [row + 1, len(gaia_table)]
        print('Querying 2MASS source {0} / {1}'.format(*pargs))
        # query 2MASS for magnitudes
        tmass_query = TWOMASS_QUERY.format(ID=gaia_table[GAIA_TWOMASS_ID][row],
                                           TWOMASS_ID=TWOMASS_ID)
        # run 2MASS query
        tmass_table = tap_query(TWOMASS_URL, tmass_query)
        # deal with no entry
        if tmass_query is None:
            jmag.append(np.nan)
            hmag.append(np.nan)
            kmag.append(np.nan)
            tmass_id.append('NULL')
        else:
            jmag.append(tmass_table['jmag'][0])
            hmag.append(tmass_table['hmag'][0])
            kmag.append(tmass_table['kmag'][0])
            tmass_id.append(tmass_table[TWOMASS_ID][0])
    # add columns to table
    gaia_table['JMAG'] = jmag
    gaia_table['HMAG'] = hmag
    gaia_table['KMAG'] = kmag
    gaia_table[TWOMASS_ID] = tmass_id
    # -------------------------------------------------------------------------
    # Clean up table - remove all entries without 2MASS
    # -------------------------------------------------------------------------
    # remove rows with NaNs in 2MASS magnitudes
    mask = np.isfinite(gaia_table['JMAG'])
    mask &= np.isfinite(gaia_table['HMAG'])
    mask &= np.isfinite(gaia_table['KMAG'])
    # mask table
    cat_table = gaia_table[mask]
    # -------------------------------------------------------------------------
    # Apply space motion
    # -------------------------------------------------------------------------
    # get entries as numpy arrays (with units)
    ra_arr = np.array(cat_table['ra']) * uu.deg
    dec_arr = np.array(cat_table['dec']) * uu.deg
    pmra_arr = np.array(cat_table['pmra']) * uu.mas/uu.yr
    pmde_arr = np.array(cat_table['pmde']) * uu.mas/uu.yr
    plx_arr = np.array(cat_table['plx']) * uu.mas
    # Get sky coords instance
    coords0 = SkyCoord(ra_arr, dec_arr,
                       pm_ra_cosdec=pmra_arr, pm_dec=pmde_arr,
                       distance=Distance(parallax=plx_arr),
                       obstime=gaia_time)
    # apply space motion
    with warnings.catch_warnings(record=True) as _:
        coords1 = coords0.apply_space_motion(obs_time)
    # -------------------------------------------------------------------------
    # make final table
    # -------------------------------------------------------------------------
    # start table instance
    final_table = Table()
    # index column
    final_table['index'] = np.arange(1, len(cat_table) + 1)
    # ra column
    final_table['x_or_RA'] = coords1.ra.value
    # dec column
    final_table['y_or_Dec'] = coords1.dec.value
    # mag columns
    final_table['niriss_f380m_magnitude'] = cat_table['KMAG']
    final_table['niriss_f430m_magnitude'] = cat_table['KMAG']
    final_table['niriss_f480m_magnitude'] = cat_table['KMAG']
    # construct out file name
    if outfile is None:
        outfile = OUTPUT_TABLE
    # log progress
    print('\nWriting catalog to {0}'.format(os.path.realpath(outfile)))
    # write table
    final_table.write(outfile, format='ascii.commented_header',
                      comment=header, overwrite=True)


# =============================================================================
# Start of code
# =============================================================================
# only run if running code (probably should import and use make_catalog
#    function)
if __name__ == "__main__":
    # run main code
    make_catalog(INPUT_RA, INPUT_DEC, CROSSMATCH, INPUT_YEAR)

# =============================================================================
# End of code
# =============================================================================
