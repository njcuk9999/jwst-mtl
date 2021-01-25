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
from typing import List, Tuple, Union
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
# columns in output file
RA_OUTCOL = 'x_or_RA'
DEC_OUTCOL = 'y_or_Dec'
F380M_OUTCOL = 'niriss_f380m_magnitude'
F430M_OUTCOL = 'niriss_f430m_magnitude'
F480M_OUTCOL = 'niriss_f480m_magnitude'


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
                 outfile=None, return_data=False
                 ) -> Union[int, Tuple[int, Table]]:
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

    :return: returns position of target in source list table (if return_data is
             False else returns a tuple 1. the position of target in source
             list table, 2. the Table of sources centered on the ra/dec
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
    # get center as SkyCoord
    coord_cent = SkyCoord(ra * uu.deg, dec * uu.deg)

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
    # find our target source (closest to input)
    separation = coord_cent.separation(coords0)
    # sort rest by brightness
    order = np.argsort(cat_table['KMAG'])
    # get the source position (after ordering separation)
    # assume our source is closest to the center
    source_pos = int(np.argmin(separation[order]))
    # -------------------------------------------------------------------------
    # make final table
    # -------------------------------------------------------------------------
    # start table instance
    final_table = Table()
    # index column
    final_table['index'] = np.arange(len(coords1))
    # ra column
    final_table[RA_OUTCOL] = coords1.ra.value[order]
    # dec column
    final_table[DEC_OUTCOL] = coords1.dec.value[order]
    # mag columns
    final_table[F380M_OUTCOL] = cat_table['KMAG'][order]
    final_table[F430M_OUTCOL] = cat_table['KMAG'][order]
    final_table[F480M_OUTCOL] = cat_table['KMAG'][order]
    # -------------------------------------------------------------------------
    # deal with return data
    if return_data:
        return source_pos, final_table
    # -------------------------------------------------------------------------
    # write file
    write_catalog(final_table, outfile)
    # return the position closest to the input coordinates
    return source_pos


def find_target_in_catalog(target_list: List[str], primary_name: str) -> int:
    """
    Find "primary_name" in "target_list" and return the position as an integer

    :param target_list: list of strings - the name of the target (one in each
                        catalog)
    :param primary_name: str, the name of the target in target_list

    :return: int, the position of "primary_name" in "target_list"
    """
    # find primary in target list
    tpos = np.where(np.array(target_list) == primary_name)[0]
    # deal with round target name
    if len(tpos) == 0:
        emsg = 'Primary name: {0} not in "target_list"'
        raise ValueError(emsg.format(primary_name))
    else:
        # return the position of "primary_name" in "target_list"
        return int(tpos)


def modify_primary(target_list: List[str], catalog_tables: List[Table],
                   target_pos: int, primary_name: str,
                   primary_ra: Union[float, None] = None,
                   primary_dec: Union[float, None] = None,
                   primary_f380m: Union[float, None] = None,
                   primary_f430m: Union[float, None] = None,
                   primary_f480m: Union[float, None] = None,
                   log: bool = False) -> List[Table]:
    """
    Modify a target with name "primary name" based on its "target_pos" in
    one of the "catalog_tables".

    We can modify ra (primary_ra), dec (primary dec) or any of the magnitudes
    (primary_f380m, primary_f430m, primary_f480m) if we do not want to set
    a value it should be set to None

    :param target_list: list of strings - the name of the target (one in each
                        catalog)
    :param catalog_tables: list of tables - the catalogs for each target
                           (sources around each target) - one for each target
                           name in target_list
    :param target_pos: int, the position of target in its catalog table (only
                       one target, even though we take in all catalog_tables)
    :param primary_name: str, the name of the target in target_list
    :param primary_ra: float, the new RA value to add to the target, set to
                       None to not update RA value
    :param primary_dec: float, the new Dec to add to the target, set to None to
                        not update Dec value
    :param primary_f380m: float, the new F380M magnitude to add to the target
                          set to None to not update
    :param primary_f430m: float, the new F430M magnitude to add to the target
                          set to None to not update
    :param primary_f480m: float, the new F480M magnitude to add to the target
                          set to None to not update

    :return: list of tables - the catalog for each target with the updated
                              target + new target properties
    """
    # get the position of "primary_name" in "target_list"
    tpos = find_target_in_catalog(target_list, primary_name)
    # get table for this position
    table = catalog_tables[tpos]
    # get the source position in given table
    source_position = target_pos[tpos]
    # -------------------------------------------------------------------------
    # add RA if set
    if primary_ra is not None:
        # log change
        if log:
            msgargs = [table[RA_OUTCOL][source_position], primary_ra]
            print('\t RA UPDATED {0}-->{1}'.format(*msgargs))
        # update table
        table[RA_OUTCOL][source_position] = primary_ra
    # add Dec if set
    if primary_dec is not None:
        # log change
        if log:
            msgargs = [table[DEC_OUTCOL][source_position], primary_dec]
            print('\t Dec UPDATED {0}-->{1}'.format(*msgargs))
        # update table
        table[DEC_OUTCOL][source_position] = primary_dec
    # add F380M magnitude if set
    if primary_f380m is not None:
        # log change
        if log:
            msgargs = [table[F380M_OUTCOL][source_position], primary_f380m]
            print('\t F380M UPDATED {0}-->{1}'.format(*msgargs))
        # update table
        table[F380M_OUTCOL][source_position] = primary_f380m
    # add F430M magnitude if set
    if primary_f430m is not None:
        # log change
        if log:
            msgargs = [table[F430M_OUTCOL][source_position], primary_f430m]
            print('\t F430M UPDATED {0}-->{1}'.format(*msgargs))
        # update table
        table[F430M_OUTCOL][source_position] = primary_f430m
    # add F480M magnitude if set
    if primary_f480m is not None:
        # log change
        if log:
            msgargs = [table[F480M_OUTCOL][source_position], primary_f480m]
            print('\t F480M UPDATED {0}-->{1}'.format(*msgargs))
        # update table
        table[F480M_OUTCOL][source_position] = primary_f480m
    # -------------------------------------------------------------------------
    # push back into catalog table list
    catalog_tables[tpos] = table
    # return catalog of tables
    return catalog_tables


def add_companion_to_cat_entry(target_list: List[str],
                               catalog_tables: List[Table],
                               target_pos: int, primary_name: str,
                               companion_separation: float,
                               companion_pa: float,
                               companion_dm: float) -> List[Table]:
    """
    Add a companion to the catalog table (primary target at position
    "target pos" - with name "primary name"

    Companion is added with:
        separation [arcsec] from primary = "companion_separation"
        position angle from N = "companion_pa"
        delta magnitude from primary ="companion_dm"

    :param target_list: list of strings - the name of the target (one in each
                        catalog)
    :param catalog_tables: list of tables - the catalogs for each target
                           (sources around each target) - one for each target
                           name in target_list
    :param target_pos: int, the position of target in its catalog table (only
                       one target, even though we take in all catalog_tables)
    :param primary_name: str, the name of the target in target_list
    :param companion_separation: float, the separation in arcsec
    :param companion_pa: float, the position angle (angle from North) in degrees
    :param companion_dm: float, the delta magnitude (Kmag) from the primary
                         i.e. dm = 5 is 5 magnitudes fainter than primary

    :return: list of tables - the catalog for each target with the companion
                              added to the end of the correct table
    """
    # get the position of "primary_name" in "target_list"
    tpos = find_target_in_catalog(target_list, primary_name)
    # get table for this position
    table = catalog_tables[tpos]
    # get the source position in given table
    source_position = target_pos[tpos]
    # change the separation to degrees
    companion_separation = companion_separation / 3600.0
    companion_separation = companion_separation * uu.arcsec
    # get ra and dec and magnitudes for primary
    primary_ra = table[RA_OUTCOL][source_position] * uu.deg
    primary_dec = table[DEC_OUTCOL][source_position] * uu.deg
    primary_f380m = table[F380M_OUTCOL][source_position]
    primary_f430m = table[F430M_OUTCOL][source_position]
    primary_f480m = table[F480M_OUTCOL][source_position]
    # primary coords as SkyCoord instance
    primary_coord = SkyCoord(primary_ra, primary_dec)
    # get companion coordinates
    companion_coord = primary_coord.directional_offset_by(companion_pa * uu.deg,
                                                          companion_separation)
    # get the companion magnitudes
    companion_f380m = primary_f380m + companion_dm
    companion_f430m = primary_f430m + companion_dm
    companion_f480m = primary_f480m + companion_dm
    # get the last row entry
    pos = np.max(table['index']) + 1
    # add entry to table
    table.add_row([pos, companion_coord.ra.value, companion_coord.dec.value,
                   companion_f380m, companion_f430m, companion_f480m])
    # push back into catalog table list
    catalog_tables[tpos] = table
    # return catalog of tables
    return catalog_tables


def write_catalog(catalog_table: Table, outfile: Union[str, None]):
    """
    Write a catalogue file in the correct format for Mirage

    :param catalog_table: astropy Table, the catalogue table
    :param outfile: str or None, if set this is the output filename and path
                    if None uses the default in the code

    :return: None - writes the table to disk
    """
    # construct out file name
    if outfile is None:
        outfile = OUTPUT_TABLE
    # log progress
    print('\nWriting catalog to {0}'.format(os.path.realpath(outfile)))
    # write table
    catalog_table.write(outfile, format='ascii.commented_header',
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
