import glob

from astropy.io import ascii

import numpy as np

'''
extract information found in Bjoen Benneke's planet.csv file to fill the simpars.txt config file
:return:
'''

sunmass_kg = 1.989e+30
sunradius_m = 696340000

for planetfile in glob.glob('/genesis/jwst/jwst-ref-soss/planet_model_atm/*/*/planet.csv'):
    a = ascii.read(planetfile)
    mass = a['Mstar'].value[0] * 1000 # gram
    radius = a['Rstar'].value[0] * 100 # cm
    teff = a['Teffstar'].value[0]
    #eccentricity = a['pl_orbeccen'].value[0]
    eccentricity = a['ecc'].value[0]
    #Name, nicename, source, starname, Mstar, Rstar, Lstar, per, ap, tt, T14, Teffstar, Distance, MpMeasured, Vmag, Jmag, Hmag, Ksmag, Mp, Rp, TimeOfPeri, ArguOfPeri, LongOfNode, incli, b, ecc, DEC, pl_tranflag, Unnamed:
    #TeffstarUpp, TeffstarLow

    arguofperi = a['ArguOfPeri'].value[0]
    longofnode = a['LongOfNode'].value[0]

    rho = mass / ((4 * np.pi / 3) * radius ** 3)
    print(planetfile)
    print('RHOSTAR ', rho)
    print('BBTEFF ', teff)
    #print('VSINI ', a['st_vsini'].value[0])
    print('PE1 ', a['pl_orbper'].value[0])
    print('BB1 ', a['pl_imppar'].value[0])
    print('eccentricity ', eccentricity)
    print('JMAG ', a['Jmag'].value[0])
    print('ArguOfPeri ', arguofperi)
    print('LongOfNode ', longofnode)
    print('ES1 ', np.sqrt(eccentricity) * np.sin(np.radians(longofnode)))
    print('EC1 ', np.sqrt(eccentricity) * np.cos(np.radians(longofnode)))
    print('RV1 ', a['pl_rvamp'].value[0])
    print()

    #EP1
    #0.0  # Center of transit time [days]
    #PE1
    #1.7497798  # Orbital period [days]    K.M.
    #BB1
    #0.6  # Impact parameter   K.M.
    #ES1
    #0.0  # sqrt(e)sin(omega)
    #EC1
    #0.0  # sqrt(e)cos(omega)
    #RV1
    #42.5  # Radial velocity semi-amplitude [m/s]  K.M.
