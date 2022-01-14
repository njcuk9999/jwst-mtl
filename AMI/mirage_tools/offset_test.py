from astropy.coordinates import SkyCoord
from astropy import units as uu
import numpy as np
import matplotlib.pyplot as plt


def loic(ra, dec, sep, pa):
	theta = pa + 90 * uu.deg
	ra2 = ra + sep * np.cos(theta) / np.cos(dec)
	dec2 = dec + sep * np.sin(theta)
	return SkyCoord(ra2, dec2, frame='icrs')


ra1, dec1 = 86.82124413 * uu.deg, -51.06593008*uu.deg
separation = 1/3600.0 * uu.deg

coord1 = SkyCoord(ra1, dec1, frame='icrs')


coord2a = coord1.directional_offset_by(0 * uu.deg, separation)
coord3a = coord1.directional_offset_by(90 * uu.deg, separation)
coord4a = coord1.directional_offset_by(180 * uu.deg, separation)
coord5a = coord1.directional_offset_by(-90 * uu.deg, separation)

coords_a = [coord2a, coord3a, coord4a, coord5a]
labels_a = ['astropy PA=0', 'astropy PA=90', 'astropy PA=180', 'astropy PA=-90']
color_a = ['r', 'g', 'b', 'm', 'c']

coord2b = loic(ra1, dec1, separation, 0*uu.deg)
coord3b = loic(ra1, dec1, separation, 90*uu.deg)
coord4b = loic(ra1, dec1, separation, 180*uu.deg)
coord5b = loic(ra1, dec1, separation, -90*uu.deg)

coords_b = [coord2b, coord3b, coord4b, coord5b]
labels_b = ['loic PA=0', 'loic PA=90', 'loic PA=180', 'loic PA=-90']
color_b = ['r', 'g', 'b', 'm', 'c']

plt.plot([coord1.ra.value], [coord1.dec.value], marker='*', color='k', label='target', linestyle='None')

for it, coord in enumerate(coords_a):
	plt.plot([coord.ra.value], [coord.dec.value], marker='o', markeredgecolor=color_a[it], 
		     markerfacecolor='None', label=labels_a[it], linestyle='None')

for it, coord in enumerate(coords_b):
	plt.plot([coord.ra.value], [coord.dec.value], marker='+', ms=10, color=color_b[it], label=labels_b[it], linestyle='None')

plt.xlabel('RA')
plt.ylabel('Dec')
plt.legend(loc=0)
plt.show()
plt.close()
