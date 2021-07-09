import SOSS.trace.tracepol as tp
import numpy as np
from astropy.io import fits
import itsosspipeline as soss
import sys
import os
import specgen.spgen as spgen
import scipy.constants as sc_cst
from scipy.interpolate import interp1d
from sys import path
from Fake_data.simu_utils import load_simu

# Imports from the extraction
from extract.overlap import TrpzOverlap
from extract.throughput import ThroughputSOSS
from extract.convolution import WebbKer

# Imports for plots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For displaying of FITS images.

import box_kim

os = 5
os1 = 1

WORKING_DIR = '/home/kmorel/ongenesis/jwst-user-soss/'

x = np.arange(2048)    # Array of pixels

clear_tr_00 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_trace_000000.fits".format(os))
clear_tr_00_1 = fits.open(WORKING_DIR + "tmp/oversampling_{}/clear_trace_000000.fits".format(os1))

padd = 10   # Because of x_padding and y_padding
padd_os = padd * os
clear_tr_ref = clear_tr_00[0].data[:2, padd_os:-padd_os, padd_os:-padd_os]   # Reference thin trace, not binned
clear_tr_ref_1 = clear_tr_00_1[0].data[:2, padd:-padd, padd:-padd]   # Reference thin trace, not binned, os=1

# Bin to pixel native
ref_i = soss.rebin(clear_tr_00[0].data, os, flux_method='mean')
clear_ref_bin = ref_i[:2, padd:-padd, padd:-padd]  # Reference thin trace, binned

data_ref = clear_tr_ref[0]  # Not binned
data_ref_bin = clear_ref_bin[0]    # Binned
data_ref_1 = clear_tr_ref_1[0]   # Os = 1

plt.figure()
plt.imshow(data_ref)
plt.show()

print(data_ref[422, 502])
print(data_ref_1[84, 100])

# Extract flux
fbox_ref = box_kim.flambda_inf_radi_adu(data_ref)
fbox_ref_bin = box_kim.flambda_inf_radi_adu(data_ref_bin)    # Infinite radius [adu/s] (only if order 1 only)
fbox_ref_1 = box_kim.flambda_inf_radi_adu(data_ref_1)        # Infinite radius [adu/s] (only if order 1 only)

indices = x * 5 + 2
print(indices)
new_fbox_ref = fbox_ref[indices]

plt.figure()
#plt.plot(x, fbox_ref_bin/os, label='Binned')
#plt.plot(x, fbox_ref_1, label='Os=1')
plt.plot(x, new_fbox_ref, lw=2, label='fbox_ref')
plt.plot(x, fbox_ref_bin * os, lw=2, label='fbox_ref_bin')
plt.ylabel(r"Extracted flux [adu/s]")
plt.xlabel(r"Pixels [x]")
plt.legend()
plt.show()

sys.exit()

plt.figure()
plt.plot(x, (fbox_ref_bin*os - fbox_ref_1) / fbox_ref_1 * 1e6)
plt.ylabel(r"Relative difference [ppm]")
plt.xlabel(r"Pixels [x]")
plt.show()
