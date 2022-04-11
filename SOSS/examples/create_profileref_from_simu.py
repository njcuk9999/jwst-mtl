'''
This script builds a trace profile reference file from
an existing simulation. It uses the clear_000000.fits file
which is heavily padded and oversampled to construct the
trace reference file.
'''

from astropy.io import fits
import numpy as np
from dms.soss_ref_files import init_spec_profile


sim_list = [
    '/genesis/jwst/userland-soss/loic_review/WFE/realization0/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization1/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization2/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization3/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization4/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization5/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization6/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization7/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization8/tmp/clear_000000.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/realization9/tmp/clear_000000.fits'
]

ref_list = [
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization0.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization1.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization2.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization3.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization4.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization5.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization6.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization7.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization8.fits',
    '/genesis/jwst/userland-soss/loic_review/WFE/ref_profiles/ref_profile_realization9.fits',
]

# The padding and oversamspling used to generate the 2D profile.
sim_padding = 100
oversample = 4

# Create one ref file for each simulation
for i in range(np.size(sim_list)):
#for i in range(1):
    print('Processing simulation ',i+1)

    # Read the profile file provided by the simulation
    profile_2d = fits.getdata(sim_list[i], ext=0)
    profile_2d = np.moveaxis(profile_2d, 0, -1)

    # set any negative pixel to zero
    profile_2d[profile_2d < 0] = 0

    # Normalize the trace, column by column, for each order
    profile_2d = profile_2d / np.nansum(profile_2d, axis=0)

    # The provided file is for SUBSTRIP256, we pad this to the FULL subarray.
    nrows, ncols, _ = profile_2d.shape
    dimy = oversample*(2048 + 2*sim_padding)
    dimx = oversample*(2048 + 2*sim_padding)

    tmp = np.full((dimy, dimx, 3), fill_value=np.nan)
    tmp[-nrows:] = profile_2d
    profile_2d = tmp

    # Remove most of the padding (it contains border effects)
    ref_padding = 10
    width = (sim_padding - ref_padding) * oversample
    profile_2d = profile_2d[width:-width,width:-width,:]

    # Fix some bad columns in order 2 and 3 right of where the trace leaves the array
    m2_natpixcutoff = 1875
    m3_natpixcutoff = 900
    profile_2d[:, (m2_natpixcutoff+ref_padding)*oversample:, 1] = 0.0
    profile_2d[:, (m3_natpixcutoff+ref_padding)*oversample:, 2] = 0.0

    # Call init_spec_profile with the prepared input data.
    hdul = init_spec_profile(profile_2d, oversample, ref_padding, 'SUBSTRIP256')

    # If necessary manual changes and additions can be made here, before saving the file.
    filename = hdul[0].header['FILENAME']
    filename = ref_list[i]
    hdul.writeto(filename, overwrite=True)
    hdul.writeto(filename + '.gz', overwrite=True)

