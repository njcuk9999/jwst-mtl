'''
Example for performing spectrum extraction
'''


from astropy.io import fits
import sys
import numpy as np
import matplotlib.pyplot as plt
import trace.tracepol as tp
import itsosspipeline as soss

# Assume that we have a fit file exposure



# Read the exposure
# Extract trace position and determine offset wrt ref file or model (x, y, w relations)
# Option - generate a new set of ref files based on observed set
# Perform classic box extraction column by column
# Perform decontamination
# Perform box extraction on decontaminated images

debug = False

'''
Read in the rateint exposure (processed through DMS)
'''
exposure_name = '/genesis/jwst/userland-soss/loic_review/test_noisy_rateints.fits'
data = fits.getdata(exposure_name)


'''
Establish the trace position relations: y(x), w(x), x(w), y(w)
Method 1 - reading reference files
Method 2 - reading the optics model
Method 3 - measuring the trace, model and figuring conversion from measured vs modelled
'''

#a = tp.test_tracepol()
#sys.exit()

# Method 1 - reading reference files
#TODO: how to read the ref files to get y(x), w(x) , x(w), y(w)

# Method 2 - reading the optics model

if False:
    trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
    pars = tp.get_tracepars(trace_filename, disable_rotation=True)
    # The following way of retrieving the trace center, y, as a function of integer x position DOES
    # NOT work. The retrieved x after converting to wavelength is off by up to 1.2 pixels from
    # the seeded x. Issue is with how tracepol gets the conversion specpix_to_wavelength() by
    # refitting the points rather than by really doing an inversion. Anyway. UNtil that is fixed,
    # resort on interpolating the wave--->specpix at integer x values.
    # x indices
    x_index1 = np.linspace(0,2047,2048)
    w, mask = tp.specpix_to_wavelength(x_index1, pars, m=1)
    x_index2, y_index, mask = tp.wavelength_to_pix(w, pars, m=1, subarray='SUBSTRIP256')
    dx = x_index2 - x_index1
    for i in range(np.size(x_index1)):
        print(i,x_index1[i], x_index2[i], dx[i], y_index[i], w[i])
    plt.figure()
    plt.imshow(data[0,:,:], vmin=0, vmax=1000, origin='lower')
    plt.plot(x_index, y_index, color='black')
    plt.show()

trace_filename = '/genesis/jwst/jwst-ref-soss/trace_model/NIRISS_GR700_trace_extended.csv'
pars = tp.get_tracepars(trace_filename, disable_rotation=False)
w = np.linspace(0.7,3.0,10000)
x, y, mask = tp.wavelength_to_pix(w, pars, m=1, subarray='SUBSTRIP256')
x_index = np.linspace(0,2047,2048).astype(int)
# np.interp needs ordered x
ind = np.argsort(x)
x, w = x[ind], w[ind]
wavelength = np.interp(x_index, x, w)
y_index = np.interp(x_index, x, y)

for i in range(np.size(x_index)):
    print(i, x_index[i], y_index[i])

if False:
    plt.figure()
    plt.scatter(x, y, color='red')
    plt.scatter(x_index, y_index, color='blue')
    plt.show()


# Method 3 - measure the trace, compare with ref files and apply necessary adjustments
#TODO: create the method 3 process


'''
Perform box aperture spectrum extraction
'''

box_aperture = soss.box_aperture(data, x_index, y_index, box_width=30.0)
hdu = fits.PrimaryHDU()
hdu.data = box_aperture
hdu.writeto('/genesis/jwst/userland-soss/loic_review/box_aperture.fits', overwrite=True)
a = soss.aperture_extract(data, x_index, box_aperture, mask=None)

print(a)
print('Extraction example 1 completed successfully.')