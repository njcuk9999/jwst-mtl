# NIRISS/SOSS - Transitfit module

## Contents:

A fast, accurate transit model for multi-wavelength observations.

features:

- multiplanet systems
- can handle 1000's of bandpasses
- custom deMCMC routines for efficient sampling
- TTVs
- multicore processing with MPI

Todo:

- GP model is not complete
- Priors needs better book keeping

## Data Format

Observations are stored in a list using the 'phot' class.

photospectra=[]

for i in range(bandpasses):

  phot=phot_class() #class for each wavelength
  phot.wavelength=wavelength[i] 
  phot.time=timestamps_array[i]
  phot.flux=observed_flux_array[i]
  phot.ferr=observed_error_array[i]
  phot.itime=integration_time[i]
  
  photospectra.append(phot)
  
  
## Fitting observations

