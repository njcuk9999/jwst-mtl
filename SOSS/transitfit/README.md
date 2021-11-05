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

```python
photospectra=[]
  
for i in range(bandpasses):
  
  phot=phot_class() #class for each wavelength  
  phot.wavelength=wavelength[i]  
  phot.time=timestamps_array[i]  
  phot.flux=observed_flux_array[i]  
  phot.ferr=observed_error_array[i]  
  phot.itime=integration_time[i]  
  photospectra.append(phot)  
```
  
## Fitting observations

Indicate the number of planets to include  
```python 
nplanet=1
```

#Set up default parameters 
tpars = sptransit_model_parameters([photospectra,nplanet])

#Fill in a few necessary parameters  
tpars.rhostar[0]=np.array([2.91])
tpars.rhostar[3]=np.array([1.0,3.0]) #search boundaries for nested sampling

tpars.period[0][0]=np.array([3.336649])
tpars.period[0][2]='fixed'

tpars.t0[0][0]=np.array([2*0.0791])
tpars.t0[0][3]=np.array([2*0.0791-0.01,2*0.0791+0.01])

tpars.rprs[0][0]=np.ones(len(photospectra))*0.0835
tpars.rprs[0][3]=np.array([0.082,0.086])

#Set search scale for zero-point (normalization)
#fmin=np.min(photospectra[0].flux)
#fmax=np.max(photospectra[0].flux)
#for p in photospectra:
#    fmin=np.min([fmin,np.min(p.flux)])
#    fmax=np.max([fmax,np.max(p.flux)])
#tpars.zeropoint[3]=np.array([fmin,fmax])
tpars.zeropoint[3]=np.array([0.998,1.002])

