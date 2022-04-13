# Change log



Starting point "spectral_transitFit_WASP-52b_david-ord12.ipydb"



- all constants moved to "params"
      - special dictionary class
- all_spec in data.spec (previously "all_spec")

- binned spectrum overrides "spec" (previously "all_spec_binned")

- phot is now a dictionary in "data":
     - keys: array: WAVELENGTH, FLUX, FLUX_ERROR, TIME, ITIME
     - size: all arrays: (n_phot, n_int)
     - previous list of classes (one for each wavelength)