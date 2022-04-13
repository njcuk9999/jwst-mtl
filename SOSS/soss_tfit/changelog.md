# Change log / Notes



Starting point "spectral_transitFit_WASP-52b_david-ord12.ipydb"


- structure
  - core: core functionality (should probably not be changed by user)
  - science: possible changes (mcmc, general, plot)
  - recipes: what the user runs
  - inputs: example inputs

- all constants moved to "params"
      - special dictionary class
- all_spec in data.spec (previously "all_spec")

- binned spectrum overrides "spec" (previously "all_spec_binned")

- phot is now a dictionary in "data":
     - keys: array: WAVELENGTH, FLUX, FLUX_ERROR, TIME, ITIME
     - size: all arrays: (n_phot, n_int)
     - previous list of classes (one for each wavelength)