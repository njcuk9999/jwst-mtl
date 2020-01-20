This is intended as an overview of all of the AWESimSOSS information currently available to the UdeM team - and contained within this folder.

# Contents

-[installation guides.md](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/installation_guides.md) - explanation of how to install awesimsoss on your local machine.

-[demo_awesimsoss.ipynb](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/demo_awesimsoss.ipynb) - an adaptation of the demo notebook that comes with your awesimsoss installation. It will walk you through the majority of the available functionalities provided by awesimsoss.

-[ng2ni1.fits](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/ng2ni1.fits) - An example output from an awesimsoss simulation of the stellar spectrum contained in scaled_spectrum.txt . The simulation has one integration with two groups. No planet is included.

-[ng4ni3.fits](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/ng4ni3.fits) - An awesimsoss simulation of scaled_spectrum.txt, with four groups and three integrations.

-[ng4ni3_uncal.fits](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/ng4ni3_uncal.fits) - An awesimsoss simulation, identical to ng4ni3.fits, but without noise.

-[scaled_spectrum.txt](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/awesimsoss/scaled_spectrum.txt) - the input spectrum to create the above awesimsoss simulations.

Note: the current version of awesimsoss does not produce headers that are compatible with the (Stage 1) STSci pipeline. To succesfully run the pipeline on awesimsoss files you need to change/add the following keywords:

-DATE-OBS is currently given as MM/DD/YYYY but the pipeline expects DD/MM/YYYY.  
-NRSTSTRT the number of resets at the start of the exposure, is missing - should be 1.  
-NRESETS the number of resets between integrations, is missing - should be 1.  
