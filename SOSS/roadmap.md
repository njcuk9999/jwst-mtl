This is a first version of a roadmap towards getting a SOSS observation simulated, reduced and extracted.
Items requiring immediate action are in **bold**.
Items that should only be adressed if time permits are in *italics*.

# Simulations

Needed:
- **Short simulation with awesimsoss.** (Needed so people can start work on the Pipeline and Extraction steps.)
- 1 complete GTO observation (to start with) from Jason. (Planet TBD.)
  - Do we need variations (noiseless, star only, star + planet, different planet spectra, etc?)

Simulations need to include the effects of:
- Different orders (m=-2,-1,0,1,2,3).
- Field background.
- Field contamination (adapt from David's code).
  - Input catalog.
  - Transformation to x, y.
  - Estimate of Teff.
  - Input stellar spectra.
- Detector noise.
  - CV3 darks.
  - HxRG noise genrator (Rauscher 2015)
  - Cosmic rays.
  - *Brighter-Fatter effect.* 
- *Jitter.*
- *Filter wheel positions.*

- To be pipeline ready need:
 - Correct headers.
 - Include bias.

# Pipeline

STSci pipeline:
- **Stage 1, use.**
- Stage 2, write our own.
- Stage 3, unused.

- Our stage 2 steps:
  - Background subtraction.
  - Cosmic rays.
  - Flat-fielding.
  - *Revisit 1/f noise.*

# Extraction

Methods of extraction:
- Aperture extraction.
- **Antoine's extraction.**
- *Differential extraction.*
- *AI extraction.*

Common components/steps:
- F277W calibration.
- Find trace positions.
- Wavelength calibration.
- Trace profile model. (m=1,2)
- *Flux calibration.*
- Field contamination subtraction. (A mask is needed?)
- *Influence of filter wheel position on:*
  - *Trace angle.*
  - *Trace x, y.*
 
