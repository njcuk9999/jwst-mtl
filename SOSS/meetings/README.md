## Meeting minutes (year 2020):

Sep 15 2020

- Installation status: Loic is at step extraction
- A pipeline guide draft is on overleaf https://www.overleaf.com/6596335464ywdqbhqmygbc
- Jason and Antoine agreed to make the jwst-mtl github public (a STScI request)
- Went through the list of feature improvements and bugs (listed at the end of the overleaf guide, BTW)
- Discussed the need for planet atmosphere models. PLATON is at R<10k, PetitRadTran is higher but less handle on non-equilibrium EOS. Bjorn’s group was planning to have python tools to easily generate yourself atmosphere models (at R=250k).
- STScI is proposing us to fork their pipeline and add the extraction to it, then make a pull request before mid January 2021 (I&T deadline) so it can make it to the DMS 7.7 version. (Version 7.6 is what is officially supported for Commissioning). We think that may represent having a double path for the extraction package. Need more discussion.

Actions:
- Caroline will generate R=250k planet atmosphere spectra for our GTO and ERS targets (Michel will provide rough atmosphere parameters)
- Jason will generate a grid of star atmosphere spectra (3500K to 10000K in steps of 250K, logg =4.5 or 5.0
- This telecon will now be every two weeks to allow its AMI counterpart to progress
- Loic to complete the installation and hand out to Neil and Geert Jan for the freeze. They are not expected to complete the review before the freeze.
- Loic will complete the documentation for the informal beta release deadline of Oct 15 2020
- Other features: vsin i as a step in specgen
- Come up with names for the different packages of our pipeline.

Sep 8 2020 14h

No meeting

Sep 3 2020 13h - Michael’s conversation
- Michael has a method to get trace model for first iteration with or without F277 observation
- Antoine has a demo code for iteratively updating the trace model
- Michael will lead the development

Sep 1 2020 14h

- Loic is installing all code on genesis and documenting
- almost done with simulation part
- Antoine is fixing the f_k instability to produce 1d spectra


Aug 25 14h

update:
Freeze on September 21
Pipeline release October 15
new server (genesis) should be online by the end of this week

actions:
Neil: Update the specgen branch to master on github
Loic: write down the input/output of each parts. There are currently 4 parts: simulation, noise, pipeline, extraction
Loic: Document the manual steps currently needed to run each part
Loic: Document the installation steps for the simulation and pipeline
Loic: Make sure that the reference files needed for the extraction exist
Geert Jan: install the detector noise component on meastria


Aug 18 - no meeting
Aug 11 - no meeting
Aug 4 - no meeting
July 28 - no meeting

July 21 14h

- My 1-D simulations of the Commissioning Time-Series
- Upgraded tracepol.py
- 1/f subtraction scheme
- Neil on the conda environment on maestria
- No news from the UdeM server
- No meeting next week (Loic on vacation)


July 14 14h

- Simulating Commissioning data requires ATLAS models
- Freeze of the tracepol.py code
- Simulation of F277W filter - ghost appeared, led to update of tracepol.py
- Antoine has now a function that returns the best sampling grid with os=2
- Bjorn proposed to use a batch schedule for the new SOSS server

July 7 - no meeting

June 30 14h00

- Michael presents his investigation on the trace profile changes with wavefront realization

June 23 14h00 - 14h45

- Discussed to complete flux normalization + little simulation details before a freeze
- Will ask Luc for a JWST accoutn on maestria and Neil will install python
- Jason can install his simulator there (requires the Intell compiler 1000$, current version is5-6 years old)
- William has issues with the blue end of the spectrum. Rp/Rs different from input value (due to limb darkening?)

June 16 2020, 14h-15h

- Recap of the CALWG meeting. It went well.
- Order 3 seems shorter than expected.
- Order 0 has wiggles
- Jason is able on the fly to produce a larger than 256x2048 to capture orders 0 and 1
- Michael has 40 free parameters to characterize the trace but still can not reproduce it with high fidelity
- Michael will look at the effect of varying wavefront on the trace profile
- Bjorn advocates using the CV3 trace but does not know how to fine tune it
- Geert Jan will fix the order 0 wiggles when we do a freeze of the code

June 9 2020 14h-15h

Minutes:
- Neil’s demo of github (https://docs.google.com/presentation/d/1S6oh-mDwJyZbHcZSSYj9OEmZlaOzuIbxJVhk0FT3ijQ)
- Jason about improvements in simulation
- Suggestion: Call Scarlet directly in the simulation code to get planet atmosphere models
- Suggestion: Use github tree structure when referencing files on disk

To do:
- (Jason) Ingest the new phoenix models
- (Loic) Doing the flux normalization
- (Neil) Document all the steps from A to Z (and put on github)
- (Loic et al.) Implement an optimal extraction function (including tilt if possible)
- (Loic) Add a step to generate PSFs directly in simulation code rather than use pre-generated PSFs

June 2 SOSS 14h-15
- Team discussion for letting STScI use the extraction, implications
- Neil’s demo of github
- Geert Jan’s recap of noise
- William will make python wrapper for specgen with Jason
- Michael Radica’s implication
- Skeleton of a paper for Extraction

minute: 
* make sure that we get credit and can publish a paper led by us. We are worried that people will confuse versions.
* we have two tools 1) the engine 2) the wrapper (or solver)
* Handed version would only have the engine and 1 reference trace
* Have one point of contact: Loic.
* x Sketch an email.

June 1st 2020, 9h30 morning
- Jason William and I about running simulations and making modifications
Username: loicalbert
Password: A*^c2GsN}gZP,2SK  (you can change it)
Nabucoco99!
Host:  kona.ubishops.ca 
ssh port: 5821
Jupyter Server:  https://kona.ubishops.ca:8000
- William will edit w2p, p2w, trace?
- Jason will update the specgen github branch
- reconvene next week same time 9h30

May 19 2020, 14-h15h

Round table:
- Geert Jan on noise
- Loic on spectro convolution kernel, models, synthetic magnitudes
- Jason on orders 0, 3 and negative
- William on differential extraction
Geert Jan on publications
Joseph Fillippazo and a meeting for extraction

actions for this week:
- Jason with order 0 ,-1, -2
- Geert Jan with noise on GJ436
- Antoine, Loic and Geert Jan on paper skeleton
- Loic to update github with his contribution

May 12 2020, 14h-15h

Antoine will present some more on extraction
Presenting master plan


May 5 2020, 14h-15h

Antoine update
Loic update
CASCA and posters
short term objective:
	- have a python wrapper for the simulation
	- include the normalization in the simulation
	- noise
	- STScI wrapper
Simulation paper - Talens

April 28 2020, 14h-15h

Talk about renewal of Compute Canada account
Making a plan with Geert Jan
Measure the fwhm of WebbPSF PSFs - Make a matrix of the spectral profile at all wavelengths to generate a convolution function.

April 21 2020, 14h-15h

Model atmosphere with specific intensities:
ftp://phoenix.astro.physik.uni-goettingen.de/

Demo of Loic’s normalization routine

Investigate if current models are full rezz. If not, request them to Peter Hauschild.

Next week - Antoine will present some more on extraction



April 14 2020, 14h-15h

wavelength, Flambda (W/m2/micron), filtername, magnitudevalue —> loic.py —> 
Run the new code by Jason



March 23 2020

32—> 16 bit
SVO filter — > send to Jason

scale Jason freely to any level

whenisgood:
http://whenisgood.net/8hbi7yw
http://whenisgood.net/8hbi7yw/results/ew24q47
http://whenisgood.net/8hbi7yw/edit/ew24q47
ew24q47


February 17 2020

Actions:
x genere des PSFs monochromatiques de 128x128 pour Jason
x Envoye les fonctions de Geert Jan a Jason
Avec Olivia, implemente l’extraction d’ouverture optimale
x Discute avec Caroline pour le bruit de Poisson
Upload les PSFs sur Compute Canada, comment?


February 10 2020

SOSS stellar var after simulation
configuration file for ngroups
update traces before extraction and loop back



January 27 2020

ask Kevin Stevenson if there is header keyword for time stamping.

Keep in mind to test for a non-linear thing that happens on your spectrum. What happens with that extraction method.





## Links to meeting minutes:

- [Minutes 2019-12-04](https://github.com/njcuk9999/jwst-mtl/edit/master/SOSS/meetings/minutes_20191204.md)

- [Minutes 2019-12-11](https://github.com/njcuk9999/jwst-mtl/edit/master/SOSS/meetings/minutes_20191211.md)

- [Minutes 2020-01-13](https://github.com/njcuk9999/jwst-mtl/edit/master/SOSS/meetings/minutes_20200113.md)


## Compute Canada

- [How to request and use Compute Canada Servers](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/meetings/ComputeCanada.md)
