# JWST Pipeline information

## General links:
The JWST pipeline is divided into 3 stages of increasingly complex data manipulation. Each of the stages performs several
reductions steps as needed for the instrument and observing mode. The JWST pipeline is documented on Jdocs and Read-the-Docs.
The descriptions on Jdocs are more abstract (like what you'd find in a paper), while Read-the-Docs documents the python code.
Currently Jdocs appears to contain the more up-to-date and readable description of the pipeline, including clear diagrams 
showing which steps are taken during each stage and for each mode as well as brief descriptions of these steps. Read-the-Docs
contains comparable information on the pipeline stages but lacks the clears diagrams and descriptions of inidividual steps during each stage. 

[Jdocs](https://jwst-docs.stsci.edu/jwst-data-reduction-pipeline/algorithm-documentation/stages-of-processing)
[Read-the-Docs](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/main.html)

## Stage 1: Ramps to Slopes.
Will be used by us.

Used for all modes.

[Jdocs](https://jwst-docs.stsci.edu/jwst-data-reduction-pipeline/algorithm-documentation/stages-of-processing/calwebb_detector1)  
[Read-the-Docs](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html#calwebb-detector1)

## Stage 2: Calibration of Slopes.
Contains steps that are needed (flat-fielding, background subtraction, etc.) but we will likely write our own code for these steps.

Two branches of Stage 2 exist for imaging and spectroscopic observations. For NIRISS SOSS observations the spectroscopic branch of this stage will be used.

[Jdocs](https://jwst-docs.stsci.edu/jwst-data-reduction-pipeline/algorithm-documentation/stages-of-processing/calwebb_spec2)  
[Read-the-Docs](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html#calwebb-spec2)

## Stage 3: Process Ensembles of Slopes.
Not used by us.

Several branches of Stage 3 exist for for imaging, coronography, ami, spectroscopy and tso observations. For NIRISS SOSS either the spectroscopic branch or the tso branch will be used.

[Jdocs spec](https://jwst-docs.stsci.edu/jwst-data-reduction-pipeline/algorithm-documentation/stages-of-processing/calwebb_spec3)  
[Read-the-Docs spec](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec3.html#calwebb-spec3)

[Jdocs tso](https://jwst-docs.stsci.edu/jwst-data-reduction-pipeline/algorithm-documentation/stages-of-processing/calwebb_tso3)  
[Read-the-Docs tso](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_tso3.html#calwebb-tso3)


## Header information

Currently the most-useful information is at this link to the JWST Keyword Dictionary, which documents details on each FITS header keyword:
https://mast.stsci.edu/portal/Mashup/Clients/jwkeywords/index.html
