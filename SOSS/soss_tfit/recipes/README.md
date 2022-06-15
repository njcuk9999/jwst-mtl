# Recipes

Scripts for direct use by the user

## yamls

example_jason.yaml - default settings by Jason - used with transit_fit_example_david.py (this is to show how one can force parameters in the code) - NOT RECOMMENDED
example_david.yaml - larger simulation by David settings (30 + 15 bins) - this is a good start point for a real run
example_neil.yaml - like example_david.yaml  but with 10 times less steps in MCMC and less binning - this is a good test case

## codes

transit_fit_example.py - clean version all parameters from yamls - This is the recommended example 
transit_fit_example_david.py - parameters from yaml but some customization (as in David's original notebook) using example_jason.yaml 
load_previous_example.fits - example of how to load pickle Sampler file after creation