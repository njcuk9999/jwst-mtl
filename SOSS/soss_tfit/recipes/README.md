# Recipes

Scripts for direct use by the user

## yamls

example.yaml - default settings by Jason
example_david_full.yaml - David settings (30 + 15 bins)
example_david.yaml - David settings (10 + 5 bins)
example_david_change_in_code.yaml - default settings (10 + 5 bins) to be run with transit_fit_example_david.py
example_neil.yaml - like example_david.yaml  but with 10 times less steps in MCMC

## codes

transit_fit_example.py - clean version all parameters from yamls
transit_fit_example_david.py - parameters from yaml but some customization (as in David's original notebook)
load_previous_example.fits - example of how to load pickle Sampler file after creation