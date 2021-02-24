## Code for the Simple Solver&trade;

The *simple solver* calculates and preforms the transformations (rotation and offset) necessary to match the 2D trace profile and wavelength map reference files to an input data frame. The intention is to make use of this *simple solver* to provide inputs to the extraction algorithm in instances where the *empirical trace construction* is not being used, or potentially in cases without F277W exposures.

An outline of the major procedural steps are as follows:
1. Determine the correct subarray for the input data.
2. Get the first order centroids for the reference 2D trace profile.
3. Extract the first order centroids for the input data using a center of mass analysis.
4. Fit the reference centroids to the extracted centroids to determine the correct rotation angle and offset.
5. Apply this transformation to the reference 2D trace profiles and wavelength maps for both the first and second orders.
6. Save the transformed reference files to disk.

These steps are carried out by calling the `simple_solver()` function. Examples can be found in this [notebook](https://github.com/njcuk9999/jwst-mtl/tree/master/SOSS/extract/simple_solver/simple_solver.ipynb).
