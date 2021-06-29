## Code for the Empirical Trace Construction&trade; Module

The *empirical trace construction* module builds uncontaminated 2D trace profiles for the first and second diffraction orders for a NIRISS/SOSS
GR700XD/CLEAR observation. The uncontaminated profiles are entirely data driven, only employing models where absolutely necessary (e.g., in the overlap region on the detector), and as such retain a high level of fidelity to the original observations.

The uncontaminated profiles are intended to be used directly as input for the *extraction* module. However, the *empirical trace construction* module can also add an arbitrary amount of padding and oversampling to both the spatial and spectral axes, such that the reconstructed profiles can be used as reference file inputs to the *simple solver* module.

An outline of the major procedural steps are as follows:
1. Replace bad pixels and find trace centroids using the edgetrig method.
2. Build the first pass, first order profile:
    1. Get anchor profiles to interpolate the contaminated region.
    2. Interpolate the first order core in the contaminated region.
    3. Stitch interpolated model to the uncontaminated data.
    4. Reconstruct first order extended wing structure to remove the second and third order cores.
    5. Rescale the column normalized model to the flux level of the original data.
3. Build the first pass, second order profile:
    1. Remove the first order from the CLEAR exposure using the first pass model.
    2. Reconstruct oversubtracted wings using the first order wings at the appropriate wavelength.
    3. Reconstruct oversubtracted trace cores, again using the first order.
4. Refine the first order profile:
    1. Subtract the first pass second order model from the CLEAR exposure to yield an uncontaminated first order core.
    2. Reconstruct the extended wing structure as before.
5. Refine the second order profile:
    1. Repeat steps for the first pass, second order profile, using the refined first order.
6. If the results are to be used as reference files, add padding and oversampling to both axes.
7. Save the results to a fits file.

These steps are carried out by initializing an EmpiricalTrace object and calling its `build_empirical_trace()` method. Examples can be found in this [notebook](https://github.com/njcuk9999/jwst-mtl/blob/master/SOSS/extract/empirical_trace/empirical_trace.ipynb).
