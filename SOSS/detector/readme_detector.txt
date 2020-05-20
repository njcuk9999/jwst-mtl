** Adding detector output **

Option #1: doing it from the terminal

call: python detector.py path/to/Jason's/fits/file.fits 1

    will use the path specified in detector.py for the forward coefficients 
    in order to add in the nonlinearity
    
    the 1 stands for True (for adding in non-linearity)
    There will be more 0/1 switches in the future as we add in some more detector effects
    

Option #2: running it from the file (detector.py)

then you should modify the imaPath variable to point to Jason's fits file
at line ~35 and run the file 

OUTPUT: Same as Jason's fits file (saved to the same name with _nonlin appended to it),
with the data modified to account for (1) Poisson noise and (2) non-linearity effects (if set to True)

** Calculating forward coefficients for non-linearity simulation **

Option #1: doing it from the terminal

call: 
    python get_forward_coeffs.py path/to/CRDS/file.fits 
    or
    python get_forward_coeffs.py path/to/CRDS/file.fits 0 100000 100 4
    
    where:
        0 and 100000 are the limits in flux over which the values are calibrated
        100 is the number of points for the calibration of 1 pixel
        4 is the degree of the polynomial that I fit to get the forward coefficients
        
    If you use the 1st call (no additional arguments) these default values will be used.
    They are the ones with which the file XXX.npy has been calculated.
    
When you run it, it prints an estimate of the time it will take to do the calibration 
over the entire array (1.5h on my laptop for the default values, but technically unless
the CRDS files are updated we shouldn't need to run it again)

Option #2: running it from the file (get_forward_coeffs.py)

then, modify the path for the crds file in the crdsPath variable around line 126.

OUTPUT: npy file with the array of coefficients having a format ((poly_deg+1),ncols,nrows)
