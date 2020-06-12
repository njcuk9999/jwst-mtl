# Adding noise to Jason's simulations.

Due to complications with the filesizes of the bias, dark, flat, etc. They are currently not part of the repo.
Please contact Geert Jan for access to these files and place them in ```jwst-mtl/SOSS/detector/files```.
A more permanent solution is under investigation and will be added later.

1. From the terminal type:
```
python detector.py --help
```
this will provide a verbose description of the current noise generation interface.
To add noise to a simulation consisting of two files, using simple normalization:
 ```
python detector.py /path/to/file1.fits /path/to/file2.fits --normalize
```
Note: if the call is made to each file seperately the normalization won't be the same.

2. To achieve the same from within python do:
```
import detector
detector.add_noise(['/path/to/file1.fits', '/path/to/file2.fits'], normalize=True)
```

# Inverting StSci non-linearity polynomials.

1. From the the terminal type:
```
python get_forward_coeffs.py --help
```
this will provide a verbose description of the current linearity inversion interface.
To run this on a non-linearity file provided by StSci:
``` 
python get_forward_coeffs.py path/to/file.fits 
```

    