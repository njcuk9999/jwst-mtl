# Tools for extraction

overlap.py includes the classes that are used to extract overlaping orders
See extract.pdf for more details on the technique.

## To extract

An extraction is done by using the class `TrpzOverlap`. All the reference images and the parameters for extraction are specified when initiating an extraction object.

An extraction can then be performed by calling the method `extract()`. The output will be the oversampled underlying flux `f_k`, so a flux that has a resolution higher than both orders on the detector. This is not the end result. What we want is the convolved flux at each order's resolution AND binned (or integrated) over a pixel grid. This is done using the `bin_to_pixel()` method.

Here is a pseudo code example
```python
# Init extraction with reference files for spatial profiles, wavelength 2D maps
# detector image, oversampling of the grid solution, threshold on the spatial profile,
# error estimate of each pixels and a bad pixel map.
extra = TrpzOverlap([psf_1, psf_2], [wv_1, wv_2], scidata=scidata, n_os=5,
                    thresh=1e-4, sig=sig, mask=bad_pix_mask)

# Extract the underlying flux (not the end result)
f_k = extra.extract()

# Bin to pixel grid
wv_center_bin = [None for i_ord in range(extra.n_ord)]
f_bin = [None for i_ord in range(extra.n_ord)]
for i_ord in range(extra.n_ord):
   wv_center_bin[i_ord], f_bin[i_ord] = extra.bin_to_pixel(f_k=f_k, i_ord=i_ord)
```
