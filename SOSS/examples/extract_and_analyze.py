import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

#from jwst.extract_1d import Extract1dStep

from jwst.pipeline import calwebb_spec2

import os

if False:
    rateints_file = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/IDTSOSS_clear_noisy_rateints.fits'
    spectrum_file = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/extracted_spectra.fits'
    outdir = '/genesis/jwst/userland-soss/loic_review/GTO/wasp52b/'
    model_file = '/genesis/jwst/userland-soss/loic_review/test_modeloutput.fits'

if False:
    rateints_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/IDTSOSS_clear_noisy_rateints.fits'
    spectrum_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/IDTSOSS_clear_extracted_spectra.fits'
    outdir = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/'
    model_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/modeloutput.fits'

if True:
    rateints_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/IDTSOSS_clear_noisy_rateints.fits'
    spectrum_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/IDTSOSS_clear_extracted_spectra.fits'
    outdir = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/'
    model_file = '/genesis/jwst/jwst-user-soss/loic_review/CAP_rehearsal/bd601753/modeloutput.fits'

bypass_stage2 = True
bypass_extract1d = True

if os.path.isfile(spectrum_file) is False:
    if bypass_stage2 is False:
        result = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rateints_file,
                               output_dir=outdir, output_file='stage2', save_results=True)
        #result = calwebb_spec2.background_step.BackgroundStep.call(result,
        #                       output_dir=outdir, output_file='stage2', save_results=True)
        result = calwebb_spec2.flat_field_step.FlatFieldStep.call(result,
                               output_dir=outdir, output_file='stage2', save_results=True)
        result = calwebb_spec2.srctype_step.SourceTypeStep.call(result,
                               output_dir=outdir, output_file='stage2', save_results=True)
        #result = calwebb_spec2.photom_step.PhotomStep.call(result,
        #                       output_dir=outdir, output_file='stage2', save_results=True)
        #TODO: handle unmasked invalid pixels in scierr after flat fielding
        '''
        2022-02-02 16:39:39,553 - stpipe.Extract1dStep - INFO - 19/20
        2022-02-02 16:39:39,553 - stpipe.Extract1dStep - INFO - 20/20
        2022-02-02 16:39:39,567 - stpipe.Extract1dStep - INFO - Using a Tikhonov factor of 9.006916883825219e-21
        2022-02-02 16:39:41,694 - stpipe.Extract1dStep - INFO - Optimal solution has a log-likelihood of -10907780.243589
        2022-02-02 16:39:43,436 - stpipe.Extract1dStep - INFO - Performing the decontaminated box extraction.
        2022-02-02 16:39:43,478 - stpipe.Extract1dStep - CRITICAL - scierr contains un-masked invalid values.
        Traceback (most recent call last):
        File "/genesis/jwst/github/jwst-mtl/SOSS/examples/extract_and_analyze.py", line 42, in <module>
        result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result,
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/stpipe/step.py", line 609, in call
        return instance.run(*args)
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/stpipe/step.py", line 430, in run
        step_result = self.process(*args)
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/jwst/extract_1d/extract_1d_step.py", line 358, in process
        result, ref_outputs = soss_extract.run_extract1d(
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/jwst/extract_1d/soss_extract/soss_extract.py", line 731, in run_extract1d
        result = extract_image(scidata_bkg, scierr, scimask, tracemodels, ref_files, transform, subarray, **kwargs)
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/jwst/extract_1d/soss_extract/soss_extract.py", line 467, in extract_image
        out = box_extract(decont, scierr_ord, scimask_ord, box_w_ord, cols=xtrace)
        File "/genesis/jwst/bin/miniconda3/envs/loic_test/lib/python3.8/site-packages/jwst/extract_1d/soss_extract/soss_boxextract.py", line 110, in box_extract
        raise ValueError(message)
        ValueError: scierr contains un-masked invalid values.
    
        Process finished with exit code 1
    '''
        if bypass_extract1d is False:
            result = calwebb_spec2.extract_1d_step.Extract1dStep.call(result,
                               output_dir=outdir, output_file='stage2', save_results=True,
                               soss_transform=[0,0,0], soss_atoca=True, soss_modelname=model_file)
    else:
        result = calwebb_spec2.extract_1d_step.Extract1dStep.call(rateints_file,
                                                                  soss_transform=[0,0,0],
                                                                  soss_atoca=True,
                                                                  soss_modelname=model_file,
                                                                  output_dir=outdir,
                                                                  output_file='stage2',
                                                                  save_results=True)

    result.write(spectrum_file)

data = fits.open(spectrum_file)
# spectra are stored at indice 1 (order 1), then 2 (order2) then 3 (order 3) then 4 (order 1, 2nd time step), ...
nint = data[0].header['NINTS']
norder = 3

wavelength = np.zeros((nint, norder, 2048))
flux = np.zeros((nint, norder, 2048))
order = np.zeros((nint, norder, 2048))
integ = np.zeros((nint, norder, 2048))

for ext in range(np.size(data)-2):
    m = data[ext+1].header['SPORDER']
    i = data[ext+1].header['INT_NUM']
    wavelength[i-1, m-1, :] = data[ext+1].data['WAVELENGTH']
    flux[i-1, m-1, :] = data[ext+1].data['FLUX']
    #print(m, i, np.size(w))

# Normalize each wavelength
flux = flux / np.nanmedian(flux, axis=0)

hdu = fits.PrimaryHDU(flux[:,0,:])
hdu.writeto(flux2d_file, overwrite=True)

plt.figure()
for i in range(nint):
    plt.plot(wavelength[i,0], flux[i,0]+0.02*i)
plt.show()

