#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:37:26 2019

@author: asoulain
"""
import numpy as np
import os, pickle, xara, NRM_tools, sys
import matplotlib.pyplot as plt
from astropy.io import fits
from Python_tools import crop_center, crop_max, AllMyFields
from termcolor import cprint

def Construct_mask(mask_name, D = 8.):
    """
    """
    #--------------------------------------
    #  Compute Fourier sampling grid model
    #--------------------------------------
    
    dic_mask = NRM_tools.Saved_mask() #Load saved aperture coordinates informations
    
    try:
        X = np.array(dic_mask[mask_name])[:, 0] #Apertures x-coordinates relative to the center
    except KeyError:
        cprint('\nError: undefined mask configuration (%s)!'%mask_name, on_color='on_red')
        print('-> Supposed to be in :', list(dic_mask.keys()))
        return None, None, None
        
    Y = np.array(dic_mask[mask_name])[:, 1] #Apertures y-coordinates relative to the center
    tmp = np.ones(len(X)) # Aperture transmission (1 correspond to identical aperture size)
            
    model = np.array([X, Y, tmp]).transpose() #Used mask or xara discretes models
    
    datadir = './_maskdir/'
            
    if not os.path.exists(datadir):
        os.system('mkdir %s'%datadir)
    
    modelname = datadir + "%s_mask_coordinates.txt"%(mask_name)
    np.savetxt(modelname, model, fmt="%+.6e %+.6e %.2f") # save the model in a good format for xara
    
    return dic_mask[mask_name], modelname, model
    
def plot_xara_matrix(modelname, display = False):
    """
    """
    cal = xara.KPO(fname=modelname)

    if False:
        cal.kpi.plot_pupil_and_uv(cmap='jet', xymax = 4, figsize=(14,7))

    kpm = cal.kpi.KPM #Kernel phase matrix
    vac = cal.kpi.VAC[:,:2] #Apertures coordinates
    blm = cal.kpi.BLM #Baseline matrix
    uvc = cal.kpi.UVC #(u-v) coordinates : UVC = BLM * VAC
        
    if display:
        plt.figure(figsize = (4,6))
        plt.subplot(2,2,1)
        
        plt.title('KPM')
        plt.imshow(kpm, origin = 'upper')
        plt.xlabel('Number of BL')
        plt.ylabel('Number of CP')
        
        plt.subplot(2,2,2)
        plt.title('VAC')
        plt.xlabel('0 = u, 1 = v')
        plt.ylabel('Number of Aperture')
        plt.imshow(vac, origin = 'upper')
        
        plt.subplot(2,2,3)
        plt.title('BLM')
        plt.xlabel('Number of Aperture')
        plt.ylabel('Number of BL')
    
        plt.imshow(blm, origin = 'upper')
        
        plt.subplot(2,2,4)
        plt.title('UVC')
        
        plt.imshow(uvc, origin = 'upper')
        plt.xlabel('0 = u, 1 = v')
        plt.ylabel('Number of BL')
        plt.tight_layout()
    #    plt.subplots_adjust(top=0.959,
    #                        bottom=0.082,
    #                        left=0.127,
    #                        right=1.0,
    #                        hspace=0.329,
    #                        wspace=0.0)
    return None

def make_cpm(modelname, model, closing_triangle, n_cp, n_bl): 
    
    cal = xara.KPO(fname=modelname)

    blm = cal.kpi.BLM #Baseline matrix

    # Associates the BL vector to the uvc indices :
    # '01' mean the vector BL from aperture 0 to 1 (ap number corresponds to the mask files).
    # => The order is important to well close the triangle.

    uv_line = []
    for ind_bl in range(n_bl):
        uv_line.append(str(np.where(blm[ind_bl] == -1)[0][0])+str(np.where(blm[ind_bl] == 1)[0][0]))
        
    # Compute the Closure phase matrix (similar to Kernel phase one)
    cpm = np.zeros((n_cp, n_bl)) #closure phase matrix
    cons_CP_ind = [] #list containting information about aperture and order
    
    for ind_cp in range(n_cp):
        bl1 = str(closing_triangle[ind_cp][0])
        bl2 = str(closing_triangle[ind_cp][1])
        bl3 = str(closing_triangle[ind_cp][2])
        
        #Closing triangle
        bl12 = bl1+bl2
        bl23 = bl2+bl3
        bl31 = bl3+bl1
                
        try:
            ind1 = uv_line.index(bl12)
            s1   = 1
        except ValueError:
            ind1 = uv_line.index(bl12[::-1])
            s1   = -1
            
        try:
            ind2 = uv_line.index(bl23)
            s2 = 1
        except ValueError:
            ind2 = uv_line.index(bl23[::-1])
            s2 = -1
            
        try:
            ind3 = uv_line.index(bl31)
            s3 = 1
        except ValueError:
            ind3 = uv_line.index(bl31[::-1])
            s3 = -1
            
        cons_CP_ind.append([[ind1, s1], [ind2, s2], [ind3, s3]])
        cpm[ind_cp][ind1] = s1
        cpm[ind_cp][ind2] = s2
        cpm[ind_cp][ind3] = s3
        
    return cpm, cal, cons_CP_ind, uv_line

def norm(tab):
    return tab/np.max(tab)

def Extract_NRM_file(filename, cpm, modelname, pscale, wl, ISZ, r1, dr, n = None, skycorr = True, apod = True, icrop=512, rot=0):
    """
    
    """
    
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    
    cube = data['CUBE']#fits.getdata(filename)
    
    hdr = data['HDR']#fits.getheader(filename)
    
    time = data['TIME']
    
    exptime = hdr['EXPTIME']
    mjd = hdr['MJD-OBS']
    target = hdr['OBJECT']
    ins = hdr['INSTRUME']
    
    try:
        filt = hdr['HIERARCH ESO INS COMB IFLT']
    except:

        filt = ''
        
    info = {'TARGET' : target,
            'INS' : ins,
            'FILTER' : filt, 
            'MJD' : mjd, 
            'EXPTIME' : exptime,
            'HDR' : hdr}
        
    n_img = cube.shape[0]
    
    if type(n) == int:
        nmax = n
    else:
        nmax = n_img
        
    print('\nExtract Vis, CP and KP from %s file...'%filename)
    print('-> Total number of images = %i'%n_img)
    
    l_visamp, l_visphi, l_vis2 = [], [], []
    l_cpamp, l_cpphi = [], []
    #l_kpdata = []
    
    save_im = []
    all_bg = []
    for i in range(nmax):
        try:
            tmp = crop_center(cube[i], icrop)
            if skycorr:                  
                try:
                    img_biased, bg = NRM_tools.Sky_correction(crop_max(tmp, ISZ)[0], r1 = r1, dr = dr, display = False)
                    type_im = type(img_biased)
                    if type_im != np.ndarray:
                        pass
                    else:
                        if apod:
                            img = NRM_tools.Apply_mask_apod(img_biased, r = ISZ//3)#ISZ//3)
                        else:
                            img = img_biased.copy()
                except:
                    print('### Pb with crop size images.')
                    pass
            else:
                img = crop_max(tmp, ISZ)
                bg  = np.mean(img[0:10, 0:10])
            cal = xara.KPO(fname=modelname)
            
            from scipy.ndimage import rotate

            img = rotate(img, rot)
            
            
            from matplotlib.colors import PowerNorm
            
            if i == 0:
                img0 = img
            
            #--------------------------------------
            #   Extract KP and CVIS using xara
            #--------------------------------------
            
            cal.extract_KPD_single_frame(img, pscale, wl, target="", method = 'LDFT1', recenter=True) 
            
            save_im.append(img)
            all_bg.append(bg)
            
            cvis = np.array(cal.CVIS)[0][0]
            
            #kpdata = kpm.dot(np.angle(cvis)) #Compute kernel phases (size = n_cp)
            cpphi = cpm.dot(np.angle(cvis)) #Compute closure phases (size = n_cp)
            cpamp = cpm.dot(np.abs(cvis)) #Compute closure phases (size = n_cp)
            visamp = np.abs(cvis) #size = n_bl
            visphi = np.angle(cvis) #size = n_bl
            vis2 = np.abs(cvis)**2 #size = n_bl
            
            l_vis2.append(vis2)
            l_visamp.append(visamp)
            l_visphi.append(visphi)
            l_cpamp.append(cpamp)
                        
            l_cpphi.append(cpphi)
            #l_kpdata.append(kpdata)
            
            a = (i+1)/nmax
            sys.stdout.write("\r" + '### Progress : %2.1f %% ###'%(a*100))
            sys.stdout.flush()
                
        except:
            print('#Warnings : pb with file number %i'%i)
            pass
        
        if len(save_im) == 0:
            print('# Zero saved images : problem!')
            return np.nan        
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img, norm=PowerNorm(.5))
    plt.subplot(1,2,2)
    plt.imshow(abs(np.fft.fftshift(np.fft.fft2(img)))**2, norm=PowerNorm(.5), cmap = 'gist_stern')
        
    dic = {'vis2' : np.array(l_vis2),
           'visamp' : np.array(l_visamp),
           'visphi' : np.array(l_visphi),
           'cpamp' : np.array(l_cpamp),
           'cpphi' : np.array(l_cpphi),
           #'kpdata' : np.array(l_kpdata),
           'info' : info,
           'tabim' : save_im,
           'all_bg': all_bg,
           'time' : time
           }
    
    out = AllMyFields(dic)
    
    return out

def NRM_datacubeTodpy(filename, filesave=None, target=None, filt=None, mjd=None, ins=None, exptime=None, time=None, 
                      Tel='JWST', Obs='me', savedir='__DatadpyDir', verbose=False):
    """
    
    """
    hdu = fits.open(filename)

    hdr = hdu[0].header

    data = hdu[0].data

    hdr_new = hdr
    
    try:
        target = hdr['OBJECT']
    except KeyError:
        if verbose:
            print('### Warning: no OBJECT in metadata.')
        if target is None:
            if verbose:
                print('-> Add a target as argument.')
        else:
            hdr_new['OBJECT'] = target

    try:
        ins = hdr['INSTRUME']
    except KeyError:
        if verbose:
            print('### Warning: no INSTRUME in metadata.')
        if ins is None:
            if verbose:
                print('-> Add an ins as argument.')
        else:
            hdr_new['INSTRUME'] = ins
    
    try: 
        filt = hdr['HIERARCH ESO INS COMB IFLT']
    except KeyError:
        if verbose:
            print('### Warning: no FILTER in metadata.')
        if filt is None:
            if verbose:
                print('-> Add an filt as argument.')
        else:
            hdr_new['HIERARCH ESO INS COMB IFLT'] = filt
    try:
        exptime = hdr['EXPTIME']
    except KeyError:
        if verbose:
            print('### Warning: no EXPTIME in metadata.')
        if exptime is None:
            if verbose:
                print('-> Add an exptime as argument.')
        else:
            hdr_new['EXPTIME'] = exptime
            
    try:
        mjd = hdr['MJD-OBS']
    except KeyError:
        if verbose:
            print('### Warning: no MJD-OBS in metadata.')
        if mjd is None:
            if verbose: 
                print('-> Add a mjd as argument.')
        else:
            hdr_new['MJD-OBS'] = mjd
                        
    hdr_new['TELESCOP'] = Tel
    hdr_new['OBSERVER'] = Obs
    
    dicdpy = {'CUBE' : data,
          'HDR'  : hdr_new,
          'TIME' : time,
         }
    
    if verbose:
        print('\n#### datacube : %s'%filename.split('/')[-1])
        print('- Instrument: %s'%ins)
        print('- Target: %s'%target)
        print('- Filter: %s'%filt)
        print('- MJD: %s'%mjd)
    
    import os
    
    if not os.path.exists(savedir+'/'):
        os.system('mkdir %s'%savedir)
    
    try:
        if type(filesave) == str:
            filedpy = savedir + '/' + filesave
        else:
            filedpy = savedir + '/%s_%s_%s_mjd%i_NRMdic.dpy'%(ins, target, filt, mjd)
        
        file = open(filedpy, 'wb')
        pickle.dump(dicdpy, file)
        file.close()
    except TypeError:
        cprint('\nError: dpy file not saved => please give the good required arguments (target, filt, etc.)', 'cyan')
        return None
    return filedpy

