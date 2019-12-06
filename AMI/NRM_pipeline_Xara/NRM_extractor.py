#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:36:22 2019

@author: asoulain
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_palette(sns.color_palette("Set2", 13))
import NRM_tools
import NRM_fct
import pickle, os

plt.close('all')

def NRM_extractor(data, mask_name, pscale, wl, bandwidth, filt = None, rot_sci = 0, rot_cal =  0, fakeerr = False, e_rel = 0,
                  nmax = 3, ISZ = 128, D = 8., r1 = 100, dr = 20, icrop = 512, Skycorr = True, apod = True, Polyfit = True,
                  save = False, savedir='_NRMsavedpy', display = False):
    r"""
    Short Summary
    -------------
    
    Module to extract complex visibilities and closure phase from aperture masking data using xara package.
    
    Parameters.
    -----------
    
    filename: str
       fits file containing the NRM data of the science target. Data is a cube containting all individual frames
       to compute error and average V2 and CP.
    filename_cal: str
       fits file containing the NRM data of the calibrator. 
    mask_name: str
        Name of the mask used in the NRM data.
    pscale: float
        Size of the pixel [mas].
    wl: float
        Wavelength of observation [m].
    bandwidth: float
        Bandwidth of observation [m].
    nmax: int
        Number of frame used to compute NRM data (Vis, V2 and CP). Default is 3.
    ISZ: int
        Size of the cropped image.
    D: float
        Diameter of the telescope, default is 8 (VLT size).
    save: boolean
        If True, the final dictionnary is saved into a pickle format file.
    display: boolean
        If True, all figures are showed.()
    
    """

    #--------------------------------------
    #  Construct mask coordinates
    #--------------------------------------
    mask, modelname, model = NRM_fct.Construct_mask(mask_name, D = D)
    
    if mask is not None:
        pass
    else:
        return None

    #--------------------------------------
    #  Compute BL and CP possibibilties
    #--------------------------------------
    
    closing_triangle, diff_BL, n_cp, n_bl, n_ap = NRM_tools.Compute_CP_index(mask, origin = 0, display = display)

    NRM_fct.plot_xara_matrix(modelname, display = display)
    
    #--------------------------------------
    #  Compute Closure phase matrix
    #--------------------------------------

    cpm, cal, cons_CP_ind, uv_line =  NRM_fct.make_cpm(modelname, model, closing_triangle, n_cp, n_bl)

    if display:   
        plt.figure(figsize = (12,4))
        plt.subplot(1,2,1)
        plt.title('Kernel phase matrix')
        plt.imshow(cal.kpi.KPM, origin = 'upper', vmin = -1, vmax = 1)
        plt.colorbar()
        plt.xlabel('Number of BL')
        plt.ylabel('Number of CP')
        
        plt.subplot(1,2,2)
        plt.title('Closure phase matrix')
        plt.imshow(cpm, origin = 'upper', vmin = -1, vmax = 1)
        plt.colorbar()
        plt.xlabel('Number of BL')
        plt.tight_layout()
        
    uvc_vis = cal.kpi.UVC #size = n_bl
    
    unit = 'arcsec-1'
    if unit == 'm':
        freq_vis = ((uvc_vis[:, 0]**2 + uvc_vis[:, 1]**2)**0.5)
    else:
        freq_vis = ((uvc_vis[:, 0]**2 + uvc_vis[:, 1]**2)**0.5)/wl/206264.806247 #arcsec-1

    #--------------------------------------
    #  Compute triangle coordinates in image and Fourier space
    #--------------------------------------

    uvc_triangle = NRM_tools.Compute_triangle_mask(model, closing_triangle)
    uvc_cp = NRM_tools.Compute_triangle_fourier(uvc_vis, cons_CP_ind, closing_triangle) #size = n_cp
    
    if unit == 'm':
        freq_cp = NRM_tools.Compute_freqcp(uvc_cp, wl) * wl * 206264.806247 
    else:
        freq_cp = NRM_tools.Compute_freqcp(uvc_cp, wl)
    
    #--------------------------------------
    #  Extract CVIS and CP from cube
    #--------------------------------------
        
    filename = data['star']    
    filename_cal =  data['calib']
             
    nrm = NRM_fct.Extract_NRM_file(filename, cpm, modelname, pscale, wl, ISZ, r1, dr, n = nmax, skycorr = Skycorr, apod = apod, icrop=icrop, rot=rot_sci)
    
    if type(filename_cal) == str:
        nrm_c = NRM_fct.Extract_NRM_file(filename_cal, cpm, modelname, pscale, wl, ISZ, r1, dr, n = nmax, skycorr = Skycorr, apod = apod, rot=rot_cal)
    elif type(filename_cal) == list:
        nrm_c = NRM_fct.Extract_NRM_file(filename_cal[0], cpm, modelname, pscale, wl, ISZ, r1, dr, n = nmax, skycorr = Skycorr, apod = apod, rot=rot_cal)
        nrm_c2 = NRM_fct.Extract_NRM_file(filename_cal[1], cpm, modelname, pscale, wl, ISZ, r1, dr, n = nmax, skycorr = Skycorr, apod = apod, rot=rot_cal)
    else:
        print('### Warning : bad calibrators format.')
        return None
    
    if type(nrm) == float:
        if np.isnan(nrm):
            return None, None, None
        
    #--------------------------------------
    #  CVIS and CP calibration/uncertainties
    #--------------------------------------
    
    fact_calib_vis2 = np.mean(nrm_c.vis2, axis = 0) #calibration factor V2 (supposed to be one)
    fact_calib_visamp = np.mean(nrm_c.visamp, axis = 0) #calibration factor Vis. (supposed to be one)
    fact_calib_visphi = np.mean(nrm_c.visphi, axis = 0) #calibration factor Vis. (supposed to be zero)
        
    if Polyfit:
        if type(filename_cal) == str:
            vis2_calibrated =  NRM_tools.Calib_poly_vis(nrm, nrm_c, display = False)
        else:
            vis2_calibrated =  NRM_tools.Calib_poly_vis(nrm, [nrm_c, nrm_c2], display = False)
    else:
        vis2_calibrated = nrm.vis2/fact_calib_vis2
        
    visamp_calibrated = nrm.visamp/fact_calib_visamp
    visphi_calibrated = nrm.visphi - fact_calib_visphi
    
    vis2_med = np.median(vis2_calibrated, axis = 0) #V2
    vis2_mean = np.mean(vis2_calibrated, axis = 0) #V2
        
    e_vis2 = np.std(vis2_calibrated, axis = 0) #Error on V2
    
    visamp_mean = np.mean(visamp_calibrated, axis = 0) #Vis. amp
    e_visamp = np.std(visamp_calibrated, axis = 0) #Vis. amp
    
    visphi_mean = np.mean(visphi_calibrated, axis = 0) #Vis. amp
    e_visphi = np.std(visphi_calibrated, axis = 0) #Vis. amp
    
    if display:
        plt.figure()
        plt.title(r'Calibrated V$^2$')
        plt.plot(vis2_calibrated.transpose()[0][0], 'gray', alpha = .2, label = 'All')
        plt.plot(vis2_calibrated.transpose(), 'gray', alpha = .2)
        plt.plot(vis2_mean, 'k--', label = 'Mean')
        #plt.plot(vis2_mean2, 'c--', label = 'new')
        plt.plot(vis2_med, 'r--', label = 'Median')
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        plt.ylabel(r'V$^2$', color = 'dimgray', fontsize = 12)
        plt.ylim(0, 1.2)
        plt.legend(loc = 'best')
        plt.tight_layout()
        
        plt.figure(figsize = (8,5))
        plt.subplot(1,2,1)
        plt.title(r'Uncalibrated V$^2$')
        plt.plot(nrm_c.vis2.transpose()[0][0], 'gray', alpha = .2, label = 'All')
        plt.plot(nrm_c.vis2.transpose(), 'gray', alpha = .2)
        plt.plot(fact_calib_vis2, 'r--', label = 'CALIB')
        plt.ylabel(r'V$^2$', color = 'dimgray', fontsize = 12)
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        plt.legend(loc = 'best')
        plt.subplot(1,2,2)
        plt.plot(nrm.vis2.transpose()[0][0], 'g', alpha = .2, label = 'All')
        plt.plot(nrm.vis2.transpose(), 'g', alpha = .2)
        plt.plot(np.mean(nrm.vis2, axis = 0), 'b--', label = 'SCI')
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        #plt.ylim(0, 2.2)
        plt.legend(loc = 'best')
        plt.tight_layout()
    
    fact_calib_cp = np.mean(nrm_c.cpphi, axis = 0) #calibration factor CP (supposed to be zero)
    fact_calib_cpamp = np.mean(nrm_c.cpamp, axis = 0)
    
    shift2pi = np.zeros(nrm.cpphi.shape)
    shift2pi[nrm.cpphi >= 6] = 2*np.pi
        
    nrm.cpphi = nrm.cpphi - shift2pi
    
    cp_cal = nrm.cpphi - fact_calib_cp
    
    cpamp_cal = nrm.cpamp/fact_calib_cpamp
    
    cp_mean = np.mean(cp_cal, axis = 0)
    cp_med = np.median(cp_cal, axis = 0)
    e_cp = np.std(cp_cal, axis = 0)
    
    cpamp_mean = np.mean(cpamp_cal, axis = 0)
    e_cpamp = np.std(cpamp_cal, axis = 0)
    
    aa = cp_cal.transpose()[0][0]
    
    if display:
        plt.figure()
        plt.title('Calibrated CP')
        plt.plot(aa, 'gray', alpha = .2, label = 'All')
        plt.plot(cp_cal.transpose(), 'gray', alpha = .2)
        plt.plot(cp_mean, 'k--', label = 'Mean')
        plt.plot(cp_med, 'r--', label = 'Median')
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        plt.ylabel(r'CP [rad]', color = 'dimgray', fontsize = 12)
        plt.ylim([-2.2*np.pi, 2.2*np.pi])
        plt.legend(loc = 'best')
        plt.tight_layout()
        
        plt.figure(figsize = (8,5))
        plt.subplot(1,2,1)
        plt.title('Sci CP')
        plt.plot(nrm.cpphi.transpose()[0][0], 'gray', alpha = .2, label = 'All')
        plt.plot(nrm.cpphi.transpose(), 'gray', alpha = .2)
        plt.plot(np.mean(nrm.cpphi, axis = 0), 'r--', label = 'Mean')
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        plt.ylabel(r'CP [rad]', color = 'dimgray', fontsize = 12)
        plt.ylim([-2.2*np.pi, 2.2*np.pi])
        plt.subplot(1,2,2)
        plt.title('Calibrator CP')
        plt.plot(nrm_c.cpphi.transpose()[0][0], 'gray', alpha = .2, label = 'All')
        plt.plot(nrm_c.cpphi.transpose(), 'gray', alpha = .2)
        plt.plot(np.mean(nrm_c.cpphi, axis = 0), 'r--', label = 'Mean')
        plt.xlabel('Index', color = 'dimgray', fontsize = 12)
        plt.ylabel(r'CP [rad]', color = 'dimgray', fontsize = 12)
        plt.ylim([-2.2*np.pi, 2.2*np.pi])
    U1COORD = uvc_cp[:, 0, 0]
    V1COORD = uvc_cp[:, 0, 1]
    U2COORD = uvc_cp[:, 1, 0]
    V2COORD = uvc_cp[:, 1, 1]
    
    #--------------------------------------
    #  Check flagged data
    #--------------------------------------
    rel_e_vis = abs(e_vis2/vis2_mean)
    rel_e_visamp = abs(e_visamp/visamp_mean)
    
    cond_vis = rel_e_vis <= 3
    cond_flag_vis = rel_e_vis > 3
    cond_flag_visamp = rel_e_visamp > 3
    
    e_cp = e_cp
    rel_e_cp = abs(e_cp/cp_mean)
    
    maxi = np.max([np.max([rel_e_cp.max(), rel_e_vis.max()]), 4])
    
    cond_cp = rel_e_cp <= 3
    cond_flag_cp = rel_e_cp > 3

    if display:
        plt.figure(figsize = (6,8))
        plt.subplot(2,1,1)
        plt.plot(freq_vis[cond_vis], rel_e_vis[cond_vis], 'o', color = 'cornflowerblue', label = 'V2')
        plt.plot(freq_vis[cond_flag_vis], rel_e_vis[cond_flag_vis], 'o', color = 'crimson', label = 'Flagged')
        plt.legend(loc = 'best')
        plt.xlabel('Sp. Freq [%s]'%unit, color = 'dimgray', fontsize = 12)
        plt.ylabel(r'Rel. uncertainties [$\sigma$]', color = 'dimgray', fontsize = 12)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 0, 1, color = 'g', alpha = .1)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 1, 2, color = 'y', alpha = .1)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 2, 3, color = 'gray', alpha = .1)
        plt.ylim(0,maxi)
        plt.xlim(freq_vis.min()-1,freq_vis.max()+1)
        plt.subplot(2,1,2)
        plt.plot(freq_cp[cond_cp], rel_e_cp[cond_cp], 'o', color = 'cornflowerblue', label = 'CP')
        plt.plot(freq_cp[cond_flag_cp], rel_e_cp[cond_flag_cp], 'o', color = 'crimson', label = 'Flagged')
        plt.legend(loc = 'best')
        plt.xlabel('Sp. Freq [m]', color = 'dimgray', fontsize = 12)
        plt.ylabel(r'Rel. uncertainties [$\sigma$]', color = 'dimgray', fontsize = 12)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 0, 1, color = 'g', alpha = .1)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 1, 2, color = 'y', alpha = .1)
        plt.fill_between([freq_vis.min()-1, freq_vis.max()+1], 2, 3, color = 'gray', alpha = .1)
        plt.ylim(0,maxi + 0.5)
        plt.xlim(freq_vis.min()-1,freq_vis.max()+1)
        
        plt.tight_layout()

        plt.figure(figsize = (14.2,7))
        plt.subplot(1,2,1)
        for i in range(model.shape[0]):
            plt.scatter(model[i][0], model[i][1], s = 1e2, c = '', edgecolors = 'navy')
            plt.text(model[i][0]+0.1, model[i][1]+0.1, i)
            
        NRM_tools.plot_triangle_mask(uvc_triangle, closing_triangle)
            
        plt.xlabel('Aperture x-coordinate [m]')
        plt.ylabel('Aperture y-coordinate [m]')
        plt.legend(fontsize = 8)
        plt.axis([-D/2., D/2., -D/2., D/2.])
        
        plt.subplot(1,2,2)
        for i in range(n_bl):
            plt.scatter(uvc_vis[i,0], uvc_vis[i,1], s = 1e2, c = '', edgecolors = 'navy')
            plt.scatter(-uvc_vis[i,0], -uvc_vis[i,1], s = 1e2, c = '', edgecolors = 'crimson')
            
        NRM_tools.plot_triangle_fourier(uvc_vis, cons_CP_ind, closing_triangle)
            
        plt.plot(0, 0, 'k+')
        plt.axis((-D, D, -D, D))
        plt.xlabel('Fourier u-coordinate [m]')
        plt.ylabel('Fourier v-coordinate [m]')
        plt.tight_layout()
        
        plt.figure(figsize = (12,6))
        plt.subplot(2,1,1)
        plt.title('star = %s, calibrator = %s, filt = %s, mask = %s'%(nrm.info.TARGET, nrm_c.info.TARGET, nrm.info.FILTER, mask_name))
        plt.ylabel(r'V$^2$', fontsize = 12)
        plt.grid(alpha = .2)
        plt.errorbar(freq_vis, vis2_mean, yerr = e_vis2, fmt='.', color='cornflowerblue',
                     ecolor='lightgray', elinewidth=1.5, capsize=0)
        plt.axis([0, freq_vis.max()+1, 1e-3, 1.3])
        plt.yscale('log')
        plt.subplot(2,1,2)
        plt.grid(alpha = .2)
        plt.xlabel('Sp. Freq [%s]'%unit, fontsize = 12)
        plt.ylabel('CP [rad]', fontsize = 12)
        
        plt.errorbar(freq_cp, cp_mean, yerr = e_cp, fmt='.', color='cornflowerblue',
                     ecolor='lightgray', elinewidth=1.5, capsize=0)
        plt.axis([0, freq_vis.max()+1, -2.2*np.pi, 2.2*np.pi])
        plt.subplots_adjust(top=0.95,
        bottom=0.08,
        left=0.05,
        right=0.99,
        hspace=0.035,
        wspace=0.185)

    True_flagT3 = False
    True_flagV2 = True
    
    if True_flagV2:
        flagV2 = cond_flag_visamp
    else:
        flagV2 = [False] * len(cond_flag_visamp)
        
    if True_flagT3:
        flagT3 = cond_flag_visamp
    else:
        flagT3 = [False] * len(cond_flag_cp)
    
    if fakeerr:
        e_vis2 = e_rel * vis2_mean
        e_cp = e_rel * cp_mean
        
    dic = {'OI_VIS2' : {'VIS2DATA' : vis2_mean,
                       'VIS2ERR' : e_vis2,
                       'UCOORD' : uvc_vis[:, 0],
                       'VCOORD' : uvc_vis[:, 1],
                       'STA_INDEX' : uv_line,
                       'MJD' : nrm.info.MJD,
                       'INT_TIME' : nrm.info.EXPTIME,
                       'TIME' : 0, 
                       'TARGET_ID' : 1,
                       'FLAG' : flagV2,
                       'FREQ' : freq_vis},
           'OI_VIS' : {'TARGET_ID' : 1,
                       'TIME' : 0, 
                       'MJD' : nrm.info.MJD,
                       'INT_TIME' : nrm.info.EXPTIME,
                       'VISAMP' : visamp_mean,
                       'VISAMPERR' : e_visamp,
                       'VISPHI' : visphi_mean,
                       'VISPHIERR' : e_visphi,
                       'UCOORD' : uvc_vis[:, 0],
                       'VCOORD' : uvc_vis[:, 1],
                       'STA_INDEX' : uv_line,
                       'FLAG' : flagV2
                      }, 
           'OI_T3' : {'MJD' : nrm.info.MJD,
                      'INT_TIME' : nrm.info.EXPTIME,
                      'T3PHI' : cp_mean,
                      'T3PHIERR' : e_cp,
                      'T3AMP' : cpamp_mean,
                      'T3AMPERR' : e_cpamp,
                      'U1COORD' : U1COORD,
                      'V1COORD' : V1COORD,
                      'U2COORD' : U2COORD,
                      'V2COORD' : V2COORD,
                      'STA_INDEX' : closing_triangle,
                      'FLAG' : flagT3,
                      'FREQ' : freq_cp
                      },
          'OI_WAVELENGTH' : {'EFF_WAVE' : wl,
                           'EFF_BAND' : bandwidth},
          'info' : {'TARGET' : nrm.info.TARGET,
                    'CALIB' : nrm_c.info.TARGET,
                    'FILT' : filt,
                    'INS' : nrm.info.INS,
                    'MASK' : mask_name,
                    'MJD' : nrm.info.MJD,
                    'HDR' : nrm.info.HDR,
                    'ISZ' : ISZ,
                    'PSCALE' : pscale,
                    'NFILE' : 0}        
                    }
          
    if not os.path.exists(savedir+'/'):
        os.system('mkdir %s'%savedir)
    
    filename = savedir + '/NRM_%s_%s_%s_%s.dpy'%(nrm.info.TARGET, nrm.info.INS, nrm.info.FILTER, mask_name)

    if save:
        NRM_tools.NRMtoOifits2(dic, verbose = False)
        
        file = open(filename, 'wb')
        pickle.dump(dic, file)
        file.close()
        
    return dic, nrm, nrm_c
       