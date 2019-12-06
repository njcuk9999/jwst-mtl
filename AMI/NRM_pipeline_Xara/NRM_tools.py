#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:12:24 2019

@author: asoulain
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.io import fits
import datetime, os
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from termcolor import cprint
from Python_tools import gauss_2d_asym, conv_fft, crop_max, crop_center
from scipy.ndimage import rotate

def Saved_mask(upload = False):
    dic_mask = {'7holes' : [[-1.46, 2.87],
                       [1.46, 2.87],
                       [2.92, -1.35],
                       [0, -3.04],
                       [-2.92, -1.35],                  
                       [-2.92, 0.34],
                       [-1.46, -0.51]
                       ],
            '7holes_naco' : np.array([[-3.51064, -1.99373],
                                [-3.51064, 2.49014],
                                [-1.56907, 1.36918],
                                [-1.56907, 3.61111],
                                [0.372507, -4.23566],
                                [2.31408, 3.61111],
                                [4.25565, 0.248215]
                                ]) * (8/10.),
            '7holes_jwst' : np.array([[0, -2.64],
                                [2.28631, 0],
                                [-2.28631, -1.32],
                                [2.28631, 1.32],
                                [1.14315, 1.98],
                                [-2.28631, 1.32],
                                [-1.14315, 1.98]
                                ]),
            '9holes' : np.array([[3.50441, -2.60135],
                       [3.50441, 2.60135],
                       [2.00252 ,-1.73423],
                       [0.500629, -4.33558],
                       [0.500631, 2.60135],
                       [0.500631, 4.33558],
                       [-2.50315, -0.867115],
                       [-4.00503, -1.73423],
                       [-4.00503, 1.73423]
                       ]) * (8/10.),
            '3holes' : [[2, 0],
                       [-1, 0],
                       [0, 1]
                       ],
            '4holes' : [[1, 0],
                       [0, 1],
                       [-1, 0],
                       [3, -1]
                       ],
            'golay3' : [[-0.5, -np.sqrt(3)/6.],
                        [0.5, -np.sqrt(3)/6.],
                        [0, np.sqrt(3)/3]],
            'golay9' : [[0, -np.sqrt(3)],
                        [-2, -np.sqrt(3)],
                        [-0.5, -3*np.sqrt(3)/2],
                        [-4, -2*np.sqrt(3)],
                        [-5, -np.sqrt(3)],
                        [5/2., -np.sqrt(3)/2],
                        [5/2., np.sqrt(3)/2.],
                        [3/2., np.sqrt(3)/2.],
                        [-1/2., 3*np.sqrt(3)/2.],
                        ]
            }
                
    return dic_mask

def Compute_CP_index(model, origin = 1, display = False):
    
    model = np.array(model)
    
    n_holes = model.shape[0]
    n_BL = int((n_holes - 1)*(n_holes)/2)
    n_CP = int((n_holes - 1)*(n_holes - 2)/2)
    print('N_cp = %s, n_bl = %s for the %s holes mask.'%(n_CP, n_BL, n_holes))
    
    a = np.arange(n_holes).astype(str)
    b = np.arange(n_holes).astype(str)
    c = np.arange(n_holes).astype(str)
    
    closing_triangle_set = []
    closing_triangle = []
    for i in range(len(a)-2):
        for j in range(len(b)-1):
             for k in range(len(c)):
                 m = a[i]+b[j]+c[k]
                 nn = len(set(np.array([m[0], m[1], m[2]])))
                 if (nn == 3) or (nn == 4):
                     if str(origin) in m[0]:
                         m_set = list(sorted(np.array([m[0], m[1], m[2]])))
                         m_o = m_set[0]+m_set[1]+m_set[2]
                         if (m_o in closing_triangle) or (m_o in closing_triangle_set):
                             pass
                         else:
                             closing_triangle.append(a[i]+b[j]+c[k])
                             closing_triangle_set.append(m_o)
                                         
    diff_BL = []                
    for i in range(len(a)-1):
        for j in range(len(b)):
            m = a[i]+b[j]
            nn = len(set(np.array([m[0], m[1]])))
            if nn == 2:
                m_set = list(sorted(np.array([m[0], m[1]])))
                m_o = m_set[0]+m_set[1]
                if m_o in diff_BL:
                    pass
                else:
                    diff_BL.append(a[i]+b[j])
                        
    if display:
        plt.figure(figsize = (6.5,6))
        for i in range(model.shape[0]):
            plt.scatter(model[i][0], model[i][1], s = 1e2, c = '', edgecolors = 'b')
            plt.text(model[i][0]+0.1, model[i][1]+0.1, i)
            
        for i in range(n_CP):
            ind1 = int(closing_triangle[i][0])
            ind2 = int(closing_triangle[i][1])
            ind3 = int(closing_triangle[i][2])
            
            u1 = model[ind1][0]
            v1 = model[ind1][1]
            u2 = model[ind2][0]
            v2 = model[ind2][1]
            u3 = model[ind3][0]
            v3 = model[ind3][1]
               
            lu = [u1, u2, u3, u1]
            lv = [v1, v2, v3, v1]
            plt.plot(u1, v1, 'r+')
            plt.plot(lu, lv, '-', linewidth = 1 + (i/4.), alpha = 1/(i+2), label = '%i%i%i'%(ind1, ind2, ind3))
            
        plt.legend(fontsize = 8)
        plt.axis([-4, 4, -4, 4])
        plt.tight_layout()
    return closing_triangle, diff_BL, n_CP, n_BL, n_holes

def plot_triangle_mask(uvc_triangle, closing_triangle):
    
    n_cp = len(uvc_triangle)
    
    for i in range(n_cp):
        
        lu = uvc_triangle[i][0]
        lv = uvc_triangle[i][1]
        
        u1 = lu[0]
        v1 = lv[0]
        
        plt.plot(u1, v1, 'r+')
        plt.plot(lu, lv, '-', linewidth = 1 + (i/4.), alpha = 1/(i+2), label = closing_triangle[i])
                
    return None

def Compute_triangle_mask(model, closing_triangle):
    
    n_cp = len(closing_triangle)
    
    uvc_triangle = []
    for i in range(n_cp):
        ind1 = int(closing_triangle[i][0])
        ind2 = int(closing_triangle[i][1])
        ind3 = int(closing_triangle[i][2])
        
        u1 = model[ind1][0]
        v1 = model[ind1][1]
        u2 = model[ind2][0]
        v2 = model[ind2][1]
        u3 = model[ind3][0]
        v3 = model[ind3][1]
           
        lu = [u1, u2, u3, u1]
        lv = [v1, v2, v3, v1]
                
        uvc_triangle.append([lu, lv])
        
    return uvc_triangle
        
        
def plot_triangle_fourier(uvc, cons_CP_ind, closing_triangle):
    
    n_cp = len(closing_triangle)
    allu = []
    for i in range(n_cp):
        ind1 = cons_CP_ind[i][0][0]
        ind2 = cons_CP_ind[i][1][0]
        ind3 = cons_CP_ind[i][2][0]
        
        sig1 = cons_CP_ind[i][0][1]
        sig2 = cons_CP_ind[i][1][1]
        sig3 = cons_CP_ind[i][2][1]
        
        u1 = sig1 * uvc[ind1, 0]
        v1 = sig1 * uvc[ind1, 1]
        u2 = sig2 * uvc[ind2, 0]
        v2 = sig2 * uvc[ind2, 1]
        u3 = sig3 * uvc[ind3, 0]
        v3 = sig3 * uvc[ind3, 1]
        
        b1 = round((u1**2 + v1**2)**.5, 1)
        b2 = round((u2**2 + v2**2)**.5, 1)
        b3 = round((u3**2 + v3**2)**.5, 1)

        allu.append(b1)
        allu.append(b2)
        allu.append(b3)
                
        lu = [u1, u2, u3, u1]
        lv = [v1, v2, v3, v1]
        plt.plot(lu, lv, '-', linewidth = 1 + (i/4.), alpha = 1/(i+2), label = closing_triangle[i])

    diff = list(sorted(set(np.abs(allu))))
    
    theta = np.linspace(0, 2*np.pi, 100)

    for b in diff:
        x = b*np.cos(theta)
        y = b*np.sin(theta)
        plt.plot(x, y, 'k--', alpha = .3)
    
    plt.text(-7, 7, 'N effective bl = %i'%len(diff))
    return None

def Compute_triangle_fourier(uvc, cons_CP_ind, closing_triangle):
    
    n_cp = len(closing_triangle)
    uvcp = []
    for i in range(n_cp):
        ind1 = cons_CP_ind[i][0][0]
        ind2 = cons_CP_ind[i][1][0]
        ind3 = cons_CP_ind[i][2][0]
        
        sig1 = cons_CP_ind[i][0][1]
        sig2 = cons_CP_ind[i][1][1]
        sig3 = cons_CP_ind[i][2][1]
        
        u1 = sig1 * uvc[ind1, 0]
        v1 = sig1 * uvc[ind1, 1]
        u2 = sig2 * uvc[ind2, 0]
        v2 = sig2 * uvc[ind2, 1]
        u3 = sig3 * uvc[ind3, 0]
        v3 = sig3 * uvc[ind3, 1]
        
        uvcp.append([[u1, v1], [u2, v2], [u3, v3]])
    return np.array(uvcp)

def Compute_freqcp(uvc_cp, wl):
    n_ok = 0
    n_cp = len(uvc_cp)
    freq_cp = []
    for uv in uvc_cp:
        p1, p2, p3 = uv[0], uv[1], uv[2]
        u1coord, v1coord = p1[0], p1[1]
        u2coord, v2coord = p2[0], p2[1]
        u3coord, v3coord = p3[0], p3[1]
        
        u3coord_check = round(-u1coord-u2coord, 3)
        v3coord_check = round(-v1coord-v2coord, 3)
            
        if (u3coord_check == round(u3coord, 3)) & (v3coord_check == round(v3coord, 3)):
            n_ok += 1
            
        B1 = np.sqrt(u1coord**2+v1coord**2)
        B2 = np.sqrt(u2coord**2+v2coord**2)
        B3 = np.sqrt(u3coord**2+v3coord**2)
        b_cp = np.max([B1,B2,B3])              
    
        freq_cp.append(b_cp/wl/206264.806247)  #arcsec-1
      
    freq_cp = np.array(freq_cp)    

    if n_ok != n_cp:
        print('### Warning : closure phase not properly closed!')    
            
    return freq_cp

def Sky_correction(imA, r1 = 100, dr = 20, display = False):
    """
    Perform background sky correction to be as close to zero as possible.
    """
    
    ISZ = imA.shape[0]

    xc, yc = ISZ//2, ISZ//2

    xx, yy = np.arange(ISZ), np.arange(ISZ)
    xx2=(xx-xc)
    yy2=(yc-yy)
    
    r2 = r1 + dr

    theta = np.linspace(0, 2*np.pi, 100)
    
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    
    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)
        
    cond_bg = (r1 <= distance) & (distance <= r2)
                
    try:
        backgroundA = np.mean(imA[cond_bg])
        
        minA = imA.min()
        
        imB = imA +  1.01*abs(minA)
                
        backgroundB = np.mean(imB[cond_bg])
            
        imC = imB - backgroundB
        
        backgroundC = np.mean(imC[cond_bg])
        
        if not backgroundC <= 1e-3:
            print('### Warnings: bg too high (%2.3f < 0.001)'%backgroundC)
    
        if display:
            print('### Background correction %2.6f -> %2.6f'%(backgroundA, backgroundC))
                
        extent = [-ISZ//2, ISZ//2, -ISZ//2, ISZ//2]
        
        if display:
            plt.figure()
            plt.imshow(imC, norm = PowerNorm(.5), cmap = 'afmhot', vmin = 0, extent = extent)
            plt.plot(x1, y1)
            plt.plot(x2, y2)
    except:
        imC = None
        backgroundC = 0
        print('### Background not computed: check the inner and outer radius rings.')
        
    return imC, backgroundC

def normalized_img(img):
    return img/np.max(img)

def Apply_mask_apod(img, r = 80, sig = 10):
    ISZ = len(img)
    
    X = [np.arange(ISZ), np.arange(ISZ), 1]

    sig = 10
    param = {'A' : 1,
         'x0' : 0,
         'y0' : 0,
         'fwhm_x' : sig,
         'fwhm_y' : sig,
         'theta' : 0
         }

    gauss = gauss_2d_asym(X, param)

    xx, yy = np.arange(ISZ), np.arange(ISZ)
    xx2=(xx-ISZ//2)
    yy2=(ISZ//2-yy)

    distance  = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)

    mask = np.zeros([ISZ, ISZ])

    mask[distance < r] = 1

    mask_apod = normalized_img(conv_fft(mask, gauss))
    
    img_apod = img * mask_apod
    
    return img_apod

def NRMtoOifits2(dic, filename = None, verbose = False):
    """
    
    """
    
    if dic is not None:
        pass
    else:
        cprint('\nError NRMtoOifits2 : Wrong data format!', on_color='on_red')
        return None
    
    datadir = 'Saveoifits/'
        
    if not os.path.exists(datadir):
        print('### Create %s directory to save all requested Oifits ###'%datadir)
        os.system('mkdir %s'%datadir)
        
    if type(filename) == str:
        pass
    else:
        filename = '%s_%i_%s_%s_%s_%2.0f.oifits'%(dic['info']['TARGET'].replace(' ', ''), dic['info']['NFILE'], dic['info']['INS'], dic['info']['MASK'], dic['info']['FILT'], dic['info']['MJD'])
    
    #------------------------------
    #       Creation OIFITS
    #------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :"%(filename))
          
    refdate = datetime.datetime(2000, 1, 1) #Unix time reference
    
    hdulist = fits.HDUList()
    
    hdr = dic['info']['HDR']

    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = datetime.datetime.now().strftime(format='%F')#, 'Creation date'
    hdu.header['ORIGIN'] = 'Sydney University'
    #hdu.header['DATE-OBS'] = hdr['DATE-OBS']
    hdu.header['CONTENT'] = 'OIFITS2'
    hdu.header['TELESCOP'] = hdr['TELESCOP']
    hdu.header['INSTRUME'] = hdr['INSTRUME']
    hdu.header['OBSERVER'] = hdr['OBSERVER']
    hdu.header['OBJECT'] = hdr['OBJECT']
    hdu.header['INSMODE'] = 'NRM'
    
    hdulist.append(hdu)
    #------------------------------
    #        OI Wavelength
    #------------------------------
    
    if verbose:
        print('-> Including OI Wavelength table...')
    data = dic['OI_WAVELENGTH']
    
    #Data
    # -> Initiation new hdu table :
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='EFF_WAVE', format='1E', unit='METERS', array=[data['EFF_WAVE']]),
        fits.Column(name='EFF_BAND', format='1E', unit='METERS', array=[data['EFF_BAND']])
        )))
    
    #Header
    hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    hdu.header['OI_REVN'] = 2#, 'Revision number of the table definition'
    hdu.header['INSNAME'] = dic['info']['INS']#'Name of detector, for cross-referencing'    
    hdulist.append(hdu) #Add current HDU to the final fits file.
    
    #------------------------------
    #          OI Target
    #------------------------------
    if verbose:
        print('-> Including OI Target table...')
    
    name_star = dic['info']['TARGET']

    customSimbad = Simbad()
    customSimbad.add_votable_fields('propermotions','sptype', 'parallax')
    
    #Add information from Simbad:
    try:
        query = customSimbad.query_object(name_star)
        coord = SkyCoord(query['RA'][0]+' '+query['DEC'][0], unit=(u.hourangle, u.deg))
    
        ra = [coord.ra.deg]
        dec = [coord.dec.deg]
        spectyp = query['SP_TYPE']
        pmra = query['PMRA']
        pmdec = query['PMDEC']
        plx = query['PLX_VALUE']
    except:
        ra = [0]
        dec = [0]
        spectyp = ['fake']
        pmra = [0]
        pmdec = [0]
        plx = [0]
    
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=[1]),
        fits.Column(name='TARGET', format='16A', array=[name_star]),
        fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=ra),
        fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=dec),
        fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=[2000]),
        fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='SYSVEL', format='1D', unit='M/S', array=[0]),
        fits.Column(name='VELTYP', format='8A', array=['UNKNOWN']),
        fits.Column(name='VELDEF', format='8A', array=['OPTICAL']),
        fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra),
        fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec),
        fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=plx),
        fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=[0]),
        fits.Column(name='SPECTYP', format='16A', array=spectyp)
        )))
    
    hdu.header['EXTNAME'] = 'OI_TARGET'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdulist.append(hdu)
        
    #------------------------------
    #           OI Array
    #------------------------------
    
    if verbose:
        print('-> Including OI Array table...')
        
    STAXY = Saved_mask()[dic['info']['MASK']]
    
    N_ap = len(STAXY)
    
    TEL_NAME = ['A%i'%x for x in np.arange(N_ap)+1]
    STA_NAME = TEL_NAME
    DIAMETER = [0] * N_ap
    
    STAXYZ = []
    for x in STAXY:
        a = list(x)
        line = [a[0], a[1], 0]
        STAXYZ.append(line)
        
    STA_INDEX = np.arange(N_ap) + 1
    
    PSCALE = dic['info']['PSCALE']/1000. #arcsec
    ISZ = dic['info']['ISZ'] #Size of the image to extract NRM data
    FOV = [PSCALE * ISZ] * N_ap
    FOVTYPE = ['RADIUS'] * N_ap
    
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
                    fits.Column(name='TEL_NAME', format='16A', array=TEL_NAME),#['dummy']),
                    fits.Column(name='STA_NAME', format='16A', array=STA_NAME),#['dummy']),
                    fits.Column(name='STA_INDEX', format='1I', array=STA_INDEX),
                    fits.Column(name='DIAMETER', unit='METERS', format='1E', array=DIAMETER),
                    fits.Column(name='STAXYZ', unit='METERS', format='3D', array=STAXYZ),
                    fits.Column(name='FOV', unit='ARCSEC', format='1D', array=FOV),
                    fits.Column(name='FOVTYPE', format='6A', array=FOVTYPE),
                    )))
    
    hdu.header['EXTNAME'] = 'OI_ARRAY'
    hdu.header['ARRAYX'] = float(0)
    hdu.header['ARRAYY'] = float(0)
    hdu.header['ARRAYZ'] = float(0)
    hdu.header['ARRNAME'] =  dic['info']['MASK']
    hdu.header['FRAME'] = 'SKY'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'

    hdulist.append(hdu)
    
    #------------------------------
    #           OI VIS
    #------------------------------
    
    if verbose:
        print('-> Including OI Vis table...')
     
    data =  dic['OI_VIS']
    npts = len(dic['OI_VIS']['VISAMP'])
        
    STA_INDEX = Format_STAINDEX_V2(data['STA_INDEX'])
    
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='TARGET_ID', format='1I', array=[data['TARGET_ID']]*npts),
            fits.Column(name='TIME', format='1D', unit='SECONDS', array=[data['TIME']]*npts),
            fits.Column(name='MJD', unit='DAY', format='1D', array=[data['MJD']]*npts),
            fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=[data['INT_TIME']]*npts),
            fits.Column(name='VISAMP', format='1D', array=data['VISAMP']),
            fits.Column(name='VISAMPERR', format='1D', array=data['VISAMPERR']),
            fits.Column(name='VISPHI', format='1D', unit='DEGREES', array=np.rad2deg(data['VISPHI'])),
            fits.Column(name='VISPHIERR', format='1D', unit='DEGREES', array=np.rad2deg(data['VISPHIERR'])),
            fits.Column(name='UCOORD', format='1D', unit='METERS', array=data['UCOORD']),
            fits.Column(name='VCOORD', format='1D', unit='METERS', array=data['VCOORD']),
            fits.Column(name='STA_INDEX', format='2I', array=STA_INDEX),
            fits.Column(name='FLAG', format='1L', array = data['FLAG'])
            ]))

    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['EXTNAME'] = 'OI_VIS'
    hdu.header['INSNAME'] = dic['info']['INS']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
    hdulist.append(hdu)
    
    #------------------------------
    #           OI VIS2
    #------------------------------
    
    if verbose:
        print('-> Including OI Vis2 table...')
     

    data =  dic['OI_VIS2']
    npts = len(dic['OI_VIS2']['VIS2DATA'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='TARGET_ID', format='1I', array=[data['TARGET_ID']]*npts),
            fits.Column(name='TIME', format='1D', unit='SECONDS', array=[data['TIME']]*npts),
            fits.Column(name='MJD', unit='DAY', format='1D', array=[data['MJD']]*npts),
            fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=[data['INT_TIME']]*npts),
            fits.Column(name='VIS2DATA', format='1D', array=data['VIS2DATA']),
            fits.Column(name='VIS2ERR', format='1D', array=data['VIS2ERR']),
            fits.Column(name='UCOORD', format='1D', unit='METERS', array=data['UCOORD']),
            fits.Column(name='VCOORD', format='1D', unit='METERS', array=data['VCOORD']),
            fits.Column(name='STA_INDEX', format='2I', array=STA_INDEX),
            fits.Column(name='FLAG', format='1L', array = data['FLAG'])
            ]))

    hdu.header['EXTNAME'] = 'OI_VIS2'
    hdu.header['INSNAME'] = dic['info']['INS']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
    hdulist.append(hdu)
    
    #------------------------------
    #           OI T3
    #------------------------------
    
    if verbose:
        print('-> Including OI T3 table...')

    data =  dic['OI_T3']
    npts = len(dic['OI_T3']['T3PHI'])
    
    STA_INDEX = Format_STAINDEX_T3(data['STA_INDEX'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=[1]*npts),
        fits.Column(name='TIME', format='1D', unit='SECONDS', array=[0]*npts),
        fits.Column(name='MJD', format='1D', unit='DAY', array=[data['MJD']]*npts),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=[data['INT_TIME']]*npts),
        fits.Column(name='T3AMP', format='1D', array=data['T3AMP']),
        fits.Column(name='T3AMPERR', format='1D', array=data['T3AMPERR']),
        fits.Column(name='T3PHI', format='1D', unit='DEGREES', array=np.rad2deg(data['T3PHI'])),
        fits.Column(name='T3PHIERR', format='1D', unit='DEGREES', array=np.rad2deg(data['T3PHIERR'])),
        fits.Column(name='U1COORD', format='1D', unit='METERS', array=data['U1COORD']),
        fits.Column(name='V1COORD', format='1D', unit='METERS', array=data['V1COORD']),
        fits.Column(name='U2COORD', format='1D', unit='METERS', array=data['U2COORD']),
        fits.Column(name='V2COORD', format='1D', unit='METERS', array=data['V2COORD']),
        fits.Column(name='STA_INDEX', format='3I', array=STA_INDEX),
        fits.Column(name='FLAG', format='1L', array = data['FLAG'])
        )))

    hdu.header['EXTNAME'] = 'OI_T3'
    hdu.header['INSNAME'] = dic['info']['INS']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = refdate.strftime('%F'), 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    #------------------------------
    #          Save file
    #------------------------------

    hdulist.writeto(datadir + filename, overwrite=True)
    cprint('\n\n### OIFITS CREATED.', 'cyan')
    
    return None

def Format_STAINDEX_V2(tab):
    STA_INDEX = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        line = np.array([ap1, ap2]) + 1
        STA_INDEX.append(line)
    return STA_INDEX

def Format_STAINDEX_T3(tab):
    STA_INDEX = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        line = np.array([ap1, ap2, ap3]) + 1
        STA_INDEX.append(line)
    return STA_INDEX

def poly2(x, param):
    a = param[0]
    b = param[1]
    c = param[2]
    
    y = a*x**2 + b*x**1 + c*x**0
    return y

def Calib_poly_vis(star, calib, order = 2, display = False):
    """
    
    """
            
    nbl = star.vis2.shape[1]
    
    #Single or multiple calibration stars
    if type(calib) == list:
        nrm_c = calib[0]
        nrm_c2 = calib[1]
        multi = True
    else:
        nrm_c = calib
        multi = False
        
    vis2cal = list(np.zeros(nbl))
    
    for n in range(nbl):
        tab_sci = np.array([star.time, star.vis2[:,n]]).transpose()
        tab_sci_sorted = np.array(sorted(tab_sci, key=lambda x: x[0]))
        Xsci, Ysci = tab_sci_sorted[:,0], tab_sci_sorted[:,1]
        
        tab_cal1 = np.array([nrm_c.time, nrm_c.vis2[:,n]]).transpose()
        tab_cal1_sorted = np.array(sorted(tab_cal1, key=lambda x: x[0]))
        Xcal1, Ycal1 = tab_cal1_sorted[:,0], tab_cal1_sorted[:,1]
        
        if multi:        
            tab_cal2 = np.array([nrm_c2.time, nrm_c2.vis2[:,n]]).transpose()
            tab_cal2_sorted = np.array(sorted(tab_cal2, key=lambda x: x[0]))
            Xcal2, Ycal2 = tab_cal2_sorted[:,0], tab_cal2_sorted[:,1]

            Xcal, Ycal = np.concatenate([Xcal1, Xcal2]), np.concatenate([Ycal1, Ycal2])
        
        else:
            Xcal, Ycal = Xcal1, Ycal1
            
        tab_cal = np.array([Xcal, Ycal]).transpose()
        tab_cal_sorted = np.array(sorted(tab_cal, key=lambda x: x[0]))
        
        Xcal_sorted, Ycal_sorted = tab_cal_sorted[:,0], tab_cal_sorted[:,1]
    
        #Fit polynomial function to the visibility curve = transfert function
        fit = np.polyfit(Xcal_sorted, Ycal_sorted, order)
    
        X_model = np.linspace(Xcal_sorted.min() - .2/24., Xcal_sorted.max() + .2/24., 100)
        model = poly2(X_model, fit)
    
        #Determine calibration factor using polynome or averages
        Fact_calib_vis2_poly = poly2(Xsci, fit)
        Fact_calib_vis2_mean = np.ones(len(Xsci)) * np.mean(Ycal_sorted)
    
        Ysci_calib_poly = Ysci/Fact_calib_vis2_poly
        Ysci_calib_mean = Ysci/Fact_calib_vis2_mean
                
        vis2cal[n] = np.array(Ysci_calib_poly)
        
        if display:
            plt.figure()
            plt.plot(Xsci, Ysci, 's', label = 'Sci')
            plt.plot(Xcal, Ycal, '+', label = 'Cal')
            plt.plot(X_model, model, 'r--', label = 'fit')
            plt.plot(X_model, np.ones(len(X_model))*np.mean(Ycal_sorted))
            plt.legend()
            
            plt.figure()
            plt.plot(Ysci_calib_poly, label = 'Polynomial fit')
            plt.plot(Ysci_calib_mean, label = 'Average normalization')
            plt.plot(np.arange(len(Ysci_calib_poly)), np.ones(len(Ysci_calib_poly)) * np.mean(Ysci_calib_poly), label = 'new')
            plt.plot(np.arange(len(Ysci_calib_mean)), np.ones(len(Ysci_calib_mean)) * np.mean(Ysci_calib_mean), label = 'before')
            
            plt.legend()
    
    return np.array(vis2cal).transpose()

def plot_square(dr, x0=0, y0=0, color = 'w'):
        a = dr//2
        sqx1 = np.array([-a, a]) + x0
        sqy1 = np.array([-a, -a]) + y0
        sqx2 = np.array([-a, a]) + x0
        sqy2 = np.array([a, a]) + y0
        sqx3 = np.array([-a, -a]) + x0
        sqy3 = np.array([-a, a]) + y0
        sqx4 = np.array([a, a]) + x0
        sqy4 = np.array([-a, a]) + y0
        plt.plot(sqx1, sqy1, '-', color = color)
        plt.plot(sqx2, sqy2, '-', color = color)
        plt.plot(sqx3, sqy3, '-', color = color)
        plt.plot(sqx4, sqy4, '-', color = color)
        return None
    
def Check_NRM_radius(im, icrop, ISZ, r1, dr, rot=0):
    """
    Check the position of each resizing parameters.
    
    Parameters:
    -----------
    
    im: numpy.array
        Image to check the good position of each function parameters.
    icrop: int
        Size of the cropped image around the middle.
    ISZ: int
        Size of the finale image where is calculated the FFT.
    r1: int
        Inner radius position to integrate the background.
    dr: int
        Outer radius position to integrate the background (r2 = r1 + dr).
    """
    
    im_rot = rotate(im, rot)
    
    tmp = crop_center(im_rot, icrop)
    
    r2 = r1 + dr

    theta = np.linspace(0, 2*np.pi, 100)
    
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    
    max_source = tmp.max()
    
    imB, cc = crop_max(tmp, ISZ)
    
    imsize = im.shape[0]
    xc = (imsize - icrop)/2 + cc[0] - 1
    yc = (imsize - icrop)/2 + cc[1] - 1
    
    
    imC, bg = Sky_correction(imB, r1 = r1, dr = dr, display = False)
    
    if type(imC) != np.ndarray:
        print('### Problem with the current frame: sky correction not performed.')
        return None
    
    extent2 = [-ISZ//2, ISZ//2, -ISZ//2, ISZ//2]

    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.imshow(im, vmin = 0, vmax = max_source, norm = PowerNorm(.5), cmap = 'afmhot')#, extent = extent)
    plot_square(icrop, imsize//2, imsize//2, color = 'w')
    plot_square(ISZ, xc, yc, color = 'gold')
    plt.plot(xc, yc, 'b+')
    plt.text(.05*imsize, .95*imsize, 'RAW', color = 'w')
    plt.text(10 + (imsize//2-icrop//2), .95*(imsize//2+icrop//2), 'CROPPED', color = 'w')
    plt.text(1.1*((imsize//2-ISZ//2) - (imsize//2 - xc)), .9*((imsize//2+ISZ//2)- (imsize//2 - yc)), 'FINALE', color = 'gold')
    
    plt.subplot(1,2,2)
    plt.imshow(imC, vmin = 0, vmax = max_source, norm = PowerNorm(.5), cmap = 'afmhot', extent = extent2)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plot_square(ISZ-0.5, color = 'gold')
    
    plt.tight_layout()
    
    return None



