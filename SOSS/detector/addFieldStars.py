# Astronomy imports
from astroquery.ipac.irsa import Irsa
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
import astropy.coordinates as crd

#General imports
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from math import sin,cos,pi
from scipy.io import readsav
from mpl_toolkits.mplot3d import Axes3D


def sossFieldSim(ra, dec , APA, refDir, binComp=None, pmInfo=None, dimY=256):
   
    """   
    Creates the field stars .fits file to be added. Needs the coordinates of the target to
    identify the right field stars. 

    ## Parameters
    **ra**: Target's right ascension
    **dec**: Target's declinaison
    **APA**: Chosen APA for this observation
    **refDir**: Directory containing reference files
    **binComp**: Missing field stars to be added [deltaRA,deltaDEC,J,H,K]
    **pmInfo**: Apply target proper motion to 2020 jan 1.
    **dimY**: Change for the right Y dimension if not SUBSCRIPT256

    """

    # stars in large field around target
    targetcrd = crd.SkyCoord([ra+' '+dec], unit=(u.hour, u.deg))
    targetcrd = targetcrd[0]
    #("target coordinates:", targetcrd)

    targetRA = targetcrd.ra.value
    targetDEC = targetcrd.dec.value
    info   = Irsa.query_region(targetcrd, catalog = 'fp_psc', spatial = 'Cone', radius = 2.5*u.arcmin) #changed
   
    # coordinates of all stars in FOV, including target
    allRA   = info['ra'].data.data 
    allDEC  = info['dec'].data.data
    Jmag        = info['j_m'].data.data
    Hmag        = info['h_m'].data.data
    Kmag        = info['k_m'].data.data
    J_Hobs      = Jmag-Hmag
    H_Kobs      = Hmag-Kmag
    
    # Distances of the field stars from the target
    distance    = np.sqrt( ((targetRA-allRA)*np.cos(targetDEC))**2 + (targetDEC-allDEC)**2 )
    targetIndex = np.argmin(distance) # the target

    #number of stars
    nStars=allRA.size
    
    #apply target proper motion to 2020 jan 1.
    if pmInfo is not None:
        epoch_cycle1=Time('2020-01-01').jyear
        epoch_coo=pmInfo[2]
        dt=epoch_cycle1-epoch_coo
        allRA[targetIndex]+=pmInfo[0]/cos(allDEC[targetIndex]*pi/180)/1000/3600*dt
        allDEC[targetIndex]+=pmInfo[1]/1000/3600*dt
        cubeNameSuf +='_custom'          

    # add any missing companion
    if binComp is not None:
        allRA    = np.append(allRA, (allRA[targetIndex] + binComp[0]/3600/cos(allDEC[targetIndex]*pi/180)))
        allDEC   = np.append(allDEC, (allDEC[targetIndex] + binComp[1]/3600))
        Jmag        = np.append(Jmag,binComp[2])
        Hmag        = np.append(Kmag,binComp[3])
        Kmag        = np.append(Kmag,binComp[4])
        J_Hobs      = Jmag-Hmag
        H_Kobs      = Hmag-Kmag
        cubeNameSuf +='_custom'          

    cooTar=crd.SkyCoord(ra=allRA[targetIndex],dec=allDEC[targetIndex], unit=(u.deg, u.deg))

    dimX = 2048
        
    #Restoring model parameters
    # modelParam = idlsave.read('idlSaveFiles/modelsInfo.sav') 
    modelParam = readsav(refDir+'modelsInfo.sav',verbose=False) 
    dimX = 2048 #Dimension of the output images
    dimYmod    = 2048 #Dimensions of the model
    dimXmod    = 6048 
    jhMod      = modelParam['jhmod']
    hkMod      = modelParam['hkmod']
    teffMod    = modelParam['teffmod'] 
    modelPadX  = dimXmod - modelParam["modelPadY"] - dimX
    if dimY == 2048: 
        modelPadY  = dimYmod - modelParam["modelPadX"] - 2048
    else:
        modelPadY  = dimYmod - modelParam["modelPadX"] - 256


    # find/assign Teff of each star
    starsT=np.empty(nStars)
    for j in range(nStars):
        color_separation = (J_Hobs[j]-jhMod)**2+(H_Kobs[j]-hkMod)**2
        min_separation_ind = np.argmin(color_separation)
        starsT[j]=teffMod[min_separation_ind]

    radeg = 180/pi
    niriss_pixel_scale = 0.065  # arcsec
    sweetSpot = dict(x=856,y=107,RA=allRA[targetIndex],DEC=allDEC[targetIndex],jmag=Jmag[targetIndex])
    sweetSpot['x'], sweetSpot['y'] = 2048-sweetSpot['y'], 2048-sweetSpot['x']

    #offset between all stars and target
    dRA=(allRA - sweetSpot['RA'])*np.cos(sweetSpot['DEC']/radeg)*3600
    dDEC=(allDEC - sweetSpot['DEC'])*3600
    
    # Put field stars positions and magnitudes in structured array
    _ = dict(RA=allRA, DEC=allDEC, dRA=dRA, dDEC=dDEC, jmag=Jmag, T=starsT,
             x=np.empty(nStars), y=np.empty(nStars), dx=np.empty(nStars), dy=np.empty(nStars))
    stars=np.empty(nStars,dtype=[(key,val.dtype) for key,val in _.items()])
    for key,val in _.items(): stars[key]=val
   
    # Initialize final fits cube that contains the modelled traces with contamination
    simuCube=np.zeros([dimY, dimX])  # cube of trace simulation at every degree of field rotation
    
    #Model image of the traces 
    #FieldStars_model = '/home/plamontagne/ongenesis/userland-soss/neighbour_star_noise/star_model_image/new_field_models.fits'
    FieldStars_model = refDir+'new_field_models.fits'
    
    V3PA=APA+0.57 #from APT
    
    #stars['dx']= (np.cos(np.pi/2+APA/radeg)*stars['dRA']-np.sin(np.pi/2+APA/radeg)*stars['dDEC'])/niriss_pixel_scale #Offsets from target
    #stars['dy']= (np.sin(np.pi/2+APA/radeg)*stars['dRA']+np.cos(np.pi/2+APA/radeg)*stars['dDEC'])/niriss_pixel_scale
    stars['dy']= -(np.cos(np.pi/2+APA/radeg)*stars['dRA']-np.sin(np.pi/2+APA/radeg)*stars['dDEC'])/niriss_pixel_scale #Offsets from target
    stars['dx']= -(np.sin(np.pi/2+APA/radeg)*stars['dRA']+np.cos(np.pi/2+APA/radeg)*stars['dDEC'])/niriss_pixel_scale
    
    stars['x'] = stars['dx']+sweetSpot['x'] #Absolute positions
    stars['y'] = stars['dy']+sweetSpot['y']
    

    # Display the star field (blue), target (red), subarray (green), full array (blue), and axes
    if True:
        plt.plot([0,2047,2047,0,0],[0,0,2047,2047,0], 'b')
        plt.plot([0,2047,2047,0,0],[2047,2047,2047-256,2047-256,2047], 'g')
        #the order 1 & 2 traces
        t1=np.loadtxt(refDir+'trace_order1.txt',unpack=True)
        t1[0], t1[1] = 2048 - t1[1], 2048 - t1[0]
        plt.plot(t1[0],t1[1],'r') #Plotting trace 1
        t2=np.loadtxt(refDir+'trace_order2.txt',unpack=True)
        t2[0], t2[1] = 2048 - t2[1], 2048 - t2[0] #Plotting trace 2
        plt.plot(t2[0],t2[1],'r')
        plt.plot(stars['x'], stars['y'], 'b*')
        plt.plot(sweetSpot['x'], sweetSpot['y'], 'r*')
        plt.title("APA= {} (V3PA={})".format(APA,V3PA))
        ax=plt.gca()

        #add V2 & V3 axes
        l,hw,hl=250,50,50
        adx,ady=-l*np.cos(-0.57/radeg),-l*np.sin(-0.57/radeg)
        ax.arrow(2500, 1800, adx,ady, head_width=hw, head_length=hl, length_includes_head=True, fc='k') #V3
        plt.text(2500+1.4*adx,1800+1.4*ady,"V3",va='center',ha='center')
        adx,ady=-l*np.cos((-0.57-90)/radeg),-l*np.sin((-0.57-90)/radeg)
        ax.arrow(2500, 1800, adx, ady, head_width=hw, head_length=hl, length_includes_head=True, fc='k') #V2
        plt.text(2500+1.4*adx,1800+1.4*ady,"V2",va='center',ha='center')
        #add North and East
        adx,ady=-l*np.cos(APA/radeg),-l*np.sin(APA/radeg)
        ax.arrow(2500, 1300, adx, ady, head_width=hw, head_length=hl, length_includes_head=True, fc='k') #N
        plt.text(2500+1.4*adx,1300+1.4*ady,"N",va='center',ha='center')
        adx,ady=-l*np.cos((APA-90)/radeg),-l*np.sin((APA-90)/radeg)
        ax.arrow(2500, 1300, adx, ady, head_width=hw, head_length=hl, length_includes_head=True, fc='k') #E
        plt.text(2500+1.4*adx,1300+1.4*ady,"E",va='center',ha='center')

        ax.set_xlim(-400,2047+800)
        ax.set_ylim(-400,2047+400)
        ax.set_aspect('equal')
        plt.show()

        # email from Michael Maszkiewicz on 2016 May 17:
        #  POM edge is:
        #     1.67-2.01 mm left of detector in x
        #     2.15-2.22 mm above detector in y
        #     2.22-2.43 mm right of detector in x
        #     1.49-1.87 mm below detector in y
        #  we take the larger value in each direction and add a 50 pixels margin
        #  given the uncertainty of the POM location wrt to detector FOV

    # Retain stars that are within the Direct Image NIRISS POM FOV
    ind, = np.where((stars['x'] >= -162) & (stars['x'] <= 2047+185) & (stars['y'] >= -154) & (stars['y'] <= 2047+174))
    starsInFOV=stars[ind]

    print("There are",starsInFOV.size,"stars in the field of view")

    #Finding the indice of the target
    target = np.where((np.round(starsInFOV['dx']) == 0) & (np.round(starsInFOV['dy']) == 0))

    

    for i in range(len(ind)):
        intx = round(starsInFOV['dx'][i])
        inty = round(starsInFOV['dy'][i])

        k=np.where(teffMod == starsInFOV['T'][target])[0][0] #Change "target" to "i" if you want the temperatures of every star to be taken into account, 
                                                                #now it's only the temperature of the target
        
        
        fluxscale = 10.0**(-0.4*(starsInFOV['jmag'][i] - sweetSpot['jmag']))

        #deal with subection sizes
        mx0=int(modelPadX-intx) #Where the added image begins
        mx1=int(modelPadX-intx+dimX) #Where the added image ends
        my0=int(modelPadY-inty)
        my1=int(modelPadY-inty+dimY)

        
        if (mx0 > dimXmod) or (my0 > dimYmod): #If the beginning of the model is after the end of the image, skip to next star...
            continue
        if (mx1 < 0) or (my1 < 0): #If the end of the model is before the beginning of the image, skip to next star... 
            continue
        
        x0  =(mx0<0)*(-mx0) #The beginning of the model in the final image
        y0  =(my0<0)*(-my0)
        mx0 *=(mx0 >= 0) #If the model image is shifted too much to the left/down, begin it at 0
        mx1 = dimXmod if mx1>dimXmod else mx1 #If the model image is shifted to much to the right/top, end it at dimXmod/dimYmod
        my0 *=(my0 >= 0)
        my1 =dimYmod if my1>dimYmod else my1


        # if target, skip to next star
        if (intx == 0) & (inty == 0):   
            continue           
            
        #field stars    
        if (intx != 0) or (inty != 0):  
            # temp_target = FieldStars_model[k].replace("/home/plamontagne/ongenesis/userland-soss/neighbour_star_noise/star_model_image/modelOrder12_","").replace(".fits","") #Temperature
            #("Temperature model of the target:",temp_target)
            with fits.open(FieldStars_model, verbose=False) as model_image_field: 
                model_field_final = model_image_field[0].data
            
            #Adding the field stars to the image
            simuCube[y0:y0+my1-my0, x0:x0+mx1-mx0] += model_field_final[k, my0:my1, mx0:mx1] * fluxscale
                
    return simuCube

