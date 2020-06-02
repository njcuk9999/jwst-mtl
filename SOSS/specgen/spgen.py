import numpy as np #numpy gives us better array management 
import os #checking status of requested files
from astropy.io import fits #astropy modules for FITS IO
import tfit5

class ModelPars:
    """Default Model Parameters
    """
    
    nplanetmax=9 #code is hardwired to have upto 9 transiting planets. 
    #default parameters -- these will cause the program to end quickly
    tstart=0.0e0 #start time (hours)
    tend=0.0e0 #end time (hours)
    exptime=0.0e0 #exposure time (s)
    deadtime=0.0e0 #dead time (s)
    modelfile='null' #stellar spectrum file name
    nmodeltype=2 #stellar spectrum type. 1=BT-Settl, 2=Atlas-9+NL limbdarkening
    rvstar=0.0e0 #radial velocity of star (km/s)
    vsini=0.0e0 #projected rotation of star (km/s)
    pmodelfile=[None]*nplanetmax #file with Rp/Rs values 
    emisfile=[None]*nplanetmax #file with emission spectrum
    ttvfile=[None]*nplanetmax #file with O-C measurements
    #nplanet is tracked by pmodelfile. 
    nplanet=0 #number of planets -- default is no planets - you will get staronly sim.
    sol=np.zeros(nplanetmax*8+1)
    sol[0]=1.0 #mean stellar density [g/cc]
    xout=2048  #dispersion axis
    yout=256   #spatial axis
    noversample=1 #oversampling
    saturation=65536.0e0 #saturation
    ngroup=1 #samples up ramp
    pid = 1 #programID
    onum = 1 #observation number
    vnum = 1 #visit number
    gnum = 1 #group visit
    spseq = 1 #parallel sequence. (1=prime, 2-5=parallel)
    anumb = 1 #activity number
    enum = 1 #exposure number
    enumos = 1 #exposure number for oversampling
    detectorname = 'NISRAPID' #confirm this 
    prodtype='cal'
    

def read_pars(filename,pars):
    """Usage:  pars=read_pars(filename,pars)
    
      filename -- text file containing model parameters
      pars     -- class with model parameters
    """
    
    linenumber=0 #track lie number
    nplanet=0
    if os.path.isfile(filename):  #Check that file exists, else print an error
        
        f = open(filename, 'r') #open file read-only
        
        for line in f:
            linenumber+=1
            line = line.strip()
            columns = line.split() #break into columns
            if len(columns) > 0: #ignore blank lines
                if columns[0][0] != '#':
                    command=str(columns[0]).lower()
                    nlcom=len(command) #number of characters in the command
                    
                    if command == 'tstart':
                        pars.tstart = np.float(columns[1])
                    elif command == 'tend':
                        pars.tend = np.float(columns[1])
                    elif command == 'exptime':
                        pars.exptime = np.float(columns[1])
                    elif command == 'deadtime':
                        pars.deadtime = np.float(columns[1])
                    elif command == 'rhostar':
                        pars.sol[0] = np.float(columns[1])
                    elif command == 'starmodel':
                        pars.modelfile = str(columns[1])
                    elif command == 'startype':
                        pars.nmodeltype = int(columns[1])
                        if pars.nmodeltype != 2:
                            print("Error: Only ATLAS-9 models are currently support (STARTYPE=2)")
                            print('Linenumber: ',linenumber)
                    elif command == 'vsini':
                        pars.vsini = np.float(columns[1])
                    elif command == 'xout':
                        pars.xout = int(np.float(columns[1]))
                    elif command == 'yout':
                        pars.yout = int(np.float(columns[1]))
                    elif command == 'xcoo':
                        pars.xcoo = np.float(columns[1])
                    elif command == 'ycoo':
                        pars.ycoo = np.float(columns[1])
                    elif command == 'roll':
                        pars.roll = np.float(columns[1])
                    elif command == 'xcen':
                        pars.xcen = np.float(columns[1])
                    elif command == 'ycen':
                        pars.ycen = np.float(columns[1])
                    elif command == 'xjit':
                        pars.xcen = np.float(columns[1])
                    elif command == 'yjit':
                        pars.ycen = np.float(columns[1])
                    elif command == 'rolljit':
                        pars.xcen = np.float(columns[1])
                    elif command == 'oversample':
                        pars.noversample = int(np.float(columns[1]))
                    elif command == 'saturation':
                        pars.saturation = np.float(columns[1])
                    elif command == 'ngroup':
                        pars.ngroup = int(np.float(columns[1]))
                    elif command == 'pid':
                        pars.pid = int(np.float(columns[1]))
                    elif command == 'onum':
                        pars.onum = int(np.float(columns[1]))
                    elif command == 'vnum':
                        pars.vnum = int(np.float(columns[1]))
                    elif command == 'gnum':
                        pars.gnum = int(np.float(columns[1]))
                    elif command == 'spseq':
                        pars.spseq = int(np.float(columns[1]))
                    elif command == 'anumb':
                        pars.anumb = int(np.float(columns[1]))
                    elif command == 'enum':
                        pars.enum = int(np.float(columns[1]))
                    elif command == 'enumos':
                        pars.enumos = int(np.float(columns[1]))
                    elif command == 'detector':
                        pars.detector = str(columns[1])
                    elif command == 'prodtype':
                        pars.prodtype = str(columns[1])
                    elif command[0:nlcom-1] == 'rprsfile':
                        if str(columns[1]) != 'null':
                            npl=int(np.float(command[nlcom-1]))
                            if (npl <= pars.nplanetmax) &(npl>0):
                                nplanet=np.max((nplanet,npl))
                                pars.pmodelfile[npl-1]=str(columns[1])
                            else:
                                print('Error: Planet number is Invalid ',npl )
                                print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'emisfile':
                        if str(columns[1]) != 'null':
                            npl=int(np.float(command[nlcom-1]))
                            if (npl <= pars.nplanetmax) &(npl>0):
                                nplanet=np.max((nplanet,npl))
                                pars.emisfile[npl-1]=str(columns[1])
                                print('Warning: Emission Spectrum has not yet been implemented')
                            else:
                                print('Error: Planet number is Invalid ',npl)
                                print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'ttvfile':
                        if str(columns[1]) != 'null':
                            npl=int(np.float(command[nlcom-1]))
                            if (npl <= pars.nplanetmax) &(npl>0):
                                nplanet=np.max((nplanet,npl))
                                pars.ttvfile[npl-1]=str(columns[1])
                            else:
                                print('Error: Planet number is Invalid ',npl )
                                print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'ep':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+1]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'pe':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+2]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'bb':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+3]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'ec':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+4]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'es':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+5]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'rv':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+6]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'el':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+7]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'al':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+8]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    else:
                        print("Not found: ",command,nlcom,command[0:nlcom-1],command[nlcom-1])
                        print('Linenumber: ',linenumber)
                        npl=int(np.float(command[nlcom-1]))
                        nplanet=np.max((nplanet,npl))
        f.close()
        pars.nplanet=nplanet*1 #store number of planets found.
        
    else:
        print('Cannot find ',filename)
    
    return pars;

class response_class:
    def __init__(self):
        self.wv=[]  #initialize arrays
        self.response=[]
        self.response_order=[]
        self.quantum_yield=[]

def readresponse(response_file):
    """Usage: response=readresponse(response_file)
    Inputs
     response_file - FITS file for instrument response.

    Outputs:

     response_class
       wv(npt) : wavelength
       response(norder,npt) : response
       response_order(norder) : order index
       quantum_yield(npt) : quantum yield
    """

    response=response_class() #class to contain response info

    hdulist = fits.open(response_file)
    tbdata = hdulist[1].data                   #fetch table data for HUD=1

    response.wv=tbdata.field(0)[0]*10.0         #Wavelength (A)
    for i in range(4):
        response.response.append(tbdata.field('SOSS_order'+str(i))[0]) #n=i response
        response.response_order.append(i) #store order
    response.quantum_yield=tbdata.field('yield')[0] #quantum yield

    hdulist.close()                            #Close FITS file

    return response

def readresponse_old(response_file):
    """Usage: ld,res0,res1,res2,res3,qy=readresponse(response_file)
    Inputs
     response_file - FITS file for instrument response.

    Outputs:

     ld - wavelength (A)
     res0,1,2,3 - responce for orders 0,1,2,3
     qy - quantum yield
    """
    hdulist = fits.open(response_file)
    tbdata = hdulist[1].data                   #fetch table data for HUD=1
    reponse_ld=tbdata.field(0)[0]*10.0         #Wavelength (A)
    reponse_n0=tbdata.field('SOSS_order0')[0]  #n=0 response
    reponse_n1=tbdata.field('SOSS_order1')[0]  #n=1 response
    reponse_n2=tbdata.field('SOSS_order2')[0]  #n=2 response
    reponse_n3=tbdata.field('SOSS_order3')[0]  #n=3 response
    quantum_yield=tbdata.field('yield')[0]
    hdulist.close()                            #Close FITS file

    return reponse_ld,reponse_n0,reponse_n1,reponse_n2,reponse_n3,quantum_yield;

def readstarmodel(starmodel_file,nmodeltype):
    """Usage: starmodel_wv,starmodel_flux=readstarmodel(starmodel_file,smodeltype)
    Inputs:
      starmodel_file - full path and filename to star spectral model
      smodeltype - type of model.  2==ATLAS 

    Returns:
      starmodel_wv : wavelength (A)
      starmodel_flux : flux
      ld_coeff : non-linear limb-darkening coefficients
    """
    starmodel_wv=[]
    starmodel_flux=[]
    ld_coeff=[]

    if nmodeltype==2:

        f = open(starmodel_file,'r')
        for line in f:
            line = line.strip() #get rid of \n at the end of the line
            columns = line.split() #break into columns with space delimiter
            starmodel_wv.append(float(columns[0]))
            flux=-float(columns[5])*np.pi*(42.0*float(columns[1])+70.0*float(columns[2])\
                    +90.0*float(columns[3])+105.0*float(columns[4])-210.0)/210.0
            starmodel_flux.append(np.max([0.0,flux]))
            ld_coeff.append([float(columns[1]),float(columns[2]),float(columns[3])\
                ,float(columns[4])])
        f.close()

        starmodel_wv=np.array(starmodel_wv)
        starmodel_flux=np.array(starmodel_flux)
        ld_coeff=np.array(ld_coeff)

    else:
        print('Currently on ATLAS-9 models are supported (nmodeltype=2)')

    return starmodel_wv,starmodel_flux,ld_coeff;

def readplanetmodel(planetmodel_file,pmodeltype):
    """Usage: planetmodel_wv,planetmodel_depth=readplanetmodel(planetmodel_file,pmodeltype)
    Inputs
      planetmodel_file : full path to planet model (wavelength,r/R*)
      pmodeltype : type of mode
        1 : space seperated - wavelength(A) R/R*
        2 : CSV - wavelength(A),transit-depth(ppm)

    Outputs
      planetmodel_wv : array with model wavelengths (A)
      planetmodel_depth : array with model r/R* values.
    """
    planetmodel_wv=[]
    planetmodel_depth=[]
    f = open(planetmodel_file,'r')
    for line in f:
        
        if pmodeltype==2:
            line = line.strip() #get rid of \n at the end of the line
            columns = line.split(',') #break into columns with comma
            if is_number(columns[0]): #ignore lines that start with '#' 
                planetmodel_wv.append(float(columns[0])*10000.0) #wavelength (um -> A)   
                tdepth=np.abs(float(columns[1]))/1.0e6 #transit depth ppm -> relative
                planetmodel_depth.append(np.sqrt(tdepth)) #transit depth- > r/R*

        elif pmodeltype==1:
            line = line.strip() #get rid of \n at the end of the line
            columns = line.split() #break into columns with comma
            if is_number(columns[0]): #ignore lines that start with '#' 
                planetmodel_wv.append(float(columns[0])) #wavelength (A)   
                planetmodel_depth.append(float(columns[1])) #r/R*
            
    f.close()

    planetmodel_wv=np.array(planetmodel_wv)       #convert to numpy array
    planetmodel_depth=np.array(planetmodel_depth)
    
    return planetmodel_wv,planetmodel_depth;

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def p2w(p,noversample,ntrace):
    """Usage: w=p2w(p,noversample,ntrace) Converts x-pixel (p) to wavelength (w)
    Inputs:
      p : pixel value along dispersion axis (float) on oversampled grid.
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3 

    Outputs:
      w : wavelength (A)
    """
    
    #co-efficients for polynomial that define the trace-position
    nc=5 #number of co-efficients
    c=[[2.60188,-0.000984839,3.09333e-08,-4.19166e-11,1.66371e-14],\
     [1.30816,-0.000480837,-5.21539e-09,8.11258e-12,5.77072e-16],\
     [0.880545,-0.000311876,8.17443e-11,0.0,0.0]]
    
    pix=p/noversample
    w=c[ntrace-1][0]
    for i in range(1,nc):
        #print(w)
        w+=np.power(pix,i)*c[ntrace-1][i]
    w*=10000.0 #um to A
                  
    return w

def w2p(w,noversample,ntrace):
    """Usage: p=w2p(w,noversample,ntrace) Converts wavelength (w) to x-pixel (p)
    Inputs:
      w : wavelength (A)
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3 

    Outputs:
      p : pixel value along dispersion axis (float) on oversampled grid.

    """



    nc=5
    
    c=[[2957.38,-1678.19,526.903,-183.545,23.4633],\
       [3040.35,-2891.28,682.155,-189.996,0.0],\
       [2825.46,-3211.14,2.69446,0.0,0.0]]
    
    wum=w/10000.0 #A->um
    p=c[ntrace-1][0]
    for i in range(1,nc):
        #print(p)
        p+=np.power(wum,i)*c[ntrace-1][i]
    p=p*noversample
                  
    return p

def ptrace(px,noversample,ntrace):
    """given x-pixel, return y-position based on trace
    Usage:
    py = ptrace(px,noversample,ntrace)

    Inputs:
      px : pixel on dispersion axis (float) on oversampled grid.
      noversample : oversampling factor (integer 10 >= 1)
      ntrace : order n=1,2,3 

    Outputs:
      py : pixel on spatial axis (float) on oversampled grid.
    """
    nc=5 #number of co-efficients
    c=[[275.685,0.0587943,-0.000109117,1.06605e-7,-3.87e-11],\
      [254.109,-0.00121072,-1.84106e-05,4.81603e-09,-2.14646e-11],\
      [203.104,-0.0483124,-4.79001e-05,0.0,0.0]]
    
    opx=px/noversample #account for oversampling
    
    ptrace=c[ntrace-1][0]
    for i in range(1,nc):
        #print(w)
        ptrace+=np.power(opx,i)*c[ntrace-1][i]
        
    ptrace=ptrace-128
    return ptrace;


def addflux2pix(px,py,pixels,fmod):
    """Usage: pixels=addflux2pix(px,py,pixels,fmod)

    Drizel Flux onto Pixels using a square PSF of pixel size unity
    px,py are the pixel position (integers)
    fmod is the flux calculated for (px,py) pixel
        and it has the same length as px and py
    pixels is the image.
    """

    xmax = pixels.shape[0] #Size of pixel array
    ymax = pixels.shape[1]

    pxmh = px-0.5 #location of reference corner of PSF square
    pymh = py-0.5

    dx = np.floor(px+0.5)-pxmh
    dy = np.floor(py+0.5)-pymh

    # Supposing right-left as x axis and up-down as y axis:
    # Lower left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)
    
    #print('n',npx,npy)
    
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*dy

    #Same operations are done for the 3 pixels other neighbouring pixels

    # Lower right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*dy

    # Upper left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*(1.0-dy)

    # Upper right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*(1.0-dy)

    return pixels;

def transitmodel (sol,time,ld1,ld2,ld3,ld4,rdr,tarray,\
    itime=-1, ntt=0, tobs=0, omc=0, dtype=0 ): 
    "read in transitmodel solution"  
    nplanet=int((len(sol)-8)/10) #number of planets
    
    if type(itime) is float :
        if itime < 0 :
            itime=np.ones(len(time))*0.020434
        else:
            itime=np.ones(len(time))*float(itime)
    
    if type(ntt) is int :
        nttin=  np.zeros(nplanet, dtype="int32") #number of TTVs measured 
        tobsin= np.zeros(shape=(nplanet,len(time))) #time stamps of TTV measurements (days)
        omcin=  np.zeros(shape=(nplanet,len(time))) #TTV measurements (O-C) (days)
    else:
        nttin=ntt
        tobsin=tobs
        omcin=omc
    
    if type(dtype) is int :
        dtypein=np.ones(len(time), dtype="int32")*int(dtype) #contains data type, 0-photometry,1=RV data

    tmodel= np.zeros(len(time)) #contains the transit model
    tfit5.transitmodel(nplanet,sol,time,itime,nttin,tobsin,omcin,tmodel,dtypein,\
        ld1,ld2,ld3,ld4,rdr,tarray)
    return tmodel;

def get_dw(starmodel_wv,planetmodel_wv,norder,pars):
    norder=1 #Order to use.

    #get spectral resolution of star spectra
    nstarmodel=len(starmodel_wv)
    dw_star_array=np.zeros(nstarmodel-1)
    sortidx=np.argsort(starmodel_wv)
    for i in range(nstarmodel-1):
        dw_star_array[i]=starmodel_wv[sortidx[i+1]]-starmodel_wv[sortidx[i]]
    dw_star=np.max(dw_star_array)
    #print('dw_star',dw_star)

    #get spectral resolution of planet spectra
    nplanetmodel=len(planetmodel_wv)
    dw_planet_array=np.zeros(nplanetmodel-1)
    sortidx=np.argsort(planetmodel_wv)
    for i in range(nplanetmodel-1):
        dw_planet_array[i]=planetmodel_wv[sortidx[i+1]]-planetmodel_wv[sortidx[i]]
    dw_planet=np.max(dw_planet_array)
    #print('dw_planet',dw_planet)    

    #get spectra resolution needed to populate grid.
    xmax=pars.xout*pars.noversample
    dw_grid_array=np.zeros(xmax-1)
    for i in range(xmax-1):
        dw_grid_array[i]=p2w(i+1,pars.noversample,norder)-p2w(i,pars.noversample,norder)
    dw_grid=np.abs(np.min(dw_grid_array))
    #print('dw_grid',dw_grid)

    dw=np.max((dw_star,dw_planet))

    if dw>dw_grid:
        print("Warning. stellar/planet model spectral resolution is too low.  Data will be interpolated.")
        dw=np.min((dw,dw_grid))
        dwflag=1
    else: #bin data to common grid.
        dw
        dwflag=0

    #print('dw',dw)
    
    return dw,dwflag



