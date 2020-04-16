import numpy as np #numpy gives us better array management 
import os #checking status of requested files
from astropy.io import fits #astropy modules for FITS IO

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
                    elif command[0:nlcom-1] == 'es':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+4]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'ec':
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
                    elif command[0:nlcom-1] == 'al':
                        npl=int(np.float(command[nlcom-1]))
                        if (npl <= pars.nplanetmax) &(npl>0):
                            nplanet=np.max((nplanet,npl))
                            pars.sol[8*(npl-1)+7]=np.float(columns[1])
                        else:
                            print('Error: Planet number is Invalid ',npl)
                            print('Linenumber: ',linenumber)
                    elif command[0:nlcom-1] == 'el':
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

def readresponse(response_file):
    "Usage: ld,res1,res2,res3=readresponse(response_file)"
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
    "Usage: starmodel_wv,starmodel_flux=readstarmodel(starmodel_file,smodeltype)"
    starmodel_wv=[]
    starmodel_flux=[]

    if nmodeltype==2:

        f = open(starmodel_file,'r')
        for line in f:
            line = line.strip() #get rid of \n at the end of the line
            columns = line.split() #break into columns with space delimiter
            starmodel_wv.append(float(columns[0])*10)
            flux=-float(columns[5])*np.pi*(42.0*float(columns[1])+70.0*float(columns[2])\
                    +90.0*float(columns[3])+105.0*float(columns[4])-210.0)/210.0
            starmodel_flux.append(np.max([0.0,flux]))
        f.close()

        starmodel_wv=np.array(starmodel_wv)
        starmodel_flux=np.array(starmodel_flux)

    else:
        print('Currently on ATLAS-9 models are supported (nmodeltype=2)')

    return starmodel_wv,starmodel_flux;
