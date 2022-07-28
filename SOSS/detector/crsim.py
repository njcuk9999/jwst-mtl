import sys
import astropy.io.fits as pyfits
import numpy as np


#np.random.seed(13578) #so we have always the same results
#This needs to be taken out in case of MC simulations..

rootname = "CRs_MCD5.5_"
crarraysize = 21
crarraysize = 21

eventspf = 1000 #number of events in each file
nfiles = 10
tot_nevents = nfiles*eventspf
#number of simulated events available in the library

crsloaded = False
events = np.empty((tot_nevents, crarraysize, crarraysize))

def getCRs(nevents, refDir, ptype='SUNMIN'):
    if((ptype  != 'SUNMIN') and (ptype != 'SUNMAX') and (ptype != 'FLARES')):
       print ('ptype = ', ptype, ' is not supported... exiting')
       sys.exit()

    global crsloaded, events
    if(not crsloaded): 
        events = loadAllEvents(refDir=refDir, ptype='SUNMIN')

    #randomly select requested number of events
    icr = np.random.randint(0, tot_nevents, size=nevents)
    return events[icr,:,:]

def loadAllEvents(refDir, ptype='SUNMIN'):
    global crsloaded, events

    #load all events
    filelst = []    
    for j in range(nfiles-1):
        ind = 1+j
        filename = refDir+rootname+ptype+'_0'+str(ind)+'.fits'
        datain = pyfits.getdata(filename)
        st = j*eventspf
        en = (j+1)*eventspf
        events[st:en,:,:] = datain[0:eventspf,:,:]
            
    crsloaded=True    
    return events

#print(events)
#This old version of getCRs loads only the necessary number of events
#in sequence. It is faster, as it opens only the necessary files, but
#always loads the same events, rather than a random selection of evnts...
def old_getCRs(nevents, ptype='SUNMIN'):
    if((ptype  != 'SUNMIN') and (ptype != 'SUNMAX') and (ptype != 'FLARES')):
       print ('ptype = ', ptype, ' is not supported... exiting')
       sys.exit()

    #see how many files do I need to load (each file has 1,000 CR events)      
    nfiles = (nevents //eventspf) +1
    remainder = nevents % eventspf

    filelst = []
    events = np.empty((nevents, crarraysize, crarraysize))
    for j in range(nfiles):
        ind = 1+j
        filename = rootname+ptype+'_0'+str(ind)+'.fits'
        datain = pyfits.getdata(filename)
        #print filename
        if(j < nfiles-1):
            st = j*eventspf
            en = (j+1)*eventspf
            events[st:en,:,:] = datain[0:eventspf:,:,:]
            #print st, en
        else:
            st = j*eventspf
            en = st+remainder
            #print st, en
            events[st:en,:,:] = datain[0:remainder,:,:]
            

    return events
  
def add_ipc(data, kernel):
    """
    Add interpixel capacitance effects to the data. This is done by
    convolving the data with a kernel. The kernel is read in from the
    file specified by self.params['Reffiles']['ipc']. The core of this
    function was copied from the IPC convolution step in MIRAGE.
    Parameters
    ----------
    data : obj
        2d numpy ndarray containing the data to which the
        IPC effects will be added
    kernel : the 4D IPC kernel to apply from the reference file
    Returns
    -------
    returns : obj
        2d numpy ndarray of the modified data
    """
    output_data = np.copy(data)
    # Shape of the data, which may include reference pix
    shape = output_data.shape
    # Find the number of reference pixel rows and columns
    # in output_data
    left_columns = 4
    right_columns = 4
    bottom_rows = 4
    top_rows = 4 

    kshape = kernel.shape

    # These axes lengths exclude reference pixels, if there are any.
    ny = shape[-2] - (bottom_rows + top_rows)
    nx = shape[-1] - (left_columns + right_columns)

    # The temporary array temp is larger than the science part of
    # output_data by a border (set to zero) that's about half of the
    # kernel size, so the convolution can be done without checking for
    # out of bounds.
    # b_b, t_b, l_b, and r_b are the widths of the borders on the
    # bottom, top, left, and right, respectively.
    b_b = kshape[0] // 2
    t_b = kshape[0] - b_b - 1
    l_b = kshape[1] // 2
    r_b = kshape[1] - l_b - 1
    tny = ny + b_b + t_b
    yoff = bottom_rows           # offset in output_data
    tnx = nx + l_b + r_b
    xoff = left_columns          # offset in output_data

    # Copy the science portion (not the reference pixels) of
    # output_data to this temporary array, then make
    # subsequent changes in-place to output_data.
    temp = np.zeros((tny, tnx), dtype=output_data.dtype)
    temp[b_b:b_b + ny, l_b:l_b + nx] = \
        output_data[yoff:yoff + ny, xoff:xoff + nx].copy()

    # After setting this slice to zero, we'll incrementally add
    # to it.
    output_data[yoff:yoff + ny, xoff:xoff + nx] = 0.

    # 4-D IPC kernel.  Extract a subset of the kernel:
    # all of the first two axes, but only the portion
    # of the last two axes corresponding to the science
    # data (i.e. possibly a subarray,
    # and certainly excluding reference pixels).

    k_temp = np.zeros((kshape[0], kshape[1], tny, tnx),
                      dtype=kernel.dtype)
    k_temp[:, :, b_b:b_b + ny, l_b:l_b + nx] = \
        kernel[:, :, yoff:yoff + ny, xoff:xoff + nx]

    # In this section, `part` has shape (ny, nx), which is
    # smaller than `temp`.
    middle_j = kshape[0] // 2
    middle_i = kshape[1] // 2
    for j in range(kshape[0]):
        jstart = kshape[0] - j - 1
        for i in range(kshape[1]):
            if i == middle_i and j == middle_j:
                continue   # the middle pixel is done last
            istart = kshape[1] - i - 1
            # The slice of k_temp includes different pixels
            # for the first or second axes within each loop,
            # but the same slice for the last two axes.
            # The slice of temp (a copy of the science data)
            # includes a different offset for each loop.
            part = k_temp[j, i, b_b:b_b + ny, l_b:l_b + nx] * \
                temp[jstart:jstart + ny, istart:istart + nx]
            output_data[yoff:yoff + ny, xoff:xoff + nx] += part
    # Add the product for the middle pixel last.
    part = k_temp[middle_j, middle_i, b_b:b_b + ny, l_b:l_b + nx] * \
        temp[middle_j:middle_j + ny, middle_i:middle_i + nx]
    output_data[yoff:yoff + ny, xoff:xoff + nx] += part  

    return output_data

def invert_ipc_kernel(kern):
    """
    Invert the IPC kernel such that it goes from being used to remove
    IPC effects from data, to being used to add IPC effects to data,
    or vice versa.
    ----------
    # Parameters
    kern : obj
        numpy ndarray, either 2D or 4D, containing the kernel
    Returns
    -------
    returns : obj
        numpy ndarray containing iInverted" kernel
    """
    shape = kern.shape
    ys = 0
    ye = shape[-2]
    xs = 0
    xe = shape[-1]
    if shape[-1] == 2048:
        xs = 4
        xe = 2044
    if shape[-2] == 2048:
        ys = 4
        ye = 2044
    if len(shape) == 2:
        subkernel = kern[ys:ye, xs:xe]
    elif len(shape) == 4:
        subkernel = kern[:, :, ys:ye, xs:xe]

    dims = subkernel.shape
    # Force subkernel to be 4D to make the function cleaner
    # Dimensions are (kernely, kernelx, detectory, detectorx)
    if len(dims) == 2:
        subkernel = np.expand_dims(subkernel, axis=3)
        subkernel = np.expand_dims(subkernel, axis=4)
    dims = subkernel.shape

    delta = subkernel * 0.
    nyc = dims[0] // 2
    nxc = dims[1] // 2
    delta[nyc, nxc, :, :] = 1.

    a1 = np.fft.fft2(subkernel, axes=(0, 1))
    a2 = np.fft.fft2(delta, axes=(0, 1))
    aout = a2 / a1
    imout = np.fft.ifft2(aout, axes=(0, 1))
    imout1 = np.fft.fftshift(imout, axes=(0, 1))
    realout1 = np.real(imout1)

    # If the input kernel was 2D, make the output 2D
    # If the input was 4D and had reference pixels, then
    # surround the inverted kernel with reference pixels
    if len(shape) == 2:
        newkernel = realout1[:, :, 0, 0]
    elif len(shape) == 4:
        newkernel = np.copy(kern)
        newkernel[:, :, ys:ye, xs:xe] = realout1

    return newkernel



flux_ds={'SUNMIN':4.8983, 'SUNMAX':1.7783, 'FLARES':3046.83}
pxsize = 18e-4 #18 micron per pixel (in cm)

def addCRs(cube, tframe, refDir, ptype='SUNMIN', f_ADC=1.6):

    """   
    Adds the cosmic rays on detector

    ## Parameters
    **cube**: Cube on which we add the CRs
    **tframe**: Time it takes to take a frame
    **refDir**: Reference files directory
    **ptype**: Type of solar activity
    **f_ADC**: Gain

    """

    #read in IPC convolution kernel
    #ipcreffile='/mnt/jwstdata/crds_cache/references/jwst/niriss/jwst_niriss_ipc_0008.fits'
    ipcreffile=refDir+'jwst_niriss_ipc_0008.fits'
    kernel = pyfits.getdata(ipcreffile)
    kernel = invert_ipc_kernel(kernel)
    cube_tot = cube # The cube with the integrations
    crmask_tot = np.zeros(cube.shape, dtype='int32')

   

    #Add cosmic rays on all iterations 
    #numdim=len(cube.shape)
        
    #Get size of active pixel array for different FULL, SUBSCRIPT256 or SUBSCRIPT96
    cube=np.squeeze(cube_tot[0,:,:,:])

    if cube.shape[1] == 2048:
    
        l1 = cube.shape[2]-8  #There is 4 reference pixels around the full image
        l2 = cube.shape[1]-8
        #print('FULL, Active pixels are', l1, 'x', l2)
        
    elif cube.shape[1] == 512: 
        
        l1 = cube.shape[2]-8  #There is 4 reference pixels around the full image
        l2 = cube.shape[1]-4  #The bottom doesn't have reference pixels for SUBSCRIPT512
        #print('SUBSCRIPT512, Active pixels are', l1, 'x', l2)        
    
    elif cube.shape[1] == 256: 
        
        l1 = cube.shape[2]-8  #There is 4 reference pixels around the full image
        l2 = cube.shape[1]-4  #The bottom doesn't have reference pixels for SUBSCRIPT256
        #print('SUBSCRIPT256, Active pixels are', l1, 'x', l2)
    
    elif cube.shape[1] == 96: 
        
        l1 = cube.shape[2]-8  #There is 4 reference pixels around the full image
        l2 = cube.shape[1]  #The bottom and top don't have reference pixels for SUBSCRIPT96
        #print('SUBSCRIPT96, Active pixels are', l1, 'x', l2)
    
    cube=cube.astype('int32')
    #compute average number of events per frame
    dsarea = (pxsize*l1)*(pxsize*l2)
    #CJW: Removed factor of 8.1 from line below due to 20180430 email from Giovanna saying it was a fudge factor to get to IPC affected level of 12% per ks, but SUNMIN hit rate is 1.6% per ks before IPC convolution
    n_average_events = flux_ds[ptype] * dsarea * tframe 
    #n_average_events = 8.1 *flux_ds[ptype] * dsarea * tframe #Cranking up the flux for testing purpposes by factor 7.5 -> N hits after 1,000 sec ~ 12-13%
    ngroups = cube.shape[0]
   
    #print ('Adding events to integrations...')

    for integr in range(len(cube_tot)): 

        cube = np.squeeze(cube_tot[integr,:,:,:])     #The cube without the integrations
        crmask = np.squeeze(crmask_tot[integr,:,:,:])  #The mask without the integrations


        for i in range(ngroups):
            #make tmpmask including reference pixels
            
            if cube.shape[1] == 2048:
                tmpmask = np.zeros([(l2+8),(l1+8)], dtype='float')
                
            elif cube.shape[1] == 96: 
                tmpmask = np.zeros([(l2),(l1+8)], dtype='float')

            else: 
                tmpmask = np.zeros([(l2+4),(l1+8)], dtype='float') 
                
                
            nevents = np.random.poisson(n_average_events) #Chooses a random amount of events around the average
            

            #place cosmic rays randomly in active pixels only
            icoors = np.random.randint(0, l1, size=nevents)+4
            
            if cube.shape[1] == 2048:
                jcoors = np.random.randint(0, l2, size=nevents)+4
                
            else: 
                jcoors = np.random.randint(0, l2, size=nevents)

            crarray = getCRs(nevents, refDir=refDir, ptype=ptype)  #An array with a bunch of 21x21 pixel events
            #return an array with the requested number of CR events,
            #extracted from the CR library by M. Robberto (in crlib/)

            #limits when adding CR arrays 
            crars = int(crarraysize/2-0.5) #10
            crare = int(crarraysize/2+0.5) #11
            
            #each CR events is a 21x21 array which needs to be added to the
            #frame array at the point of the CR hit
            #for j in range(nevents):
            for j in range(nevents):
                x = icoors[j]
                y = jcoors[j]

                if cube.shape[1] == 96: 

                    #dealing with edges            
                    xs, xe = crars, crare
                    ys, ye = crars, crare
                    xars, xare = 0, crarraysize
                    yars, yare = 0, crarraysize
                    if(x < crars):
                        xs = x
                        xars = crars - x
                    if(y < crars):
                        ys = y
                        yars = crars - y
                    if(x > 4+l1-crare):
                        xe = 4+l1-x
                        xare = crars + (4+l1-x)
                    if(y > l2-crare):
                        ye = l2-y
                        yare = crars + (l2-y)
                    
                    array2add = f_ADC * crarray[j, yars:yare, xars:xare] #Array of the noise to add 
                    
                    
                    #recording event in mask
                    tmpmask[y-ys:y+ye,x-xs:x+xe] = tmpmask[y-ys:y+ye,x-xs:x+xe]+array2add   


                else: 
               
                    #dealing with edges            
                    xs, xe = crars, crare
                    ys, ye = crars, crare
                    xars, xare = 0, crarraysize
                    yars, yare = 0, crarraysize
                    if(x < crars):
                        xs = x
                        xars = crars - x
                    if(y < crars):
                        ys = y
                        yars = crars - y
                    if(x > 4+l1-crare):
                        xe = 4+l1-x
                        xare = crars + (4+l1-x)
                    if(y > 4+l2-crare):
                        ye = 4+l2-y
                        yare = crars + (4+l2-y)
                    
                    array2add = f_ADC * crarray[j, yars:yare, xars:xare] #Array of the noise to add 
                    
                    
                    #recording event in mask
                    tmpmask[y-ys:y+ye,x-xs:x+xe] = tmpmask[y-ys:y+ye,x-xs:x+xe]+array2add
            
                    
            #Set all ref pixels to zero in case the odd one got a partial cr
            #y-axis
            if tmpmask.shape[0] == 2048:
                tmpmask[:4,:]=0.0
                tmpmask[2044:,:]=0.0

            elif tmpmask.shape[0] == 256:
                tmpmask[252:,:]=0.0

            elif tmpmask.shape[0] == 512:
                tmpmask[508:,:]=0.0

            #x-axis
            tmpmask[:,:4]=0.0
            tmpmask[:,2044:]=0.0

            #Convolve group with IPC
            tmpmask=add_ipc(tmpmask, kernel)
            
            #Adding events to mask
            crmask[i:ngroups, :,:] = crmask[i:ngroups, :,:] + tmpmask
            
            #Adding events to group
            cube[i:ngroups,:,:] = cube[i:ngroups,:,:]+crmask[i,:,:]
                
        
        
        #need to take care of saturation here
        isat = np.where(cube > 65535)
        cube[isat] = 65535
        msat = np.where(crmask > 65535)
        crmask[msat] = 65535

        #casting to unsigned short (16 bits)
        crmaskout = crmask.astype('ushort')
        #del crmask
        cubeout = cube.astype('ushort')
        #Add dimension for N_Int=4
    
        #cubeout = np.expand_dims(cubeout, axis=0)
        #crmaskout = np.expand_dims(crmaskout, axis=0)
        
        #Add the integration to the output
        cubeout = np.expand_dims(cubeout, axis=0) #Reexpand the arrays to fit with integration
        crmaskout = np.expand_dims(crmaskout, axis=0)

        cube_tot[integr,:,:,:] = cubeout   
        crmask_tot[integr,:,:,:] = crmaskout
        #print ('Done adding cosmic rays for int', integr+1)

    #print('The number of groups per integration is:',ngroups)
    #print ('Average events per group : ',n_average_events)

    return [cube_tot, crmask_tot]


##########Here starts class & functions for `IRS2'###########

class IRS2Cube:
    
    def __init__(self, irs2cube):
        self.N=16
        self.R=4
        self.framesize = 2048
        self.cubein = irs2cube
        self.ngroups = self.cubein.shape[0]

        #Lists to keep track of the various coordinates intervals
        #to go irs2 frame <-> traditional frame
        self.intervalstr=[]
        self.intervalend=[]
        self.ltradstr=[]
        self.ltradend=[]

        #lists for keeping track of index of ref pixels (handy when looking for CR hits...)
        self.lirs2ref_pix=[] #IRS2 ref pix index in traditional frame coordinates
        self.lirs2ref_pix_irs2=[] #IRS2 ref pix index in IRS2 frame coordinates

        #Building interval lists (they are all the same for all the frmaes)
        nintervals = self.framesize/(self.N)

        ref_channel_sizex = 640
        st = ref_channel_sizex+0
        se = st + self.N/2
        tradstr = 0

        for k in range(0, nintervals):
            self.intervalstr.append(st)
            self.intervalend.append(se)
            self.ltradstr.append(tradstr)
            l = se-st
            tradend = tradstr+l
            self.ltradend.append(tradend)
            self.lirs2ref_pix.append(tradend-1) #x index of first colum of (irs2) ref pix in trad frame
            self.lirs2ref_pix.append(tradend) #x index of second colum of (irs2) ref pix in trad frame
            self.lirs2ref_pix_irs2.append(se) #x index of first colum of (irs2) ref pix in irs2 frame
            self.lirs2ref_pix_irs2.append(se+2) #x index of third colum of (irs2) ref pix in irs2 frame

            st = se+self.R
            se = st+self.N
            tradstr = tradstr+l
            

        #Adding the intervals for the last 8 illuminated pixels
        se = se-self.N/2
        l = se-st
        #print st, se, tradstr, l
        self.intervalstr.append(st)
        self.intervalend.append(se)
        self.ltradstr.append(tradstr)
        self.ltradend.append(tradstr+l)
        self.lirs2ref_pix.append(tradend-1) #x index of first colum of (irs2) ref pix in traditional array
        self.lirs2ref_pix.append(tradend) #x index of second colum of (irs2) ref pix in  traditional array
        #print self.intervalstr
        #print self.intervalend
        #print self.lirs2ref_pix
        
        #It will be handy to have a mask as well
        self.crmask=None
            
    def extractTradFrame(self, framein):
        tradframe = np.zeros([self.framesize, self.framesize], dtype='ushort')
       
        for k, st in enumerate(self.intervalstr):
            se = self.intervalend[k]
            tradstr = self.ltradstr[k]
            tradend = self.ltradend[k]
            tradframe[:,tradstr:tradend]=framein[:,st:se]

        return tradframe
        

    def extractTradCube(self):
        cubetrad = np.zeros([self.ngroups, self.framesize, self.framesize], dtype='ushort')
        
        print ("Constructing traditional cube")
        for k in range(self.ngroups):
            cubetrad[k,:,:] = self.extractTradFrame(self.cubein[k,:,:])

        return cubetrad

    def rebuildIRS2Frame(self, tradframe, iframe):
        print ("")
        print ('Rebuilding frame'), iframe
        irs2frame = np.zeros([self.cubein.shape[1],self.cubein.shape[2]], dtype='ushort')
        #Adding reference output
        irs2frame[:,0:640] = self.cubein[iframe, :, 0:640]
        #Write out traditional frame for testing purposes
        #pyfits.writeto('trad'+str(iframe)+'.fits',tradframe,None,clobber=True)

        #Copying content of tradional cube (with CRs) onto irs2frame
        for i, tradstr in enumerate(self.ltradstr):
            start = self.intervalstr[i]
            end =  self.intervalend[i]
            #print start, end, self.ltradstr[i], self.ltradend[i]
            irs2frame[:,start:end] = tradframe[:,self.ltradstr[i]:self.ltradend[i]]

        #### Taking care of reference pixels #######
        #1. Filling in original value of reference pixels irs2columns 
        for se in self.intervalend[0:len(self.intervalend)-1]:
            irs2frame[:,se:se+self.R] = self.cubein[iframe, :, se:se+self.R]

        
        #2. Getting mask to see whether any of the reference pixel (top cornice) has been hit - 
        ref_mask = self.crmask[iframe,0:4,:] 
        #NB Using mask at frame level (only the hits for that frame) will make sure that this 
        #Wrting out mask for  testing purpose
        #pyfits.writeto('ref_mask'+str(iframe)+'.fits',ref_mask,None,clobber=True)
        
        #3. Finding the hit in the ref-pix raw used by IRS2: ASSUMING ROW ADJ to ILLUMINATED PIXEL (ref_row=0)
        ref_row=0
        ihits = np.where(ref_mask[ref_row,:] > 0)
        #print "Ref pix hits"
        #print ihits
        #Establishing whether any of these are on a "IRS2 ref pix"
        for j in range(ihits[0].shape[0]):
            
            if(ihits[0][j] in self.lirs2ref_pix):
                ih = self.lirs2ref_pix.index(ihits[0][j])
                x_index_refhit = self.lirs2ref_pix[ih]
                val = ref_mask[ref_row, x_index_refhit]
                print ('IRS2 ref-pix hit index (trad coordinate): '+str(x_index_refhit)+' Val = '+str(val))
                print ("")
                #val_in_frame = tradframe[2044, index_refhit]
                #print 'Hit value = ', val, 'Value in frame', val_in_frame
                
                #4. Propagating value through frame & cube
                #i. Need to get an hit raw index
                y_index_refhit = self.getYindexRefHit()
                
                #ii. Propagating value through frame (2 columns, first one from y_index_refhit, the other entirely):
                ## NB need to know reading direction for that particular output...
                x_index_refhit_irs2 = self.lirs2ref_pix_irs2[ih]
                irs2frame[y_index_refhit:,x_index_refhit_irs2:x_index_refhit_irs2+2] = irs2frame[y_index_refhit:,x_index_refhit_irs2:x_index_refhit_irs2+2]+val

                #iii. Propagating value through cube, both two entire columns 
                #Note we are modifying the value of the irs2 read of the effected reference pixel
                self.cubein[iframe+1:,:,x_index_refhit_irs2:x_index_refhit_irs2+2] = self.cubein[iframe+1:,:,x_index_refhit_irs2:x_index_refhit_irs2+2]+val


        return irs2frame

    def rebuildIRS2Cube(self, trad_cube):
        cubeout = np.zeros(self.cubein.shape, dtype='ushort')

        print ("Rebuilding IRS2Cube")
        for k in range(self.ngroups):
            cubeout[k,:,:] = self.rebuildIRS2Frame(trad_cube[k,:,:], k)
            
        return cubeout

    def addCRsIRS2(self, tframe, ptype='SUNMIN', f_ADC=1):
      
    
        cubetrad = self.extractTradCube()
        #Add CR to traditional cube
        [cubecr, self.crmask] = addCRs(cubetrad, tframe, ptype='SUNMIN', f_ADC=1)
        cubeout = self.rebuildIRS2Cube(cubecr)
            
        return [cubeout, self.crmask]

    
    
    def getYindexRefHit(self):
        ref_row=0
        return np.random.randint(ref_row, 2048, size=1)[0]
