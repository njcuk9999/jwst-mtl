program spgenV2
!Version 2 of Spec-generator.  This version does time-series and get FITS formating correct. 
!generates GR700-SOSS spectrum with 3-order traces + PSF + response
use precision
use response
!use response
implicit none
integer, dimension(3) :: funit !number of FITS I/O
integer, dimension(3) :: firstpix(3) !used for multiple writes to FITS
character(200), dimension(3) :: fileout !name of FITS files
!file name vars
integer :: pid,onum,vnum,gnum,spseq,anumb,enum,enumos,maxint
character(8) :: detectorname,prodtype
!random number vars
integer, dimension(3) :: now
integer :: seed
real(double) :: ran2,dumr
!Kernel vars
integer :: nrK,nKs
real(double), dimension(:,:,:), allocatable :: rKernel
!orders : traces and responces
integer :: ntracemax,ntrace
integer, dimension(2) :: ybounds
real(double) :: w2p,p2w,ptrace
real(double), allocatable, dimension(:) :: yres1,yres2,yres3
!spectral model parameters
integer :: nmodel !number of model points read in
integer :: nmodelmax !guess for size of array (will be auto increased if too small)
real(double) :: saturation !saturation of ADC.
real(double) :: vsini !rotational broadening of spectrum
real(double) :: drizzlefac,w1,w2
!next line of variabiles contain wavelength, flux info
real(double), dimension(:), allocatable :: wmod,fmod,wmod2,fmod2,fmod_not 
real(double), dimension(:,:), allocatable :: nll,nll2 !limb-darkening co-efficients.
character(80) :: modelfile
!planet model parameters
integer,parameter :: nplanetmax=9
integer :: nplanet
real(double), dimension(:), allocatable :: rprs
character(80) :: pmodelfile(nplanetmax),emisfile(nplanetmax),ttvfile(nplanetmax)
!orbital model parameters
real(double) :: tstart,tend,exptime,rhostar,T0,Per,esinw,ecosw,orbmodel,bt, &
   deadtime,dt,time
real(double) :: sol(nplanetmax*8+1)
!resampled spectral model
integer :: npt,nmodel_bin
real(double) :: dnpt
real(double), dimension(:), allocatable :: wmod_bin, fmod_bin
!image model arrays
integer :: xmax,ymax !size of oversampled grid
integer :: npx,npy
real(double) :: dxmaxp1,dymaxp1,px,py,awmod,respond,fmodres
real(double), dimension(:,:), allocatable :: pixels,wpixels,cpixels,wcpixels
!jitter model variables
real(double) :: xcoo,ycoo,roll,xcen,ycen,xjit,yjit,rolljit
!results that go into FITS files
integer :: xout, yout, ngroup, nint, maxnint,maxnintos,nint1,nint1os
real(double) :: dnossq
real(double), dimension(:,:), allocatable :: opixels
!displayfits
real(double) :: bpix,tavg,sigscale
!local vars
integer :: i,j,ii,jj,jjos !counters
integer :: noversample,nunit,filestatus,nmodeltype,iargc,iflag,nover
real(double) :: rvstar,b
character(80) :: cline,paramfile !used to readin commandline parameters
!temp vars
real(double) :: b1,b2

interface
   subroutine writefitsphdu(fileout,funit)
      use precision
     	implicit none
      integer :: funit
      character(200), dimension(3) :: fileout
   end subroutine writefitsphdu
   subroutine readmodel(nunit,nmodelmax,nmodel,wmod,fmod,iflag)
      use precision
      implicit none
      integer, intent(inout) :: nunit,nmodelmax,nmodel,iflag
      real(double), dimension(:), intent(inout) :: wmod, fmod
   end subroutine
   subroutine readatlas(nunit,nmodelmax,nmodel,wmod,fmod,nll,iflag)
      use precision
      implicit none
      integer, intent(inout) :: nunit,nmodelmax,nmodel,iflag
      real(double), dimension(:), intent(inout) :: wmod, fmod
      real(double), dimension(:,:), intent(inout) :: nll
   end subroutine
   subroutine readKernels(nrK,nK,rKernel,noversample)
      use precision
      implicit none
      integer,intent(inout) :: nrK,nK,noversample
      real(double), dimension(:,:,:), intent(inout) :: rKernel
   end subroutine
   subroutine readpmodel(nunit,nmodel,wmod,rprs)
      use precision
      implicit none
      integer :: nunit,nmodel
      real(double), dimension(:) :: rprs,wmod
   end subroutine
   subroutine tmodel(nmodel,fmod,rprs,nll,b)
      use precision
      implicit none
      integer, intent(inout) :: nmodel
      real(double), intent(inout) :: b
      real(double), dimension(:), intent(inout) :: fmod,rprs
      real(double), dimension(:,:), intent(inout) :: nll
   end subroutine
   subroutine binmodel(npt,wv,nmodel,wmod,fmod,fmodbin,rv)
      use precision
      implicit none
      integer, intent(inout) :: npt,nmodel
      real(double), intent(inout) :: rv
      real(double), dimension(:), intent(inout) :: wv,wmod,fmod
      real(double), dimension(:), intent(inout) :: fmodbin
   end subroutine
   subroutine writefitsdata(funit,xout,yout,pixels,ngroup,nint,nover,firstpix)
      use precision
     implicit none
     integer :: funit,xout,yout,ngroup,nint,nover,firstpix
     real(double), dimension(:,:) :: pixels
   end subroutine writefitsdata
   subroutine displayfits(nxmax,nymax,parray,bpix,tavg,sigscale)
      use precision
      implicit none
      integer, intent(inout) :: nxmax,nymax
      real(double), dimension(:,:), intent(inout) :: parray
      real(double), intent(inout) :: bpix,tavg
      real(double), intent(in) :: sigscale
   end subroutine displayfits
end interface

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Command line arguments
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
if(iargc().lt.1)then
   write(0,*) "Usage: spgen <configfile>"
   write(0,*) "   <configfile> - Configuration file."
   stop
endif


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!  Model Parameters 
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!open file
call getarg(1,paramfile)
nunit=10 !unit number for data spectrum
open(unit=nunit,file=paramfile,iostat=filestatus,status='old')
if(filestatus>0)then !trap missing file errors
   write(0,*) "Cannot open ",paramfile
   stop
endif
call readparameters(nunit,tstart,tend,exptime,deadtime,modelfile, &
   nmodeltype,rvstar,vsini,pmodelfile,emisfile,ttvfile,nplanet,nplanetmax,sol,&
   xout,yout,xcoo,ycoo,roll,xcen,ycen,xjit,yjit,rolljit,noversample,saturation,&
   ngroup, pid,onum,vnum,gnum,spseq,anumb,enum,enumos,detectorname,prodtype)
close(nunit)
!close file

!dealing with limits on FITSIO for buffered output with kind=4 integers
maxint=huge(firstpix(1))
!write(0,*) "Largest Integer ",maxint

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
! Random Number Initialization
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC!Initialization of random number
call itime(now)
seed=abs(now(3)+now(1)*now(2)+now(1)*now(3)+now(2)*now(3)*100)
dumr=ran2(-seed)

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!read in Kernels
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
nrK=30 !number of Kernels to readin
nKs=64*noversample !natural size of Kernels times oversampling
allocate(rKernel(nrK,nKs,nKs))
call readKernels(nrK,nKs,rKernel,noversample)

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!read in response
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
ntracemax=3 !number of traces used
call readresponse() !responce is returned via pointers from response.mod
!use global variables: nres,ld,res1,res2,res3 for response function
allocate(yres1(nres),yres2(nres),yres3(nres))
call spline(ld,res1,nres,1.d30,1.d30,yres1) !set up cubic splines
call spline(ld,res2,nres,1.d30,1.d30,yres2)
call spline(ld,res3,nres,1.d30,1.d30,yres3)

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!read in a model spectrum
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
nunit=11 !unit number for data spectrum
open(unit=nunit,file=modelfile,iostat=filestatus,status='old')
if(filestatus>0)then !trap missing file errors
   write(0,*) "Cannot open ",modelfile
   stop
endif

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Read in spectral model for star
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

nmodelmax=2000000 !initital guess at the number of data points
allocate(wmod(nmodelmax),fmod(nmodelmax))
allocate(nll(nmodelmax,4)) !limb-darkening co-efficients

iflag=0 !flag traces data i/o
nmodel=0   !initalize counter for number of data points
do !we do a loop.  If there are memory errors, we can get more this way
   if(nmodeltype.eq.1)then
      call readmodel(nunit,nmodelmax,nmodel,wmod,fmod,iflag) !read in spectrum
      nll=0.0d0 !no limb-darkening
   elseif(nmodeltype.eq.2)then
      call readatlas(nunit,nmodelmax,nmodel,wmod,fmod,nll,iflag)
   endif
   if(iflag.eq.1) then !reallocate array space (we ran out of space)
      allocate(wmod2(nmodelmax),fmod2(nmodelmax),nll2(nmodelmax,4)) !allocate temp arrays
      wmod2=wmod   !copy over the data we read
      fmod2=fmod
      nll2=nll
      deallocate(wmod,fmod,nll) !deallocate data arrays
      nmodelmax=nmodelmax*2 !lets get more memory
      write(0,*) "warning, increasing nmodelmax: ",nmodelmax
      allocate(wmod(nmodelmax),fmod(nmodelmax),nll(nmodelmax,4)) !reallocate array
      do i=1,nmodelmax/2  !copy data back into data arrays
         wmod(i)=wmod2(i)
         fmod(i)=fmod2(i)
         do j=1,4
            nll(i,j)=nll2(i,j)
         enddo
      enddo
      deallocate(wmod2,fmod2,nll2) !deallocate temp arrays
      iflag=2  !set flag that we are continuing to read in data
      cycle !repeat data read loop
   endif
   exit !successively break from data read loop.
enddo
close(nunit) !close file.
write(0,*) "Number of star model points: ",nmodel  !report number of data points read.

fmod=fmod/maxval(fmod(1:nmodel))*saturation !scale input flux
!write(0,*) "fbounds: ",minval(fmod(1:nmodel)),maxval(fmod(1:nmodel))

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Read in planet model
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


!Now we readin the planet model and interpolate onto the spectral model grid
allocate(rprs(nmodel))
rprs=0.0d0 !initialize to zero.

if(nplanet.gt.0)then
   !read in a planet model spectrum.  
   !the planet model is resampled on the stellar wavelength grid.  
   !wmod is used as input and it not changed on output. 
   nunit=11 !unit number for data spectrum
   open(unit=nunit,file=pmodelfile(1),iostat=filestatus,status='old')
   if(filestatus>0)then !trap missing file errors
      write(0,*) "Cannot open ",pmodelfile(1)
      stop
   endif
   call readpmodel(nunit,nmodel,wmod,rprs)
   close(nunit)
endif

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!initializing arrays for pixel values
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
xmax=xout*noversample !size of over sampled detector array.
ymax=yout*noversample
!array to hold detector array values
allocate(pixels(xmax,ymax),wpixels(xmax,ymax))
dxmaxp1=dble(xmax+1) !xmax+1 converted to double
dymaxp1=dble(ymax+1) !ymax+1 converted to double
!allocate array for convolved image
allocate(cpixels(xmax,ymax),wcpixels(xmax,ymax))
allocate(opixels(xout,yout)) !array for native grid output

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Define resampled grid
!if full range is not covered, then model will be extrapolated. 
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
npt=200000 !this sets the number of spectral points when resampled
allocate(wmod_bin(npt),fmod_bin(npt))
!resample model spectra on a uniform grid from 1000-40000 A
dnpt=dble(npt)
do i=1,npt
   wmod_bin(i)=1000.0+(40000.0-1000.0)/dnpt*dble(i) !make a wavelength grid
enddo

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
! Scale stellar flux to get approx 2e16 counts after drizzle
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
w1=p2w(dble(1),1,1)
w2=p2w(dble(xmax),1,1)
drizzlefac=dble(npt)/39000.0*abs(w2-w1)
write(0,*) 'drizzlefac: ',drizzlefac,w1,w2

allocate(fmod_not(nmodelmax))
fmod_not=fmod !make a copy of the star-flux input that is transit free.

!estimate nint
dt=exptime+deadtime
nint=int((tend-tstart)/dt)+1  !expected number of integrations
time=tstart

!initialize firstpix to 1.
firstpix=1 !initalize firstpix to 1 for all FITS files. 

!determine if we need to create new 
maxnint=int(dble(maxint)/(dble(xout)*dble(yout)*dble(ngroup)))
maxnintos=int(dble(maxint)/(dble(noversample*xout)*dble(noversample*yout)*dble(ngroup)))
if (maxnint.le.0) then
   write(0,*) "Error: Image dimensions are too large for kind=4 arrays"
   stop
endif
if (maxnint.le.0) then
   write(0,*) "Error: Image dimensions are too large for kind=4 arrays"
   write(0,*) "Oversampling must be less than: ",int(sqrt(dble(maxint)/(dble(xout)*dble(yout)*dble(ngroup))))
   stop
endif
!write(0,*) 'maxnint: ',maxnint,maxnintos

jj=1 !counts number of nint in current buffer
jjos=1
nint1=min(nint,maxnint)
nint1os=min(nint,maxnintos)
write(0,*) "NINT to be executed: ",nint,nint1,nint1os
!!!!!! Good place to start a loop for different impact parameters
do ii=1,nint

   if(jj.gt.maxnint)then !need to make new FITS file
      jj=1
      nint1=min(maxnint,nint-ii+1)
      call closefits(funit(1))
      call closefits(funit(2))
      enum=enum+1
      write(0,*) "New FITS enum =", enum
   endif

   if(jjos.gt.maxnintos)then !need to make new FITS file for oversampled sims
      jjos=1
      nint1os=min(maxnintos,nint-ii+1)
      call closefits(funit(3))
      enumos=enumos+1
      write(0,*) "New FITS enum =", enumos
   endif


   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !create FITS files and insert primary HDU for each output data product. 
   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   ! 1 - simulation with no convolution,  Native resolution
   ! 2 - simulation with convolution, native resolution
   ! 3 - simulation with convolution, over-sampled resolution
   if(jj.eq.1)then
      firstpix(1)=1
      firstpix(2)=1
      call getfilename(pid,onum,vnum,gnum,spseq,anumb,enum,enumos,detectorname,prodtype,fileout)
      call writefitsphdu(fileout(1),funit(1))
      call writefitsphdu(fileout(2),funit(2))
   endif
   if(jjos.eq.1)then
      firstpix(3)=1
      call writefitsphdu(fileout(3),funit(3))
   endif

   if(nplanet.gt.0)then
      bt=orbmodel(time,sol)
   else
      bt=2.0
   endif
   write(0,*) "Step #: ",ii,time,bt

   fmod=fmod_not !copy star-flux only model into fmod array.
   pixels=0.0d0 !initialize array to zero
   cpixels=0.0d0 !initialize array to zero

   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !Set up transit model (rprs -> flux)
   ! The tmodel routine will modify fmod to include the planet transit.
   ! If there is no transit, then fmod is unmodified. 
   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   if(bt.lt.maxval(rprs(1:nmodel))+1.0d0)then !do a check that we have a transit
      call tmodel(nmodel,fmod,rprs,nll,bt)
   endif

   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !Resample model onto a uniform grid
   !if full range is not covered, then model will be extrapolated. 
   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !read(5,*)
   fmod_bin=0.0d0 !initialize array
   !resample with equal spacing.  resampled model is in wmod_bin and fmod_bin
   call binmodel(npt,wmod_bin,nmodel,wmod,fmod,fmod_bin,rvstar)
   !write(0,*) "Done binning model"
   !store number of binned model points in easy to read array
   nmodel_bin=npt

   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !lets fill in pixel values
   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   !Main loop that does drizzle and convolution for each spectral order.
   !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   do ntrace=1,ntracemax !loop over all traces
      write(0,*) "Trace # ",ntrace
      wpixels=0.0d0 !initialize array to zero
      !trace parts of array used for spectrum (speeds up convolution later)
      ybounds(1)=ymax !initalize to bad bounds for max/min search
      ybounds(2)=1
      do i=1,nmodel_bin
         px=w2p(wmod_bin(i),noversample,ntrace) !get x-pixel value
         !is the wavelength on the detector?
         if((px.gt.1.0d0).and.(px.lt.dxmaxp1))then
            py=ptrace(px,noversample,ntrace) !get y-pixel value
   !      write(0,*) px,py
            if((py.gt.1.0d0).and.(py.lt.dymaxp1))then !check y-pixel value
               npx=int(px) !convert pixel values to integer
               npy=int(py)
               !find extremes of CCD use to speed up later
               ybounds(1)=min(ybounds(1),npy)
               ybounds(2)=max(ybounds(2),npy)
   !           wpixels(npx,npy)=wpixels(npx,npy)+fmod(i) !add flux to pixel
   !           add in response.

   !           wmod is in A.. convert to nm for easy comparision
               awmod=wmod_bin(i)/10.0d0
               if((awmod.lt.500.0).or.(awmod.gt.5500.0))then
                  respond=0.0d0
               else
               ! cubic interpolation of response along traces
                  select case(ntrace)
                     case(1)
                        call splint(ld,res1,yres1,nres,awmod,respond)
                     case(2)
                        call splint(ld,res2,yres2,nres,awmod,respond)
                     case(3)
                        call splint(ld,res3,yres3,nres,awmod,respond)
                  end select
               endif
               !the max statement makes sure we don't add negative flux.
               fmodres=fmod_bin(i)*max(respond,0.0d0) !flux to add to pixel
               call addflux2pix(px,py,xmax,ymax,wpixels,fmodres)
            endif
         endif
      enddo

      !Now we convolve the narrow spectra with the PSF kernel
      write(0,*) minval(wpixels),maxval(wpixels)
      wcpixels=0.0d0
   !   call convolve(xmax,ymax,wpixels,nrK,nKs,rKernel,noversample,wcpixels,&
   !      ybounds,ntrace)
      call convolveft(xmax,ymax,wpixels,nrK,nKs,rKernel,noversample,wcpixels, &
         ybounds,ntrace)
      write(0,*) minval(wcpixels),maxval(wcpixels)

      !copy trace to master array for output
      pixels=pixels+wpixels !unconvolved image
      cpixels=cpixels+wcpixels !convolved image
   enddo
   
   opixels=0.0d0 !initalize the array
   dnossq=noversample*noversample
   do i=noversample,xmax,noversample
      do j=noversample,ymax,noversample
         opixels(i/noversample,j/noversample)=                             &
            Sum(pixels(i-noversample+1:i,j-noversample+1:j))
      enddo
   enddo

   !write out unconvolved file
   write(0,*) "Writing unconvolved data"
   nover=1
   call writefitsdata(funit(1),xout,yout,opixels,ngroup,nint1,nover,firstpix(1))

   opixels=0.0d0 !reinitialize the array
   dnossq=noversample*noversample
   do i=noversample,xmax,noversample  !resample (bin) the array.
      do j=noversample,ymax,noversample
         opixels(i/noversample,j/noversample)=                             &
            Sum(cpixels(i-noversample+1:i,j-noversample+1:j))
   !      opixels(i/noversample,j/noversample)=                             &
   !         opixels(i/noversample,j/noversample)/dnossq
      enddo
   enddo

   !!display fits file
   !!call pgopen('?')
   !call pgopen('/xserve')
   !!call pgopen('trace.ps/vcps')
   !call PGPAP (8.0 ,1.0) !use a square 8" across
   !call pgsubp(1,4)
   !bpix=1.0e30
   !tavg=0.0
   !sigscale=3.0
   !call pgpage()
   !call displayfits(xout,yout,opixels,bpix,tavg,sigscale)
   !call pgclos()

   !write out convolved file
   write(0,*) "Writing Convolved data"
   nover=1
   call writefitsdata(funit(2),xout,yout,opixels,ngroup,nint1,nover,firstpix(2))

   !write out oversampled grid.
   write(0,*) "Writing oversampled convolved data"
   nover=noversample
   call writefitsdata(funit(3),xmax,ymax,cpixels,ngroup,nint1os,nover,firstpix(3))

   !update time-step
   time=time+dt

   jj=jj+1 !increase counter for number of nint writen for current FITS file
   jjos=jjos+1 !increase counter for number of nint written for current FITS_os file

enddo !end main loop

deallocate(wcpixels,wpixels) !work arrays no longer needed
deallocate(wmod,fmod,pixels,cpixels,opixels)
deallocate(wmod_bin,fmod_bin)
deallocate(yres1,yres2,yres3) !free up memory space.

!close the FITS file
call closefits(funit(1))
call closefits(funit(2))
call closefits(funit(3))

end program spgenV2


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
function ptrace(px,noversample,ntrace)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
integer noversample,i,ntrace
integer, parameter :: nc=5
real(double) :: ptrace,px,opx
real(double), dimension(nc) :: c1,c2,c3,c
data c1/275.685,0.0587943,-0.000109117,1.06605d-7,-3.87d-11/
data c2/254.109,-0.00121072,-1.84106e-05,4.81603e-09,-2.14646e-11/
data c3/203.104,-0.0483124,-4.79001e-05,0.0,0.0/

select case(ntrace)
   case(1)
      c=c1
   case(2)
      c=c2
   case(3)
      c=c3
end select

opx=px/dble(noversample) !account for oversampling
ptrace=c(1)
do i=2,nc
   !polynomial fit to trace. Good to about 1-2 pix
   ptrace=ptrace+opx**dble(i-1)*c(i)
enddo
ptrace=(ptrace-80.0d0)*dble(noversample) !account for oversampling

return
end

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
function p2w(p,noversample,ntrace)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!convert pixel to wavelength
! n=1 polynomial fit
!output is in Angstroms
use precision
implicit none
integer :: i,noversample,ntrace
integer, parameter :: nc=5
real(double) :: pix,p,p2w
real(double), dimension(nc) :: c1,c2,c3,c
data c1/2.60188,-0.000984839,3.09333e-08,-4.19166e-11,1.66371e-14/
data c2/1.30816,-0.000480837,-5.21539e-09,8.11258e-12,5.77072e-16/
data c3/0.880545,-0.000311876,8.17443e-11,0.0,0.0/

select case(ntrace)
   case(1)
      c=c1
   case(2)
      c=c2
   case(3)
      c=c3
end select

pix=p/dble(noversample)
p2w=c(1)
do i=2,nc
   p2w=p2w+pix**dble(i-1)*c(i)
enddo
p2w=p2w*10000.0 !um->A

!write(0,*) p2w,pix,p
!read(5,*)

return
end

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
function w2p(w,noversample,ntrace)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!convert wavelength to pixel
! n=1 polynomial fit.
! input is in angtroms
use precision
implicit none
integer :: i,noversample,ntrace
integer, parameter :: nc=5
real(double) w2p,w,wum
real(double), dimension(nc) :: c1,c2,c3,c
data c1/2957.38,-1678.19,526.903,-183.545,23.4633/
data c2/3040.35,-2891.28,682.155,-189.996,0.0/
data c3/2825.46,-3211.14,2.69446,0.0,0.0/

select case(ntrace)
   case(1)
      c=c1
   case(2)
      c=c2
   case(3)
      c=c3
end select

wum=w/10000.0 !A->um
w2p=c(1)
do i=2,nc
   w2p=w2p+wum**dble(i-1)*c(i)  !polynomial fit to trace. Good to about 1-2 pix
enddo
w2p=w2p*dble(noversample) !account for oversampling
!write(0,*) w2p,w/10000.0d0
!read(5,*)

return
end
