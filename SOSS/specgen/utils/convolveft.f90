subroutine convolveft(xmax,ymax,pixels,nrK,nKs,rKernel,noversample,       &
   cpixels,ybounds,ntrace)
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
!iso_c_binding is for FFTW3 interface
use, intrinsic :: iso_c_binding
implicit none
!add in the FFTW3 modules
include 'fftw3.f03'

integer :: xmax,ymax,nrK,nKs,noversample,ntrace,XF,YF,i
integer, dimension(2) :: ybounds
real(double) :: wl,wls,wle,dwl,p,wlp,p2w,fac
real(double), dimension(xmax,ymax) :: pixels,cpixels
real(double), dimension(nrK,nKs,nKs) :: rKernel
real(double), allocatable, dimension(:,:) :: Kernel,A,B,C
character(80) :: line

!FFTW3 vars
type(C_PTR) :: planA,planB,planC
integer ( kind = 4 ) :: nh
complex(C_DOUBLE_COMPLEX), allocatable, dimension(:,:) :: AC,BC,CC

interface
   subroutine genkernel(nrK,nKs,rKernel,Kernel,wl,wls,wle,dwl)
      use precision
      integer, intent(inout) :: nrK,nKs
      real(double), intent(inout) :: wl,wls,wle,dwl
      real(double), dimension(:,:),intent(inout) :: Kernel
      real(double), dimension(:,:,:), intent(inout) :: rKernel
   end subroutine
end interface

cpixels=0.0d0! initalize convolved image to zero.

wls=0.5d0 !starting wavelength of Kernels
wle=3.4d0 !ending wavelengths of Kernels
dwl=0.1d0 !wavelength intervals
allocate(Kernel(nKs,nKs))
XF=nKs+xmax !size of zero padded arrays for convolution
YF=nKs+ymax
allocate(A(XF,YF),B(XF,YF),C(XF,YF)) !arrays to apply FFT
nh=(XF/2)+1 !size for complex array
allocate(AC(nh,YF),BC(nh,YF),CC(nh,YF)) !allocate complex arrays for FT

!only need to FFT pixels once..
A=0.0d0 !initalize to zero
A(1:xmax,1:ymax)=pixels(1:xmax,1:ymax) !assign Image to zero-padded A
planA=fftw_plan_dft_r2c_2d(YF,XF,A,AC,FFTW_ESTIMATE)
call fftw_execute_dft_r2c(planA,A,AC)
!write(0,*) "ttt: ",minval(pixels),maxval(pixels)
!write(6,*) pixels
!read(5,*)
call fftw_destroy_plan(planA)

!We can precompute plans..
planB=fftw_plan_dft_r2c_2d(YF,XF,B,BC,FFTW_ESTIMATE)
planC=fftw_plan_dft_c2r_2d(YF,XF,CC,C,FFTW_ESTIMATE)

wl=wls
do while(wl.lt.wle)
   write(line,502) "Convolution: ",wl,wle
   502 format(A13,F6.3,1X,F6.3)
   call ovrwrt(line,2)
!   write(0,*) "wl: ",wl
!gets get a Kernel (wl is wavelength in um)
   call genkernel(nrK,nKs,rKernel,Kernel,wl,wls,wle,dwl)
!   Kernel=transpose(Kernel)
   B=0.0d0
   C=0.0d0
!  Kernel should be centered around 0,0 on B
   B(1:nKs/2      ,1:nKs/2)      =Kernel(nKs/2:nKs,nKs/2:nKs)
   B(XF-nKs/2+1:XF,YF-nKs/2+1:YF)=Kernel(1:nKs/2  ,1:nKs/2)
   B(XF-nKs/2+1:XF,1:nKs/2)      =Kernel(1:nKs/2  ,nKs/2:nKs)
   B(1:nKs/2,      YF-nKs/2+1:YF)=Kernel(nKs/2:nKs,1:nKs/2)

!   write(0,*) "Computing FFTs"
!   planB=fftw_plan_dft_r2c_2d(YF,XF,B,BC,FFTW_ESTIMATE)
   call fftw_execute_dft_r2c(planB,B,BC)
!   call fftw_destroy_plan(planB)
   !multiply
   CC=AC*BC
!   write(0,*) "Start iFFT"
!   planC=fftw_plan_dft_c2r_2d(YF,XF,CC,C,FFTW_ESTIMATE)
   call fftw_execute_dft_c2r(planC,CC,C)
!   call fftw_destroy_plan(planC)
   do i=1,xmax
      p=dble(i)
      wlp=p2w(p,noversample,ntrace)/10000.0d0

      !This part of the code does the linear interpolation.  There is
      !probably a problem at the edges for parts of the spectrum
      !beyond wls and wle which should have fac=1 rather than decaying

!      if(abs(wlp-wl).gt.dwl)then
!         if((wlp.ge.wle).or.(wlp.lt.wls))then
!            fac=1.0
!         else
!            fac=0.0
!         endif
!      else
!         fac=1.0d0-abs(wlp-wl)/dwl
!      endif
      fac=max(0.0,1.0d0-abs(wlp-wl)/dwl)
!      write(0,*) "t: ",minval(C(i,1:ymax)/dble(xmax*ymax)*fac),maxval(C(i,1:ymax)/dble(xmax*ymax)*fac)
!      write(0,*) "tt: ",minval(C(i,1:ymax)),maxval(C(i,1:ymax))
      cpixels(i,1:ymax)=cpixels(i,1:ymax)+C(i,1:ymax)/dble(xmax*ymax)*fac
   enddo
   wl=wl+dwl
enddo
call fftw_destroy_plan(planB)
call fftw_destroy_plan(planC)

return
end subroutine convolveft










