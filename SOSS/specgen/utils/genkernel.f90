subroutine genkernel(nrK,nKs,rKernel,Kernel,wl,wls,wle,dwl)
!Jason Rowe 2015 - jasonfrowe@gmail.com
!get a Kernel given a wavelength
!wl = wavelength in um
use precision
implicit none
integer nrK,nKs,nK,nk1,nk2,i,j,noversample,ii,jj,l,m,n,Ks
real(double) :: wl,wls,wle,rwln,wdif,mSum,dnover,x1,x2,y,dwl
real(double), dimension(:,:,:) :: rKernel
real(double), dimension(:,:) :: Kernel

!write(0,*) "wl:",wl

if(wl.le.wls)then
   Kernel=rKernel(1,:,:) !case for when wavelength is too short
elseif(wl.ge.wle)then
   Kernel=rKernel(nrK,:,:) !case for when wavelength is too long
else
   rwln=(wl-wls)/dwl+1.0d0  !starts at 0.5 um then increases in 0.1 increments
!   write(0,*) wl,rwln
   nk1=int(rwln) !indices of the closest two Kernels
   nk2=nk1+1
   wdif=rwln-dble(nk1) !fractional difference
   do i=1,nKs
      do j=1,nKs
         Kernel(i,j)=rKernel(nk1,i,j)*(1.0d0-wdif)+rKernel(nk2,i,j)*wdif
      enddo
   enddo
endif

return
end
