!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine convolve(xmax,ymax,pixels,nrK,nKs,rKernel,noversample,       &
   cpixels,ybounds,ntrace)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer :: xmax,ymax,nK,i,j,l,m,nKd2,ii,jj,ii1,ii2,jj1,jj2,nrK,nKs,     &
   noversample,ntrace
integer, dimension(2) :: ybounds
real(double) :: mSum,p2w,wl,p,wls,wle,dwl
real(double), dimension(xmax,ymax) :: pixels,cpixels
real(double), allocatable, dimension(:,:) :: subIm,subIm2,Kernel,wKernel
real(double), dimension(nrK,nKs,nKs) :: rKernel
character(80) :: line

interface
   subroutine genkernel(nrK,nKs,rKernel,Kernel,wl,wls,wle,dwl)
      use precision
      integer, intent(inout) :: nrK,nKs
      real(double), intent(inout) :: wl,wls,wle,dwl
      real(double), dimension(:,:),intent(inout) :: Kernel
      real(double), dimension(:,:,:), intent(inout) :: rKernel
   end subroutine
end interface

!write(0,*) "Convolution begins"

!adjust ybounds for Kernel width for convolution speed up
nKd2=nKs/2 !half size of kernel
ybounds(1)=max(1,ybounds(1)-nKd2)
ybounds(2)=min(ymax,ybounds(2)+nKd2)
write(0,*) 'ybounds: ',ybounds(1),ybounds(2)

cpixels=0.0d0 !initialize convoluted image to zero
allocate(subIm(nKs,nKs),subIm2(nKs,nKs)) !sub-Image stamps for multiplication

wls=0.5d0 !starting wavelength of Kernels
wle=3.4d0 !ending wavelengths of Kernels
dwl=0.1d0 !wavelength intervals
nK=nKs !Kernel should already match image scale
if(allocated(Kernel)) deallocate(Kernel)
allocate(Kernel(nK,nK))

do i=1,xmax

!  This part is where we generate the wavelength specific Kernel
   if(mod(i,1).eq.0)then !only need to update Kernel every pixel
      Kernel=0.0d0
      p=dble(i)
      wl=p2w(p,noversample,ntrace)/10000.0d0 !A -> um
!      write(0,*) "new Kernel wl: ",wl
      !get a wavelength specific Kernel
      call genkernel(nrK,nKs,rKernel,Kernel,wl,wls,wle,dwl)
      mSum=Sum(Kernel)  !make sure Kernel is normalized
      Kernel=Kernel/mSum
   endif

   write(line,502) "Convolution: ",xmax,i,wl
   502 format(A13,I5,1X,I5,1X,F6.3)
   call ovrwrt(line,2)
!   write(0,'(A80)') line
   do j=ybounds(1),ybounds(2) !only scan area with non-zero pixels
      subIm=0.0d0 !initialize subImage
      ii=0 !ii,jj mark sub-Image position
      do l=i-nkd2,i-nkd2+nK-1
         ii=ii+1
         jj=0
         do m=j-nkd2,j-nkd2+nK-1
            jj=jj+1
            !check that pixel location is valid
            if((l.gt.0).and.(l.le.xmax).and.(m.gt.0).and.(m.le.ymax))then
!               write(0,*) ii,jj,l,m,nK
               subIm(ii,jj)=pixels(l,m) !copy image value to stamp
            endif
         enddo
      enddo
      mSum=Sum(abs(subIm))

      if(mSum.gt.0.0e-30)then !if array is full of zeros, skip..
!      write(0,*) "multiply ",i,j
         subIm2=matmul(subIm,Kernel) !multiple subImage by Kernel
         ii=0
         do l=i-nkd2,i-nkd2+nK-1
            ii=ii+1
            jj=0
            do m=j-nkd2,j-nkd2+nK-1
               jj=jj+1
               if((l.gt.0).and.(l.le.xmax).and.(m.gt.0).and.(m.le.ymax))then
                  cpixels(l,m)=cpixels(l,m)+subIm2(ii,jj) !copy convolved stamp
               endif
            enddo
         enddo
      endif
   enddo
enddo
write(0,*) " " !clear next line of text output from ovrwrt usage above

return
end
