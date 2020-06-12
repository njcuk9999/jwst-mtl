!Old code that isn't used anymore.. only analytic kernel
!
!Old code to use simple analytic Kernel
!
!nK=41*noversample
!allocate(Kernel(nK,nK))
!call makekernel(nK,Kernel,noversample) !generate a PSF
!do j=1,nK
!   write(6,501) (Kernel(i,j)*100.0,i=1,nK)
!enddo
!501 format (150000(F13.10,1X))
!read(5,*)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine makekernel(nK,Kernel,noversample)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Jason Rowe 2015 - jasonfrowe@gmail.com
!old analytic function for generating a simple Kernel
use precision
implicit none
integer nK,i,j,i1,noversample
real(double) :: di,b,sig,kSum,prosum,leftover,dnK
real(double), dimension(41) :: profile
real(double), dimension(nK,nK) :: Kernel
data profile /0.00,0.00,0.001,0.002,0.003,0.004,0.006,0.015,0.025,0.034,&
 0.047,0.058,0.053,0.045,0.045,0.041,0.034,0.033,0.035,0.035,0.033,     &
 0.032,0.033,0.038,0.055,0.067,0.060,0.046,0.042,0.026,0.018,0.012,     &
 0.006,0.005,0.004,0.003,0.002,0.001,0.00,0.00,0.00/

b=(nK-1)/2+1 !center of Gaussian to smudge profile
sig=4.0d0*dble(noversample) !width of Gaussian to smudge profile
dnK=dble(nK)
write(0,*) "b: ",b

do i=1,nk
   di=dble(i)
   do j=1,nk
      i1=int(41.0*dble(j)/dnK)
      if(i1.lt.1)then
         prosum=profile(1)
      elseif(i1+1.gt.nK)then
         prosum=profile(nK)
      else
         leftover=41.0*dble(j)/dnK-i1
         prosum=profile(i1)*(1.0d0-leftover)+profile(i1+1)*(leftover)
         prosum=prosum/2.0d0
      endif
!      write(0,*) prosum,profile(i1),i1
      Kernel(i,j)=prosum*exp(-(di-b)**2.0d0/sig)
   enddo
enddo

kSum=Sum(Kernel)
Kernel=Kernel/kSum !normalize kernel to conserve flux

return
end
