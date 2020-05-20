subroutine tmodel(nmodel,fmod,rprs,nll,b)
use precision
implicit none
!import vars
integer :: nmodel
real(double) :: b
real(double), dimension(:) :: fmod,rprs
real(double), dimension(:,:) :: nll
!local vars
integer :: i
real(double) :: tflux, mulimbf(1,5)

do i=1,nmodel
   call occultsmall(RpRs(i),nll(i,1),nll(i,2),nll(i,3),nll(i,4),1,b,tflux)
!   call occultnl(RpRs(i),nll(i,1),nll(i,2),nll(i,3),nll(i,4),b,tflux,mulimbf,1)
   fmod(i)=fmod(i)*tflux !modify flux with rprs and limb-darkening
!   write(0,*) "tflux: ",tflux
!   read(5,*)
enddo

end subroutine tmodel
