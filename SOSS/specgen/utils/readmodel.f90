subroutine readmodel(nunit,nmax,npt,wv,flux,iflag)
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer :: npt,iflag,nunit,i,nmax,filestatus
real(double) :: DF
real(double), dimension(:) :: wv,flux

DF=-8.0d0 !Pheonix zero point

iflag=0
i=npt+1
do
   if(i.gt.nmax)then !ran out of array space
      npt=i-1
      iflag=1
      return
   endif
   read(nunit,500,iostat=filestatus) wv(i),flux(i) !wv = Angstrom
   flux(i)=10.0**(flux(i)+DF)
!   flux(i)=log10(flux(i))-DF
!   write(0,*) wv(i),flux(i)
!   read(5,*)
   500 format(1X,F12.4,E12.9)
   if(filestatus == 0) then
      i=i+1
      cycle
   elseif(filestatus == -1) then
      exit
   else
      write(0,*) "File Error!!"
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop
   endif
enddo
npt=i-1 !number of spectral lines read in

return
end subroutine readmodel

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine readatlas(nunit,nmax,npt,wv,flux,nll,iflag)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer :: npt,iflag,nunit,i,j,nmax,filestatus
real(double) :: Pi
real(double), dimension(:) :: wv,flux
real(double), dimension(:,:) :: nll

Pi=acos(-1.d0)

iflag=0
i=npt+1
do
   if(i.gt.nmax)then !ran out of array space
      npt=i-1
      iflag=1
      return
   endif
   read(nunit,*,iostat=filestatus) wv(i),(nll(i,j),j=1,4),flux(i)
   flux(i)=-flux(i)*pi*(42.0d0*nll(i,1)+70.0d0*nll(i,2)+90.0d0*nll(i,3)+&
     105.0d0*nll(i,4)-210.0d0)/210.0d0
   flux(i)=max(0.0d0,flux(i))
!   write(0,*) i,wv(i),flux(i)
!   read(5,*)
   if(filestatus == 0) then
      i=i+1
      cycle
   elseif(filestatus == -1) then
      exit
   else
      write(0,*) "File Error!!"
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop
   endif
enddo
npt=i-1 !number of spectral lines read in

return
end subroutine readatlas

