subroutine readpmodel(nunit,nmodel,wmod,rprs)
use precision
implicit none
integer :: nunit,nmodel
real(double), dimension(:) :: rprs,wmod
!local vars
integer :: npt,i,filestatus
real(double) :: minpwv,maxpwv
real(double), allocatable, dimension(:) :: pwv,pmod,y2

rprs=0.1188

!read in the model points.
allocate(pwv(nmodel),pmod(nmodel))
i=1
do
   if(i.gt.nmodel)then !ran out of array space
      write(0,*) "Critical Error: Planet model has higher resolution that Star model"
      stop
   endif
   read(nunit,*,iostat=filestatus) pwv(i),pmod(i) !Angstroms, RpRs
!   write(0,*) i,pwv(i),pmod(i)
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
npt=i-1
write(0,*) "Number of planet model points: ",npt

minpwv=minval(pwv(1:npt))
maxpwv=maxval(pwv(1:npt))

!resample the spectra to match the star spectra.
!try out a spline (this might be super slow)
allocate(y2(npt))
call spline(pwv,pmod,npt,1.d30,1.d30,y2)

do i=1,nmodel
   if(wmod(i).lt.minpwv)then
      rprs(i)=pmod(1)
   elseif(wmod(i).gt.maxpwv)then
      rprs(i)=pmod(npt)
   else
      call splint(pwv,pmod,y2,npt,wmod(i),rprs(i))
   endif
!   write(0,*) i,wmod(i),rprs(i)
!   read(5,*)
enddo

!rprs=0.1188

return
end subroutine readpmodel
