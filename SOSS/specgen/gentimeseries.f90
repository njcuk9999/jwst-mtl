program gentimeseries
use precision
implicit none
integer i,j,n1,n2,nobs,nwv,nunit,filestatus,seed
integer, dimension(3) :: now
real(double) :: wv,fl,dt,time,d2s,exptime,ran2,gasdev,dumr
real(double), allocatable, dimension(:) :: spwave
real(double), allocatable, dimension(:,:) :: spflux,spfluxerr
character(80) :: filename

n1=1   !filename number to start with
n2=333 !filename number to finish with

dt=16.473 !delta time between data points
exptime=dt  !integration time

d2s=86400.0 !number of seconds in a day.

nobs=n2-n1+1 !number of observations
nwv=2048     !number of wavelengths observed

!Initialization of random number
call itime(now)
seed=abs(now(3)+now(1)*now(2)+now(1)*now(3)+now(2)*now(3)*100)
dumr=ran2(-seed)

allocate(spwave(nwv),spflux(nwv,nobs),spfluxerr(nwv,nobs))

do i=n1,n2
   write(filename,500) "spgen_m_",i,".txt"
   500 format(A8,I5.5,A4)

   nunit=10
   open(unit=nunit,file=filename,iostat=filestatus,status='old')
   if(filestatus>0)then !trap missing file errors
      write(0,*) "Cannot open ",filename
      stop
   endif

   j=0
   do
      if(j.gt.nwv)then
         write(0,*) "Increase nwv to match data points"
         write(0,*) "nwv: ",nwv
         stop
      endif
      read(nunit,*,iostat=filestatus) wv,fl
      !if sucessfully read in file, then assign to array
      if(filestatus == 0) then
         j=j+1
         !assuming the wavelengths are the same of every file
         spwave(j)=wv
         !first indice is for wavelength, second is for flux
         spflux(j,i)=fl
      elseif(filestatus == -1) then
         exit  !successively break from data read loop.
      else
         write(0,*) "File Error!! Line:",j+1
         write(0,900) "iostat: ",filestatus
         900 format(A8,I3)
         stop
      endif
   enddo
   close(nunit) !close file
   !write(0,*) filename,j
enddo

!now we can write out the contents.
write(6,504) "#nwv",20!nwv  !number of bandpasses (spectral elements)
504 format(A4,1X,I8)
write(6,502) "#Wavelength ", (spwave(i),i=1,nwv)
502 format(A11,2048(1X,1PE17.10))

!assign uncertainty
spfluxerr=sqrt(spflux)/50.0
do i=1,nobs
   do j=1,nwv
      spflux(j,i)=spflux(j,i)+spfluxerr(j,i)*gasdev(seed)
   enddo
enddo

time=0.0
do i=1,nobs
   !write out time in days and un-normalized flux for each wavelength
   write(6,503) (time/d2s,spflux(j,i),spfluxerr(j,i),exptime,j=1,20)!nwv)
   503 format(10000(1PE17.10,1X))
   time=time+dt
enddo

end program gentimeseries
