!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine initdatacube(fileout,funit,naxesout)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
!import vars
integer :: funit
integer, dimension(3) :: naxesout
character(80) :: fileout
!local vars
integer status,blocksize,naxis,bitpix
logical simple,extend

status=0

!if file already exists.. delete it.
!call deletefile(fileout,status)

!get a unit number
call ftgiou(funit,status)

!Create the new empty FITS file.  The blocksize parameter is a
!historical artifact and the value is ignored by FITSIO.
blocksize=1
status=0
call ftinit(funit,fileout,blocksize,status)
if(status.ne.0)then
   write(0,*) "Status: ",status
   write(0,*) "Critial Error open FITS for writing"
   write(0,'(A80)') fileout
   stop
endif

!Initialize parameters about the FITS image.
!BITPIX = 16 means that the image pixels will consist of 16-bit
!integers.  The size of the image is given by the NAXES values.
!The EXTEND = TRUE parameter indicates that the FITS file
!may contain extensions following the primary array.
simple=.true.
bitpix=-32
naxis=3
extend=.true.
!Write the required header keywords to the file
call ftphpr(funit,simple,bitpix,naxis,naxesout,0,1,extend,status)
!write(0,*) "ftphpr: ",status

return
end subroutine initdatacube

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine writedatacube(naxesin,datain,funit,firstpix)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
!import vars
integer :: funit,firstpix
integer, dimension(2) :: naxesin
real(double), dimension(:,:) :: datain
!local vars
integer npixels,group,nbuf,i,j,nbuffer,status
real(double), allocatable, dimension(:) :: buffer


!Write the array to the FITS file.
npixels=naxesin(1)*naxesin(2)
group=1
nbuf=naxesin(1)
j=0

!write(0,*) "funit: ",naxesin

allocate(buffer(nbuf))
do while (npixels.gt.0)
!read in 1 column at a time
   nbuffer=min(nbuf,npixels)

   j=j+1
!find max and min values
   do i=1,nbuffer
      buffer(i)=datain(i,j)
   enddo

   status=0
   call ftpprd(funit,group,firstpix,nbuffer,buffer,status)
   if(status.ne.0)then
      write(0,*) "Status: ",status
      write(0,*) "Critial Error writing FITS"
      write(0,'(A80)') funit
      stop
   endif

!update pointers and counters
   npixels=npixels-nbuffer
   firstpix=firstpix+nbuffer

enddo

return
end subroutine writedatacube
