program makecube
use precision
implicit none
integer iargc,nunit,filestatus,i,xmax,ymax,nkeys,nkeysmax,nfiles,funit, &
   status,firstpix
integer, dimension(2) :: naxesin
integer, dimension(3) :: naxesout
real(double) :: dmin,dmax,bpix,dscale
real(double), allocatable, dimension(:,:) :: datain
character(80) :: filelist,fitsmodelin,fileout
character(80), allocatable, dimension(:) :: header

interface

   subroutine getfits(Refname,naxes,Ref,Rmin,Rmax,nkeys,header,bpix)
      use precision
      implicit none
      integer :: nkeys
      integer, dimension(2), intent(inout) :: naxes
      real(double), intent(inout) :: Rmin,Rmax,bpix
      real(double), dimension(:,:), intent(inout) :: Ref
      character(80), intent(inout) :: Refname
      character(80), dimension(:), intent(inout) :: header
   end subroutine getfits

   subroutine writedatacube(naxesin,datain,funit,firstpix)
      use precision
      implicit none
      integer, intent(inout) :: funit,firstpix
      integer, dimension(2), intent(inout) :: naxesin
      real(double), dimension(:,:), intent(inout) :: datain
   end subroutine writedatacube

end interface

bpix=1.0d30 !marking bad pixels
xmax=2048 !dimensions of 2D FITS files to read in.
ymax=256
nkeysmax=700 !maximum size of header
allocate(header(nkeysmax))

if(iargc().lt.2)then
   write(0,*) "Usage: makecube <filelist> <prefix_out>"
   write(0,*) "  <filelist>   - text file with list of FITS files"
   write(0,*) "  <prefix_out> - prefix name for output (prefix.fits)"
   stop
endif

!read in a model spectrum
call getarg(1,filelist)
nunit=10 !unit number for data spectrum
open(unit=nunit,file=filelist,iostat=filestatus,status='old')
if(filestatus>0)then !trap missing file errors
   write(0,*) "Cannot open ",filelist
   stop
endif

nfiles=0
do
   read(nunit,'(A80)',iostat=filestatus) fitsmodelin
   if(filestatus == 0) then
      nfiles=nfiles+1
      cycle
   elseif(filestatus == -1) then
      exit
   else
      write(0,*) "File Error!!"
      write(0,900) "iostat: ",filestatus
      stop
   endif
enddo
write(0,*) "Number of FITS files to read in: ",nfiles
rewind(nunit) !re-read filelist from beginning

!allocate space for 2D FITS to read in
allocate(datain(xmax,ymax))

!read in datacube name
call getarg(2,fileout)
fileout=trim(fileout)//".fits"

firstpix=1

!read in the model spectra
i=0 !counter to count number of FITS files read in
!loop over each line in filelist
do
   !read in line
   read(nunit,'(A80)',iostat=filestatus) fitsmodelin
!   write(0,*) fitsmodelin

   !check status of read
   if(filestatus == 0) then
      i=i+1

      !read in FITS file
      call getfits(fitsmodelin,naxesin,datain,dmin,dmax,nkeys,header,bpix)
      !write(0,*) i,dmin,dmax

      !after reading in first file, we can init datacube
      if(i.eq.1) then
         !write(0,*) "Init datacube.."
         !size of the datacube
         naxesout(1)=naxesin(1)
         naxesout(2)=naxesin(2)
         naxesout(3)=nfiles
         !initialize the datacube
         call initdatacube(fileout,funit,naxesout)
         !initialize data-scale
         dscale=2.0**16.0/dmax
      endif

      !write(0,*) "Write Datacube"
      datain=datain*dscale
      !write(0,*) "scale: ",maxval(datain)
      call writedatacube(naxesin,datain,funit,firstpix)

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

close(nunit)

!close fits file
call ftclos(funit,status)
call ftfiou(funit,status)

end program makecube
