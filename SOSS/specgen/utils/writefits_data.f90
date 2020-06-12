subroutine writefitsdata(funit,xout,yout,pixels,ngroup,nint,nover,firstpix)
use precision
implicit none
!import arrays
integer :: funit,xout,yout,ngroup,nint,nover,firstpix
real(double), dimension(:,:) :: pixels
!local arrays
integer :: i,j,k,l !counters
integer :: status,bitpix,naxis,npixels,group,maxpixels
integer, dimension(:), allocatable :: naxes,buffer
real(double) :: dngrpfac
!pixels for writing
!real(double), dimension(:,:,:,:), allocatable :: pixelsout
!plot arrrays
real(double) :: bpix,tavg,sigscale

status=0 !tracks errors for FITSIO routines

maxpixels=xout*yout*ngroup*nint
if (firstpix.eq.1) then !if firstpix==1, then this is the first call. Initiate extension

   !BITPIX = 16 means that the image pixels will consist of 16-bit
   !integers.  The size of the image is given by the NAXES values.
   bitpix=-32 !using float-doubles. 
   !bitpix=16 !using float-single.
   naxis=4 !JWST obs have 4 axes
   allocate(naxes(naxis)) 
   naxes(1) = xout
   naxes(2) = yout
   naxes(3) = ngroup
   naxes(4) = nint

   !insert a new IMAGE extension immediately following the CHDU
   call FTIIMG(funit,bitpix,naxis,naxes,status)
   !add EXTNAME card
   call ftpkys(funit,'EXTNAME','SCI','',status)

   !add NOVERSAMP card
   call ftpkyj(funit,'NOVERSAM',nover,'/ Oversampling of image.  if ==1, then native resolution',status)
endif

!write current frame to FITS file as a ramp.
group=1 !this var does nothing, leave it alone
npixels =xout*yout
if(firstpix+npixels.le.maxpixels)then !check for potential overflows
   do k=1,ngroup
      dngrpfac=dble(k)/dble(ngroup)
      call ftpprd(funit,group,firstpix,npixels,pixels(1:xout,yout:1:-1)*dngrpfac,status)
      !call ftppre(funit,group,firstpix,npixels,pixels(1:xout,yout:1:-1)*dngrpfac,status)
      if(status.ne.0)then
         write(0,*) "Error FTPPR status ",status
         status=0
      endif
      firstpix=firstpix+npixels
   enddo
else
   write(0,*) 'Error: Too many pixels to write. (firstpix+npixels>maxpixels)',firstpix,npixels,maxpixels
endif

return
end subroutine writefitsdata
