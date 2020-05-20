subroutine writefitsdata(funit,xout,yout,pixels,ngroup,nint)
use precision
implicit none
!import arrays
integer :: funit,xout,yout,ngroup,nint
real(double), dimension(:,:) :: pixels
!local arrays
integer :: i,j,k,l !counters
integer :: status,bitpix,naxis,npixels,group,firstpix,nbuf,nbuffer
integer, dimension(:), allocatable :: naxes,buffer
real(double) :: dngrpfac
!pixels for writing
real(double), dimension(:,:,:,:), allocatable :: pixelsout
!plot arrrays
real(double) :: bpix,tavg,sigscale

status=0 !tracks errors for FITSIO routines

!BITPIX = 16 means that the image pixels will consist of 16-bit
!integers.  The size of the image is given by the NAXES values.
bitpix=-32 !using float-doubles. 
naxis=4 !JWST obs have 4 axes
allocate(naxes(naxis)) 
naxes(1) = xout
naxes(2) = yout
naxes(3) = ngroup
naxes(4) = nint


allocate(pixelsout(xout,yout,ngroup,nint))
do k=1,ngroup
   dngrpfac=dble(k)/dble(ngroup)
   pixelsout(1:xout,1:yout,k,1)=pixels(1:xout,yout:1:-1)*dngrpfac
enddo

!insert a new IMAGE extension immediately following the CHDU
call FTIIMG(funit,bitpix,naxis,naxes,status)
!add EXTNAME card
call ftpkys(funit,'EXTNAME','SCI','',status)

firstpix=1
group=1 !this var does nothing, leave it alone
npixels=1
do i=1,naxis
	npixels=npixels*naxes(i)
enddo
call ftpprd(funit,group,firstpix,npixels,pixelsout,status)

return
end subroutine writefitsdata
