subroutine readKernels(nrK,nK,rKernel,noversample)
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer :: nrK,nK,nkeys,nkeysmax,i,j,noversample
integer, dimension(2) :: naxes
real(double) :: Krmin,Krmax,bpix
real(double), allocatable, dimension(:,:) :: Kernel
real(double), dimension(:,:,:) :: rKernel
character(80), allocatable, dimension(:) :: filenames,header
character(80) :: cfs

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
end interface

bpix=1000000.0 !marking bad pixels

if(noversample.lt.10)then
   cfs='(A14,I1,A32)'
else
   cfs='(A14,I2,A32)'
endif

!filenames of Kernels
allocate(filenames(nrK))
write(filenames(1),cfs) "Kernel/Kernels",noversample,"/psf_500nm_x10_oversampled.fits "
write(filenames(2),cfs) "Kernel/Kernels",noversample,"/psf_600nm_x10_oversampled.fits "
write(filenames(3),cfs) "Kernel/Kernels",noversample,"/psf_700nm_x10_oversampled.fits "
write(filenames(4),cfs) "Kernel/Kernels",noversample,"/psf_800nm_x10_oversampled.fits "
write(filenames(5),cfs) "Kernel/Kernels",noversample,"/psf_900nm_x10_oversampled.fits "
!1000nm is missing so just read in 900 twice.
write(filenames(6),cfs) "Kernel/Kernels",noversample,"/psf_900nm_x10_oversampled.fits "
write(filenames(7),cfs) "Kernel/Kernels",noversample,"/psf_1100nm_x10_oversampled.fits"
write(filenames(8),cfs) "Kernel/Kernels",noversample,"/psf_1200nm_x10_oversampled.fits"
write(filenames(9),cfs) "Kernel/Kernels",noversample,"/psf_1300nm_x10_oversampled.fits"
write(filenames(10),cfs) "Kernel/Kernels",noversample,"/psf_1400nm_x10_oversampled.fits"
write(filenames(11),cfs) "Kernel/Kernels",noversample,"/psf_1500nm_x10_oversampled.fits"
write(filenames(12),cfs) "Kernel/Kernels",noversample,"/psf_1600nm_x10_oversampled.fits"
write(filenames(13),cfs) "Kernel/Kernels",noversample,"/psf_1700nm_x10_oversampled.fits"
write(filenames(14),cfs) "Kernel/Kernels",noversample,"/psf_1800nm_x10_oversampled.fits"
write(filenames(15),cfs) "Kernel/Kernels",noversample,"/psf_1900nm_x10_oversampled.fits"
write(filenames(16),cfs) "Kernel/Kernels",noversample,"/psf_2000nm_x10_oversampled.fits"
write(filenames(17),cfs) "Kernel/Kernels",noversample,"/psf_2100nm_x10_oversampled.fits"
write(filenames(18),cfs) "Kernel/Kernels",noversample,"/psf_2200nm_x10_oversampled.fits"
write(filenames(19),cfs) "Kernel/Kernels",noversample,"/psf_2300nm_x10_oversampled.fits"
write(filenames(20),cfs) "Kernel/Kernels",noversample,"/psf_2400nm_x10_oversampled.fits"
write(filenames(21),cfs) "Kernel/Kernels",noversample,"/psf_2500nm_x10_oversampled.fits"
write(filenames(22),cfs) "Kernel/Kernels",noversample,"/psf_2600nm_x10_oversampled.fits"
write(filenames(23),cfs) "Kernel/Kernels",noversample,"/psf_2700nm_x10_oversampled.fits"
write(filenames(24),cfs) "Kernel/Kernels",noversample,"/psf_2800nm_x10_oversampled.fits"
write(filenames(25),cfs) "Kernel/Kernels",noversample,"/psf_2900nm_x10_oversampled.fits"
write(filenames(26),cfs) "Kernel/Kernels",noversample,"/psf_3000nm_x10_oversampled.fits"
write(filenames(27),cfs) "Kernel/Kernels",noversample,"/psf_3100nm_x10_oversampled.fits"
write(filenames(28),cfs) "Kernel/Kernels",noversample,"/psf_3200nm_x10_oversampled.fits"
write(filenames(29),cfs) "Kernel/Kernels",noversample,"/psf_3300nm_x10_oversampled.fits"
write(filenames(30),cfs) "Kernel/Kernels",noversample,"/psf_3400nm_x10_oversampled.fits"

nkeysmax=700
allocate(Kernel(nK,nK),header(nkeysmax))

do i=1,nrK
   call getfits(filenames(i),naxes,Kernel,Krmin,Krmax,nkeys,header,bpix)
!   write(0,*) i,Krmin,Krmax
   rKernel(i,:,:)=transpose(Kernel(:,:))
enddo


return
end
