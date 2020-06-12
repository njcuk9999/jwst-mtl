program genoskernel
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer noversample,nrK,nKs,i,j,k,ii,jj,l,m,nK,nnative,iargc,nsample
real(double) :: dnover,x1,x2,y
real(double), allocatable, dimension(:) :: x1a,x2a
real(double), allocatable, dimension(:,:) :: wKernel,Kernel,y2a
real(double), allocatable, dimension(:,:,:) :: rKernel
character(80), allocatable, dimension(:) :: filenames
character(80) :: cline

interface
   subroutine readKernels(nrK,nK,rKernel,noversample)
      use precision
      implicit none
      integer,intent(in) :: noversample
      integer,intent(inout) :: nrK,nK
      real(double), dimension(:,:,:), intent(inout) :: rKernel
   end subroutine
end interface
interface
   subroutine writefits(nxmax,nymax,parray,fileout)
      use precision
      implicit none
      integer, intent(inout) :: nxmax,nymax
      real(double), dimension(:,:), intent(inout) :: parray
      character(80) :: fileout
   end subroutine
end interface

if(iargc().lt.1)then
   write(0,*) "Usage: genoskernel noversample"
   write(0,*) " noversample - is new sampling for Kernel (must be > 0)"
   stop
endif

!oversampling
call getarg(1,cline)
read(cline,*) noversample !read in noversample
if(noversample.le.0)then
   write(0,*) "noversample must be greater than zero"
   stop
endif

nnative=10 !native size of Kernels
dnover=dble(nnative)/dble(noversample) !double int -> double

!read in Kernels
nrK=30 !number of Kernels to readin
nKs=640 !native size of Kernels
allocate(rKernel(nrK,nKs,nKs))
call readKernels(nrK,nKs,rKernel,nnative) !read in native size Kernels

!get filenames for output
allocate(filenames(nrK))
call getfilenames(noversample,nrK,filenames)

!allocate space for oversample Kernels
allocate(wKernel(nKs,nKs))
nK=nKs*noversample/nnative
allocate(Kernel(nK,nK))


!set up arrays for spline calculations
allocate(x1a(nKs),x2a(nKs)) !allocate array for spline X,Y co-ordinates
do i=1,nKs
   x1a(i)=dble(i) !co-ordinates for grid to spline
enddo
x2a=x1a !copy x1a to x2a, since they are the same (square-array)
allocate(y2a(nKs,nKs)) !array to hold derivatives from splie2

do k=1,nrK
   write(0,501) "Kernel #",k,"/",nrK
   write(0,502) filenames(k)
   501 format(A8,I2,A1,I2)
   502 format(A80)
   wKernel=rKernel(k,:,:)
!  use a cubic spline to oversample the Kernel
   call splie2(x1a,x2a,wKernel,nKs,nKs,y2a)  !calculate derivatives

!   do i=1,nKs
!      do j=1,nKs
!         ii=i*noversample-noversample+1
!         jj=j*noversample-noversample+1
!         do l=1,noversample
!            do m=1,noversample
!               x1=dble(ii+l-1)/dnover
!               x2=dble(jj+m-1)/dnover
!               call splin2(x1a,x2a,wKernel,y2a,nKs,nKs,x1,x2,y)
!               Kernel(ii+l-1,jj+m-1)=y
!!            write(0,*) i,j,wKernel(i,j),Kernel(ii+l-1,jj+m-1)
!!            read(5,*)
!            enddo
!         enddo
!      enddo
!   enddo

   do i=1,nK  !loop over new Kernel
      do j=1,nK
         x1=dble(i)*dnover
         x2=dble(j)*dnover
         call splin2(x1a,x2a,wKernel,y2a,nKs,nKs,x1,x2,y)
!         write(0,*) i,j,x1,x2,y
         Kernel(i,j)=y
      enddo
   enddo

!  write the new oversampled Kernel
   Kernel=transpose(Kernel) !need to flip Kernel
   call writefits(nK,nK,Kernel,filenames(k))
enddo

end program genoskernel

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine getfilenames(noversample,nrK,filenames)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
integer noversample,nrK
character(80), dimension(nrK) :: filenames
character(80) :: cfs
!filenames of Kernels

if(noversample.lt.10)then
   cfs='(A7,I1,A32)'
else
   cfs='(A7,I2,A32)'
endif

write(filenames(1),cfs) "Kernels",noversample,"/psf_500nm_x10_oversampled.fits "
write(filenames(2),cfs) "Kernels",noversample,"/psf_600nm_x10_oversampled.fits "
write(filenames(3),cfs) "Kernels",noversample,"/psf_700nm_x10_oversampled.fits "
write(filenames(4),cfs) "Kernels",noversample,"/psf_800nm_x10_oversampled.fits "
write(filenames(5),cfs) "Kernels",noversample,"/psf_900nm_x10_oversampled.fits "
write(filenames(6),cfs) "Kernels",noversample,"/psf_1000nm_x10_oversampled.fits"
write(filenames(7),cfs) "Kernels",noversample,"/psf_1100nm_x10_oversampled.fits"
write(filenames(8),cfs) "Kernels",noversample,"/psf_1200nm_x10_oversampled.fits"
write(filenames(9),cfs) "Kernels",noversample,"/psf_1300nm_x10_oversampled.fits"
write(filenames(10),cfs) "Kernels",noversample,"/psf_1400nm_x10_oversampled.fits"
write(filenames(11),cfs) "Kernels",noversample,"/psf_1500nm_x10_oversampled.fits"
write(filenames(12),cfs) "Kernels",noversample,"/psf_1600nm_x10_oversampled.fits"
write(filenames(13),cfs) "Kernels",noversample,"/psf_1700nm_x10_oversampled.fits"
write(filenames(14),cfs) "Kernels",noversample,"/psf_1800nm_x10_oversampled.fits"
write(filenames(15),cfs) "Kernels",noversample,"/psf_1900nm_x10_oversampled.fits"
write(filenames(16),cfs) "Kernels",noversample,"/psf_2000nm_x10_oversampled.fits"
write(filenames(17),cfs) "Kernels",noversample,"/psf_2100nm_x10_oversampled.fits"
write(filenames(18),cfs) "Kernels",noversample,"/psf_2200nm_x10_oversampled.fits"
write(filenames(19),cfs) "Kernels",noversample,"/psf_2300nm_x10_oversampled.fits"
write(filenames(20),cfs) "Kernels",noversample,"/psf_2400nm_x10_oversampled.fits"
write(filenames(21),cfs) "Kernels",noversample,"/psf_2500nm_x10_oversampled.fits"
write(filenames(22),cfs) "Kernels",noversample,"/psf_2600nm_x10_oversampled.fits"
write(filenames(23),cfs) "Kernels",noversample,"/psf_2700nm_x10_oversampled.fits"
write(filenames(24),cfs) "Kernels",noversample,"/psf_2800nm_x10_oversampled.fits"
write(filenames(25),cfs) "Kernels",noversample,"/psf_2900nm_x10_oversampled.fits"
write(filenames(26),cfs) "Kernels",noversample,"/psf_3000nm_x10_oversampled.fits"
write(filenames(27),cfs) "Kernels",noversample,"/psf_3100nm_x10_oversampled.fits"
write(filenames(28),cfs) "Kernels",noversample,"/psf_3200nm_x10_oversampled.fits"
write(filenames(29),cfs) "Kernels",noversample,"/psf_3300nm_x10_oversampled.fits"
write(filenames(30),cfs) "Kernels",noversample,"/psf_3400nm_x10_oversampled.fits"

return
end
