subroutine readresponse
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
use response !this module returns the trace via pointers
implicit none
integer :: fstatus,unitfits,blocksize,readwrite,nhuds,hud,hudtype,nkeys,&
   nrecmax,i,j,nrows,ncols,datacode,width
integer, target :: irepeat
real(double), allocatable, dimension(:), target :: lambda,respondo1,    &
   respondo2,respondo3
character(80) :: filename
character(80), allocatable, dimension(:) :: header
character(8) :: imagetype,extname
logical anyf
!keep arrays alive for pointers
save irepeat,lambda,respondo1,respondo2,respondo3

nrecmax=700 !maximum number of entries allowed in the header
allocate(header(nrecmax)) !allocate array space for header

!name of FITS file with response table
filename="trace/NIRISS_Throughput_EBOI.fits"

fstatus=0 !flag to check for errors
!get an unused UNIT number
call ftgiou(unitfits,fstatus)
readwrite=0 !0-readonly, 1=read/write
!open the FITS file
call ftopen(unitfits,filename,readwrite,blocksize,fstatus)
!get the number of the extensions (huds) in the image
if(fstatus.ne.0)then
   write(0,*) "Status: ",fstatus
   write(0,*) "Cannot open "
   write(0,'(A80)') filename
   stop
endif
call ftthdu(unitfits,nhuds,fstatus)
!write(0,*) "nhuds: ",nhuds
if(fstatus.ne.0) then
   write(6,*) "whoops in readresponse: ",fstatus
endif

!loop over all extensions
do hud=1,nhuds
!This command moves us to the next HUD
   call ftmahd(unitfits,hud,hudtype,fstatus)
!   write(0,*) "HUDType: ",hudtype
   if(fstatus.eq.0) then
      !Read in all the header information
      call readheader(unitfits,fstatus,header,nkeys)
      do j=1,nkeys
!         write(0,'(A80)') header(j)(1:80)
         if(header(j)(1:8).eq."XTENSION") then
            read(header(j)(12:19),'(A8)') imagetype
!            write(6,*) header(j)(1:8),header(j)(12:19)
!            write(6,*) "xtension:",imagetype
         endif
      enddo
!     check extension name.  If we have a binary table, then read it in
      if(imagetype.eq."BINTABLE")then
         !get the number of rows and columns in the table.
         call FTGNRW(unitfits,nrows,fstatus)
         call FTGNCL(unitfits,ncols,fstatus)
 !        write(0,*) "NR,NC:",nrows,ncols
         do i=1,ncols
            !see what kind of data string we are dealing with.
            !in the target table.
            call ftgtcl(unitfits,i,datacode,irepeat,width,fstatus)
!            write(6,*) "cc:",i,datacode,irepeat,width,fstatus

            select case(i)
               case(1) !read wavelength
                  allocate(lambda(irepeat)) !allocate array space
                  !now we readin the table values for this entry
                  call ftgcvd(unitfits,i,1,1,irepeat," ",lambda,anyf,fstatus)
                  if(fstatus.ne.0)then !check for errors
                     write(0,*) "Error Table Lambda Read: ",fstatus
                  endif
               case(23) !read n=1 response
                  allocate(respondo1(irepeat)) !allocate array space
                  !now we readin the table values for this entry
                  call ftgcvd(unitfits,i,1,1,irepeat," ",respondo1,anyf,fstatus)
                  if(fstatus.ne.0)then !check for errors
                     write(0,*) "Error Table Order1 Read: ",fstatus
                  endif
               case(24) !read n=2 response
                  allocate(respondo2(irepeat)) !allocate array space
                  !now we readin the table values for this entry
                  call ftgcvd(unitfits,i,1,1,irepeat," ",respondo2,anyf,fstatus)
                  if(fstatus.ne.0)then !check for errors
                     write(0,*) "Error Table Order2 Read: ",fstatus
                  endif
               case(25) !read n=3 response
                  allocate(respondo3(irepeat)) !allocate array space
                  !now we readin the table values for this entry
                  call ftgcvd(unitfits,i,1,1,irepeat," ",respondo3,anyf,fstatus)
                  if(fstatus.ne.0)then !check for errors
                     write(0,*) "Error Table Order3 Read: ",fstatus
                  endif
            end select
         enddo
      endif
   else
      write(0,*) "Error in HUDloop: ",fstatus
   endif
enddo

!write(0,*) "All good..",fstatus
!read(5,*)

!update pointers to return values
nres=>irepeat   !number of measurements
ld=>lambda      !wavelength (nm)
res1=>respondo1 !response for first order
res2=>respondo2 !response for second order
res3=>respondo3 !response for third order

return
end subroutine readresponse
