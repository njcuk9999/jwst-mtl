program rescale
use precision
implicit none
!rescaling FITS files to fix scaling mistake
integer iargc

character(80) :: file1,file2,file3

if(iargc().lt.4)then
   write(0,*) "Usage: rescale SPECref SPECfix FITSfix"
   stop
endif

call getarg(1,file1)
call getarg(2,file2)
call getarg(3,file3)

call getspec(file1,

end program rescale
