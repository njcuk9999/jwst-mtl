subroutine closefits(funit)
!simple routine that closes the active FITS file.
use precision
implicit none
integer :: funit,status

status=0

!close fits file
call ftclos(funit,status)
call ftfiou(funit,status)

return
end subroutine
