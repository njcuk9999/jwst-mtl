subroutine addgnoise(xout,yout,opixels,snr,seed)
use precision
implicit none
integer :: xout,yout,seed,i,j
real(double) :: snr,maxcol,noiseamp,gasdev
real(double), dimension(:,:) :: opixels


maxcol=0.0d0
do i=1,xout
   maxcol=max(maxcol,Sum(opixels(i,:)))
enddo

noiseamp=maxcol/snr !scale based on S/N of strongest pixel.

do i=1,xout
   do j=1,yout
      opixels(i,j)=opixels(i,j)+noiseamp*gasdev(seed)
   enddo
enddo

return
end subroutine addgnoise
