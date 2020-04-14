subroutine addshotnoise(xout,yout,opixels,seed)
use precision
implicit none
!import vars
integer :: xout,yout,seed
real(double), dimension(:,:) :: opixels
!local vars
integer :: i,j
real(double) :: pmax,sat,gasdev,readnoise

!scale largest pixel to 75000 e-
sat=75000.0d0 !e-
!readnoise
readnoise=20.0d0 !e-

!find largest pixel value
pmax=maxval(opixels(1:xout,1:yout))
!scale (undone below)
opixels=opixels*sat/pmax

!add shot-noise (Guassian)
do i=1,xout
   do j=1,yout
      opixels(i,j)=opixels(i,j)+sqrt(abs(opixels(i,j)))*gasdev(seed)
   enddo
enddo

!add read noise
do i=1,xout
   do j=1,yout
      opixels(i,j)=opixels(i,j)+readnoise*gasdev(seed)
   enddo
enddo

!undo scaling
opixels=opixels/sat*pmax


end subroutine addshotnoise
