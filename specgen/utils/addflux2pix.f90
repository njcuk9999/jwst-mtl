subroutine addflux2pix(px,py,xmax,ymax,pixels,fmod)
!Jason Rowe 2015 - jasonfrowe@gmail.com
use precision
implicit none
integer xmax,ymax,npx,npy
real(double) :: fmod,dx,dy,px,py,pxmh,pymh,acheck
real(double), dimension(xmax,ymax) :: pixels

!so we add in flux with a 'psf' that is the size of one pixel

pxmh=px-0.5d0 !location of reference corner of PSF square
pymh=py-0.5d0

!acheck=0.0d0 !check that area is unity

!start with reference pixel
dx=floor(px+0.5)-pxmh !find edge of square
dy=floor(py+0.5)-pymh
npx=int(pxmh)
npy=int(pymh)
!check that pixel location is valid part of array
if((npx.gt.0).and.(npx.le.xmax).and.(npy.gt.0).and.(npy.le.ymax))then
   pixels(npx,npy)=pixels(npx,npy)+fmod*dx*dy
endif
!acheck=acheck+dx*dy

!+dx direction
npx=int(pxmh)+1
npy=int(pymh)
if((npx.gt.0).and.(npx.le.xmax).and.(npy.gt.0).and.(npy.le.ymax))then
   pixels(npx,npy)=pixels(npx,npy)+fmod*(1.0d0-dx)*dy
endif
!acheck=acheck+(1.0d0-dx)*dy

!+dy direction
npx=int(pxmh)
npy=int(pymh)+1
if((npx.gt.0).and.(npx.le.xmax).and.(npy.gt.0).and.(npy.le.ymax))then
   pixels(npx,npy)=pixels(npx,npy)+fmod*dx*(1.0d0-dy)
endif
!acheck=acheck+dx*(1.0d0-dy)

!+dx+dy direction
npx=int(pxmh)+1
npy=int(pymh)+1
if((npx.gt.0).and.(npx.le.xmax).and.(npy.gt.0).and.(npy.le.ymax))then
   pixels(npx,npy)=pixels(npx,npy)+fmod*(1.0d0-dx)*(1.0d0-dy)
endif
!acheck=acheck+(1.0d0-dx)*(1.0d0-dy)

!write(0,*) "area check: ",acheck

return
end


