!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
subroutine binmodels_py(wv1,wv2,dw,snpt,starmodel_wv,starmodel_flux,ld_coeff, &
	pnpt,planetmodel_wv,planetmodel_rprs, &
	bmax,bin_starmodel_wv,bin_starmodel_flux,bin_ld_coeff,bin_planetmodel_wv, &
	bin_planetmodel_rprs)
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
use precision
implicit none
integer :: snpt,pnpt,bmax
real(double) :: wv1,wv2,dw

real(double) :: starmodel_wv(snpt),starmodel_flux(snpt),ld_coeff(snpt,4)
real(double) :: bin_starmodel_wv(bmax),bin_starmodel_flux(bmax),bin_ld_coeff(bmax,4)

real(double) :: planetmodel_wv(pnpt),planetmodel_rprs(pnpt)
real(double) :: bin_planetmodel_wv(bmax),bin_planetmodel_rprs(bmax)

integer :: i,b
real(double) :: s_wv
integer, dimension(:), allocatable :: bin_count

!CCCCCCCCCCCCCCCCCCCCCC
! Start with Star model
!CCCCCCCCCCCCCCCCCCCCCC

!Initialize array
bin_starmodel_wv=0.0d0
bin_starmodel_flux=0.0d0
bin_ld_coeff=0.0d0

!allocate array to count samples in each bin
allocate(bin_count(bmax)) 
bin_count=0

do i=1,snpt
	
	s_wv=starmodel_wv(i) !current wavelength
	b=int((s_wv-wv1)/dw)+1 !bin number

	if ((b.gt.0).and.(b.le.bmax))then
		bin_starmodel_wv(b)=bin_starmodel_wv(b)+starmodel_wv(i)
		bin_starmodel_flux(b)=bin_starmodel_flux(b)+starmodel_flux(i)
		bin_ld_coeff(b,:)=bin_ld_coeff(b,:)+ld_coeff(i,:)
		bin_count(b)=bin_count(b)+1
	endif

enddo

do b=1,bmax
	if (bin_count(b).gt.0) then
		bin_starmodel_wv(b)=bin_starmodel_wv(b)/dble(bin_count(b))
		bin_starmodel_flux(b)=bin_starmodel_flux(b)/dble(bin_count(b))
		bin_ld_coeff(b,:)=bin_ld_coeff(b,:)/dble(bin_count(b))
	else
		bin_starmodel_wv(b)=dw*(dble(b)+0.5)+wv1
	endif
enddo


!CCCCCCCCCCCCCCCCCCCCCC
! Bin Planet Model
!CCCCCCCCCCCCCCCCCCCCCC

bin_planetmodel_wv=0.0d0
bin_planetmodel_rprs=0.0d0

!reset bin_count
bin_count=0

do i=1,pnpt
	
    s_wv=planetmodel_wv(i) !current wavelength
	b=int((s_wv-wv1)/dw)+1 !bin number

	if ((b.gt.0).and.(b.le.bmax))then
		bin_planetmodel_wv(b)=bin_planetmodel_wv(b)+planetmodel_wv(i)
		bin_planetmodel_rprs(b)=bin_planetmodel_rprs(b)+planetmodel_rprs(i)
		bin_count(b)=bin_count(b)+1
	endif

enddo

do b=1,bmax
	if (bin_count(b).gt.0) then
		bin_planetmodel_wv(b)=bin_planetmodel_wv(b)/dble(bin_count(b))
		bin_planetmodel_rprs(b)=bin_planetmodel_rprs(b)/dble(bin_count(b))
	else
		bin_planetmodel_wv(b)=dw*(dble(b)+0.5)+wv1
	endif
enddo

return
end subroutine binmodels_py
 