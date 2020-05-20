function orbmodel(t,sol)
!returns impact parameter b.
use precision
implicit none
!import vars
real(double) :: t,sol(6),orbmodel
!local vars
real(double) :: Pi,tPi,pid2,G
real(double) :: per,bs2,b,ecw,esw,eccn,w,adrs,incl,epoch,Eanom,phi0
real(double) :: phi,Manom,Tanom,drs,x2,y2
real(double) :: trueanomaly,distance

Pi=acos(-1.d0)!define Pi and 2*Pi
tPi=2.0d0*Pi 
pid2=Pi/2.0d0
G=6.674d-11 !N m^2 kg^-2  Gravitation constant

per=sol(3)        !Period (days)
bs2=sol(4)*sol(4) !impact parameter squared
b=sqrt(bs2)       !impact parameter

ecw=sol(6) 
esw=sol(5)
eccn=(ecw*ecw+esw*esw) !eccentricity

if(eccn.ge.1.0) eccn=0.99 !force upper limit on eccentricity
if(eccn.eq.0.0d0)then
   w=0.0d0
else
   if(ecw.eq.0.0d0)then
       w=Pi/2.0d0
   else
       w=atan(esw/ecw)
   endif
   if((ecw.gt.0.0d0).and.(esw.lt.0.0d0))then
       w=tPi+w
   elseif((ecw.lt.0.0d0).and.(esw.ge.0.0d0))then 
       w=Pi+w
   elseif((ecw.le.0.0d0).and.(esw.lt.0.0d0))then
       w=Pi+w
   endif
endif    

!calculate the scaled semi-major axis a/R*
adrs=1000.0*sol(1)*G*(Per*86400.0d0)**2/(3.0d0*Pi)  
adrs=adrs**(1.0d0/3.0d0)

epoch=sol(2)   !center of transit time (days)

Eanom=tan(w/2.0d0)/sqrt((1.0d0+eccn)/(1.0d0-eccn)) !mean anomaly
Eanom=2.0d0*atan(Eanom)
phi0=Eanom-eccn*sin(Eanom)

phi=t/per-floor(t/per)
phi=phi*tPi+phi0
Manom=phi
if(Manom.gt.tPi) Manom=Manom-tPi
if(Manom.lt.0.0d0) Manom=Manom+tPi
call kepler(Manom,Eanom,eccn)
Tanom=trueanomaly(eccn,Eanom)
if(phi.gt.Pi) phi=phi-tPi            
drs=distance(adrs,eccn,Tanom)

incl=acos(b/drs)
x2=drs*Sin(Tanom-w)
y2=drs*Cos(Tanom-w)*cos(incl)
b=sqrt(x2*x2+y2*y2)

orbmodel=b !return impact parameter for function call.

return 
end 
