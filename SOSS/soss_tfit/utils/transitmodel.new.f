CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine transitmodel(nfit,nplanet,sol,npt,time,itime,tmodel,
     .  dtype)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      integer nfit,npt,i,j,nintg,dtype(npt),ii,nplanet
      parameter(nintg=11)
      double precision sol(nfit),per,epoch,b,RpRs,tmodel(npt),
     .  time(npt),phi,adrs,bt(nintg),bs2,Pi,tpi,c1,c2,c3,c4,
     .  itime(npt),t,tflux(nintg),dnintg,dnintgm1,pid2,zpt,eccn,w,Eanom,
     .  Tanom,trueanomaly,phi0,vt(nintg),K,voff,drs,distance,dil,G,
     .  esw,ecw,Manom,Ted,ell,Ag,Cs,fDB,tide(nintg),alb(nintg),
     .  albedomod,phase,ratio,ab,tm,mu(nintg),y2,x2,incl
      
      Pi=acos(-1.d0)!define Pi and 2*Pi
      tPi=2.0d0*Pi 
      pid2=Pi/2.0d0
      G=6.674d-11 !N m^2 kg^-2  Gravitation constant
      Cs=2.99792458e8 !Speed of light
      fDB=0.0!1.896 !Doppler Boosting factor
      
      c1=sol(2)      !non-linear limb-darkening
      c2=sol(3)
      c3=sol(4)
      c4=sol(5)
      dil=sol(6)     !dilution parameter (model scaling)
      voff=sol(7)    !velocity zero point
      zpt=sol(8)     !flux zero point.
      
      do 17 i=1,npt
        tmodel(i)=0.0d0
 17   continue
      
      do 16 ii=1,nplanet
      
        per=sol(10*(ii-1)+8+2)     !Period (days)
        bs2=abs(sol(10*(ii-1)+8+3))
        b=sqrt(bs2)       !impact parameter
        RpRs=sol(10*(ii-1)+8+4)    !Rp/R*

        ecw=sol(10*(ii-1)+8+5)
        esw=sol(10*(ii-1)+8+6)
        eccn=sqrt(ecw*ecw+esw*esw) !eccentricity
        if(eccn.ge.1.0) eccn=0.99
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
      
c        write(0,*) sol(7),sol(8),w
c        write(0,*) "w:",acos(sol(7)/eccn),asin(sol(8)/eccn)
c        read(5,*)

C       a/R*
c        adrs=sol(5)*per/tpi*sqrt(1-sol(3))*(1+sol(8))/sqrt(1-eccn*eccn)
        adrs=1000.0*sol(1)*G*(Per*86400.0d0)**2/(3.0d0*Pi)
        adrs=adrs**(1.0d0/3.0d0)
c        write(0,*) "a/R*:",adrs

C       Find inclination
        incl=acos(b/adrs)


        K=sol(10*(ii-1)+8+7)
      
        ted=sol(10*(ii-1)+8+8)/1.0d6 !Occultation Depth
        ell=sol(10*(ii-1)+8+9)/1.0d6 !Ellipsoidal variations
        ag=sol(10*(ii-1)+8+10)/1.0d6 !Phase changes
      
        dnintg=dble(nintg) !convert integer to double
        dnintgm1=2.0*dnintg-2.0
      
C     Find phase at centre of transit
        epoch=sol(10*(ii-1)+8+1)   !center of transit time (days)
c        phi1=(epoch/per-int(epoch/per))*twopi
        Eanom=tan(w/2.0d0)/sqrt((1.0d0+eccn)/(1.0d0-eccn)) !mean anomaly
        Eanom=2.0d0*atan(Eanom)
        phi0=Eanom-eccn*sin(Eanom)
      
        do 10 i=1,npt
            do 11 j=1,nintg
                tflux(j)=0.0 !initialize model
C               sample over integration time
                t=time(i)+itime(i)*(2.0*dble(j)-dnintg-1.0)/dnintgm1-
     .              epoch
C               get orbital position (mean anomaly)
                phi=t/per-floor(t/per)
                phi=phi*tPi+phi0
                Manom=phi
                if(Manom.gt.tPi) Manom=Manom-tPi
                if(Manom.lt.0.0d0) Manom=Manom+tPi
                call kepler(Manom,Eanom,eccn)
                Tanom=trueanomaly(eccn,Eanom)
                if(phi.gt.Pi) phi=phi-tPi            
                drs=distance(adrs,eccn,Tanom-w)
                x2=adrs*Sin(Tanom-w)
                y2=adrs*Cos(Tanom-w)*cos(incl)
c                bt(j)=sqrt(bs2+(drs*sin(Tanom-w))**2)
                bt(j)=sqrt(x2*x2+y2*y2)
C               Correct for light-travel time!
c            if((abs(bt(j))-RpRs.le.1.0d0).and.(abs(phi).gt.Pid2))then
c              t=time(i)-ltt+itime(i)*(2.0*dble(j)-dnintg-1.0)/dnintgm1
c     .              -epoch
c              phi=(t-ltt)/per-floor((t-ltt)/per)
c              phi=phi*tPi
c              Manom=phi+w
c              if(Manom.gt.tPi) Manom=Manom-tPi
c              if(Manom.lt.0.0d0) Manom=Manom+tPi
c              call kepler(Manom,Eanom,eccn)
c              Tanom=trueanomaly(eccn,Eanom)
c                if(phi.gt.Pi) phi=phi-tPi            
c                drs=distance(adrs,eccn,Tanom)
c                bt(j)=sqrt(bs2+(drs*sin(Tanom-phi0))**2)
c            endif
c                vt(j)=K*(cos(Pid2+Tanom-phi0)+eccn*cos(w))
                vt(j)=K*(cos(Tanom-w+pid2)+eccn*cos(-w+pid2))
c                tide(j)=ell*cos(2.0d0*(Pid2+Tanom-phi0))
                tide(j)=ell*(drs/adrs)**(1.0d0/3.0d0)*
     .              cos(2.0d0*(Pid2+phi))
     
c                alb(j)=albedomod(Pi,ag,Tanom-phi0)
                alb(j)=albedomod(Pi,ag,phi)*adrs/drs
            
                if(j.eq.nintg/2+1)then
                    phase=Tanom-w!phi(nintg/2+1)
                    if(phase.gt.Pi) phase=phase-tPi
                    if(phase.lt.-Pi) phase=phase+tPi
                endif

 11         continue
            if(dtype(i).eq.0)then
c                if(abs(phase).lt.Pid2)then
                if(y2.ge.0.0d0)then
C       If we have a transit
                    if((c3.eq.0.0).and.(c4.eq.0.0))then
                       call occultquad(bt,c1,c2,RpRs,tflux,mu,nintg)
                    else
                       call occultsmall(RpRs,c1,c2,c3,c4,nintg,bt,tflux)
                    endif
                    tm=0.0d0
                    do 12 j=1,nintg
                        if(RpRs.le.0.0)tflux(j)=1.0d0
C                   model=transit+doppler+ellipsodial 
                        tm=tm+tflux(j)-fDB*vt(j)/Cs+tide(j)+alb(j)
 12                 continue
                    tm=tm/dnintg
                else
C       We have an eclipse
                    tm=0.0d0
                    do 14 j=1,nintg
                        ratio=1.0d0
                        ab=dabs(bt(j))
                        if((ab.ge.1.0d0).and.(ab-RpRs.le.1.0d0))then
                            ratio=(1.0d0+RpRs-ab)/(2.0d0*RpRs)
                        elseif((ab.lt.1.0d0).and.(ab+RpRs.ge.1.0d0))then
                            ratio=(RpRs+1.0d0-ab)/(2.0d0*RpRs)
                        elseif(ab-RpRs.gt.1.0d0)then
                            ratio=0.0d0
                        endif
                        if(RpRs.le.0.0d0) ratio=0.0d0
c                    write(0,*) ab,RpRs,ratio
c                    read(5,*) 
                        tm=tm+(1.0d0-ted*ratio)
     .                      -fDB*vt(j)/Cs+tide(j)+alb(j)
 14                 continue
                    tm=tm/dnintg
                endif
                tm=tm+(1.0d0-tm)*dil-1.0d0!add dilution
            else
                tm=0.0d0
                do 13 j=1,nintg
                    tm=tm+vt(j)
 13             continue
                tm=tm/dnintg
c            write(0,*) "rv:",tmodel(i)
c            read(5,*)
            endif
            tmodel(i)=tmodel(i)+tm
 10     continue
 
C     Need to add zero points (voff, zpt)
      
c        do 9 i=1,npt
c            write(6,*) time(i),tmodel(i)
c 9      continue
  
 16   continue  
 
      do 15 i=1,npt
        if(dtype(i).eq.0)then
            tmodel(i)=tmodel(i)+zpt+1.0d0
        else
            tmodel(i)=tmodel(i)+voff
        endif
 15   continue
      
      return
      end
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      double precision function albedomod(Pi,ag,phi)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      double precision Pi,phi,alpha,phase,ag

      phi=phi+Pi
      if(phi.gt.2.0*Pi) phi=phi-2.0*Pi


      alpha=abs(phi)      
c      alpha=2.0*Pi*t/Per+phi
      alpha=alpha-2.0*Pi*int(alpha/(2.0*Pi))
      if(alpha.gt.Pi) alpha=abs(alpha-2.0*pi)
c      write(6,*) t,alpha
c      phase=(1.0d0+cos(alpha))/2.0d0
      phase=(sin(alpha)+(Pi-alpha)*cos(alpha))/Pi  !Lambertian Sphere
      
      albedomod=ag*phase
      
      return
      end