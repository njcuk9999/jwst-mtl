CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine transitmodel(nfit,nplanet,nplanetmax,sol,nmax,npt,time,
     .  itime,ntt,tobs,omc,tmodel,dtype,nintg)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     My Transit-model.  Handles multi-planets, TTVs, phase-curves and
C     radial-velocities
C     (c) jasonfrowe@gmail.com
      USE OMP_LIB
      implicit none
      integer nfit,npt,i,j,nintg,dtype(npt),ii,nplanet,nplanetmax,nmax,
     .   caltran,nintgmax
      parameter(nintgmax=41)
      double precision sol(nfit),per,epoch,b,RpRs,tmodel(npt),
     .  time(npt),phi,adrs,bt(nintgmax),bs2,Pi,tpi,c1,c2,c3,c4,
     .  itime(npt),t,tflux(nintgmax),dnintg,dnintgm1,pid2,zpt,eccn,w,
     .  Eanom,Tanom,trueanomaly,phi0,vt(nintgmax),K,voff,drs,distance,
     .  dil,G,esw,ecw,Manom,Ted,ell,Ag,Cs,fDB,tide(nintgmax),
     .  alb(nintgmax),albedomod,phase,ratio,ab,tm,mu(nintgmax),y2,x2,
     .  incl,mulimbf(nintgmax),occ(nintgmax),bp(nintgmax),jm1,tdnintg
      integer ntt(nplanetmax)
      double precision tobs(nplanetmax,nmax),omc(nplanetmax,nmax),ttcor

      if(nintg.gt.nintgmax)then
         write(0,*) "Critical Error: nintg is greater than nintgmax"
         write(0,*) "nintgmax = ",nintgmax
         return
      endif

      Pi=acos(-1.d0)!define Pi and 2*Pi
      tPi=2.0d0*Pi
      pid2=Pi/2.0d0
      G=6.674d-11 !N m^2 kg^-2  Gravitation constant
      Cs=2.99792458e8 !Speed of light
c      fDB=2.21 !Doppler Boosting factor
      fDB=1.0

c sol(1) now expected to be log(rho), affects line 79 below, DL changed this
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
        bs2=sol(10*(ii-1)+8+3)*sol(10*(ii-1)+8+3)
        b=sqrt(bs2)       !impact parameter
        RpRs=abs(sol(10*(ii-1)+8+4))    !Rp/R*

        ecw=sol(10*(ii-1)+8+6)
        esw=sol(10*(ii-1)+8+5)
        eccn=(ecw*ecw+esw*esw)

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

        adrs=1000.0*exp(sol(1))*G*(Per*86400.0d0)**2/(3.0d0*Pi)
        adrs=adrs**(1.0d0/3.0d0)

CCCC        K=sol(10*(ii-1)+8+7)

        ted=sol(10*(ii-1)+8+8)/1.0d6 !Occultation Depth
        ell=sol(10*(ii-1)+8+9)/1.0d6 !Ellipsoidal variations
        ag=sol(10*(ii-1)+8+10)/1.0d6 !Phase changes

        dnintg=dble(nintg) !convert integer to double
        tdnintg=2.0d0*dnintg
        dnintgm1=2.0*dnintg-2.0

C     Find phase at centre of transit
        epoch=sol(10*(ii-1)+8+1)   !center of transit time (days)
        Eanom=tan(w/2.0d0)/sqrt((1.0d0+eccn)/(1.0d0-eccn)) !mean anomaly
        Eanom=2.0d0*atan(Eanom)
        phi0=Eanom-eccn*sin(Eanom)
C       added 2019/08/14
        Tanom=trueanomaly(eccn,Eanom)
        drs=distance(adrs,eccn,Tanom)
        incl=acos(b/drs)


!Add parallel commands here
        !write(6,*) 'Number of threads',OMP_GET_NUM_THREADS()
!$OMP PARALLEL DO PRIVATE(j,jm1,ttcor,tflux,t,phi,Manom,Tanom,drs,
!$OMP& x2,y2,bt,vt,tide,alb,caltran,mu,tm,bp,ratio,occ)
!$OMP& FIRSTPRIVATE (Eanom,c1,c2,c3,c4)
        do i=1,npt
            call lininterp(tobs,omc,nplanetmax,nmax,ii,ntt,time(i),
     .          ttcor)
            do 11 j=1,nintg
                jm1=dble(j-1)
                tflux(j)=0.0 !initialize model
C               sample over integration time

!              new time-convolution (basically gives same results)
                t=time(i)-itime(i)*(0.5d0-1.0d0/tdnintg-jm1/dnintg)-
     .              epoch-ttcor

C               get orbital position (mean anomaly)
                phi=t/per-floor(t/per)
                phi=phi*tPi+phi0
                Manom=phi
                if(Manom.gt.tPi) Manom=Manom-tPi
                if(Manom.lt.0.0d0) Manom=Manom+tPi
                call kepler(Manom,Eanom,eccn)
                Tanom=trueanomaly(eccn,Eanom)
                if(phi.gt.Pi) phi=phi-tPi
                drs=distance(adrs,eccn,Tanom)
C              Added this (2014/04/23)
                x2=drs*Sin(Tanom-w)
                y2=drs*Cos(Tanom-w)*cos(incl)

                bt(j)=sqrt(x2*x2+y2*y2)

C commented by DL                vt(j)=K*(cos(Tanom-w+pid2)+eccn*cos(-w+pid2))

C modified below by DL
                if(ell.eq.0.0) then
                    tide(j)=0.0
                else
                    tide(j)=ell*(drs/adrs)**(1.0d0/3.0d0)*
     .              cos(2.0d0*(Pid2+Tanom-w))
                endif

C modified below by DL
                if(ag.eq.0.0) then
                    alb(j)=0.0
                else
                    alb(j)=albedomod(Pi,ag,Tanom-w)*adrs/drs
                endif

 11         continue
            if(y2.ge.0.0d0)then
C       If we have a transit
                    caltran=0 !if zero, there is no transit
                    do 18 j=1,nintg
                        if(bt(j).le.1.0d0+RpRs)then
                           caltran=1
                        endif
 18                 continue
                    if(caltran.eq.1) then
                       !quadratic co-efficients
                       if((c3.eq.0.0).and.(c4.eq.0.0))then
                         call occultquad(bt,c1,c2,RpRs,tflux,mu,nintg)
                      !Kipping co-efficients
                       elseif((c1.eq.0.0).and.(c2.eq.0.0))then
                          c1=2.0d0*sqrt(c3)*c4 !convert to regular LD
                          c2=sqrt(c3)*(1.0d0-2.0d0*c4)
                          call occultquad(bt,c1,c2,RpRs,tflux,mu,nintg)
                          c1=0.0d0  !zero out entries.
                          c2=0.0d0
                       else
                      !non-linear law.
                           call occultsmall(RpRs,c1,c2,c3,c4,nintg,bt,
     .                     tflux)
                       endif
                    else
                        do 19 j=1,nintg
                           tflux(j)=1.0d0
 19                     continue
                    endif
                    tm=0.0d0
                    do 12 j=1,nintg
                        if(RpRs.le.0.0)tflux(j)=1.0d0
C                   model=transit+doppler+ellipsodial
                        tm=tm+tflux(j)+alb(j)+tide(j)!-fDB*vt(j)/Cs
 12                 continue
                    tm=tm/dnintg
            else
C       We have an eclipse
                    tm=0.0d0
                    do 20 j=1,nintg
                      bp(j)=bt(j)/RpRs
 20                 continue
                    call occultuniform(bp,1.0/RpRs,occ,nintg)
                    do 14 j=1,nintg
                        ratio=1.0d0-occ(j)

C                      Old estimate, replaced by analytic function
c                        ratio=1.0d0
c                        ab=dabs(bt(j))
c                        if((ab.ge.1.0d0).and.(ab-RpRs.le.1.0d0))then
c                            ratio=(1.0d0+RpRs-ab)/(2.0d0*RpRs)
c                        elseif((ab.lt.1.0d0).and.(ab+RpRs.ge.1.0d0))then
c                            ratio=(RpRs+1.0d0-ab)/(2.0d0*RpRs)
c                        elseif(ab-RpRs.gt.1.0d0)then
c                            ratio=0.0d0
c                        endif
c                        write(0,*) bt(j),ratio,rationew
c                        read(5,*)
                        if(RpRs.le.0.0d0) ratio=0.0d0
                        tm=tm+(1.0d0-ted*ratio)
     .                      -fDB*vt(j)/Cs+tide(j)+alb(j)
 14                 continue
                    tm=tm/dnintg
            endif
            tm=tm+(1.0d0-tm)*dil-1.0d0!add dilution
            tmodel(i)=tmodel(i)+tm
        enddo !loop over npt
!$OMP END PARALLEL DO

C     Need to add zero points (voff, zpt)

c        do 9 i=1,npt
c            write(6,*) time(i),tmodel(i)
c 9      continue

 16   continue !loop over planets

      do 15 i=1,npt
            tmodel(i)=tmodel(i)+zpt+1.0d0
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