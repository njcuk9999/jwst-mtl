C     Cubic-spline from numerical recipes.  Slight changes to make
C     routine double precision and remove pause statements that trip up
C     gfortran compilers
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE splie2(x1a,x2a,ya,m,n,y2a)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      INTEGER m,n,NN
      REAL*8 x1a(m),x2a(n),y2a(m,n),ya(m,n)
      PARAMETER (NN=200000) !Maximum expected value of n and m.
C     USES spline
C     Given an m by n tabulated function ya(1:m,1:n), and tabulated
C     independent variables x2a(1:n), this routine constructs
C     one-dimensional natural cubic splines of the rows of ya and
C     returns the second-derivatives in the array y2a(1:m,1:n). (The
C     array x1a is included in the argument list merely for consistency
C     with routine splin2.)
      INTEGER j,k
      REAL*8 y2tmp(NN),ytmp(NN)
      do 13 j=1,m
         do 11 k=1,n
            ytmp(k)=ya(j,k)
 11      continue
         call spline(x2a,ytmp,n,1.d30,1.d30,y2tmp)
         do 12 k=1,n
            y2a(j,k)=y2tmp(k)
 12      continue
 13   continue
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE splin2(x1a,x2a,ya,y2a,m,n,x1,x2,y)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      INTEGER m,n,NN
      REAL*8 x1,x2,y,x1a(m),x2a(n),y2a(m,n),ya(m,n)
      PARAMETER (NN=200000) !Maximum expected value of n and m.
C     USESspline,splint
C     Given x1a, x2a, ya, m, n as described in splie2 and y2a as
C     produced by that routine; and given a desired interpolating point
C     x1,x2; this routine returns an interpolated function value y by
C     bicubic spline interpolation.
      INTEGER j,k
      REAL*8 y2tmp(NN),ytmp(NN),yytmp(NN)
      do 12 j=1,m
         do 11 k=1,n
            ytmp(k)=ya(j,k)
            y2tmp(k)=y2a(j,k)
 11      continue
         call splint(x2a,ytmp,y2tmp,n,x2,yytmp(j))
 12   continue
      call spline(x1a,yytmp,m,1.d30,1.d30,y2tmp)
      call splint(x1a,yytmp,y2tmp,m,x1,y)
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      INTEGER n,NMAX
      REAL*8 yp1,ypn,x(n),y(n),y2(n)
      PARAMETER (NMAX=200000)
C     Given arrays x(1:n) and y(1:n) containing a tabulated function,
C     i.e., yi = f(xi), with x1 < x2 < ... < xN, and given values yp1
C     and ypn for the first derivative of the inter- polating function
C     at points 1 and n, respectively, this routine returns an array
C     y2(1:n) of length n which contains the second derivatives of the
C     interpolating function at the tabulated points xi. If yp1 and/or
C     ypn are equal to 1 x 1030 or larger, the routine is signaled to
C     set the corresponding boundary condition for a natural spline,
C     with zero second derivative on that boundary.
C     Parameter: NMAX is the largest anticipated value of n.
      INTEGER i,k
      REAL*8 p,qn,sig,un,u(NMAX)
      if (yp1.gt..99d30) then !The lower boundary condition is set
         y2(1)=0.             !either to be �natural�
         u(1)=0.
      else                !or else to have a specified first derivative.
         y2(1)=-0.5
         u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
         sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
         p=sig*y2(i-1)+2.
         y2(i)=(sig-1.)/p
         u(i)=(6.*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))
     .      /(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
 11   continue
      if (ypn.gt..99d30) then !The upper boundary condition is set
         qn=0.                !either to be �natural�
         un=0.
      else           !or else to have a specified first derivative.
         qn=0.5
         un=(3./(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.)
      do 12 k=n-1,1,-1
         y2(k)=y2(k)*y2(k+1)+u(k)
 12   continue
      return
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      implicit none
      INTEGER n
      REAL*8 x,y,xa(n),y2a(n),ya(n)
C     Given the arrays xa(1:n) and ya(1:n) of length n, which tabulate a
C     function (with the xai�s in order), and given the array y2a(1:n),
C     which is the output from spline above, and given a value of x,
C     this routine returns a cubic-spline interpolated value y.
      INTEGER k,khi,klo
      REAL*8 a,b,h
      klo=1
      khi=n
 1    if (khi-klo.gt.1) then
         k=(khi+klo)/2
         if(xa(k).gt.x)then
            khi=k
         else
            klo=k
         endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) then
         write(0,*) 'bad xa input in splint' !The xa�s must be distinct.
         stop
      endif
      a=(xa(khi)-x)/h !Cubic spline polynomial is now evaluated.
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+
     .   ((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.
      return
      END

