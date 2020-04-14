subroutine readparameters(nunit,tstart,tend,exptime,deadtime,modelfile, &
   nmodeltype,rvstar,vsini,pmodelfile,emisfile,ttvfile,nplanet,nplanetmax,sol,&
   xout,yout,xcoo,ycoo,roll,xcen,ycen,xjit,yjit,rolljit,noversample,saturation,&
   ngroup, pid,onum,vnum,gnum,spseq,anumb,enum,enumos,detectorname,prodtype)
use precision
implicit none
!import vars
integer :: nmodeltype,nplanet,nplanetmax,xout,yout,noversample,ngroup,pid,onum, &
   vnum,gnum,spseq,anumb,enum,enumos,nunit,xcoo,ycoo,roll,xcen,ycen,xjit,yjit,rolljit
real(double) :: tstart,tend,exptime,deadtime,rvstar,vsini,saturation
real(double) :: sol(nplanetmax*8+1)
character(8) :: detectorname,prodtype
character(80) :: modelfile,pmodelfile(nplanetmax),emisfile(nplanetmax),ttvfile(nplanetmax)
!local parameters
integer :: i,ic,filestatus,np,icp2,ic2
real(double) :: dvalue
character(200) :: command,keyword,commandadj,to_upper

!default parameters -- these will cause the program to end quickly
tstart=0.0d0 !start time (hours)
tend=0.0d0 !end time (hours)
exptime=0.0d0 !exposure time (s)
deadtime=0.0d0 !dead time (s)
sol(1)=1.0d0 !mean stellar density (cgs)
modelfile='null' !stellar spectrum file name
nmodeltype=2 !stellar spectrum type. 1=BT-Settl, 2=Atlas-9+NL limbdarkening
rvstar=0.0d0 !radial velocity of star (km/s)
vsini=0.0d0 !projected rotation of star (km/s)
pmodelfile='null' !file with Rp/Rs values 
!nplanet is tracked by pmodelfile. 
nplanet=0 !number of planets -- default is no planets - you will get staronly sim.
sol(2)=0.0 !T0 (days)
sol(3)=1.0 !period (days)
sol(4)=0.0 !impact parameter 
sol(5)=0.0 !sqrt(e)sinw
sol(6)=0.0 !sqrt(e)cosw
sol(7)=0.0 !KRV (m/s)
sol(8)=0.0 !geometric albedo
sol(9)=0.0 !ellipsodial variations
xout=2048  !dispersion axis
yout=256   !spatial axis
noversample=1 !oversampling
saturation=65536.0d0 !saturation
ngroup=1 !samples up ramp
pid = 1 !programID
onum = 1 !observation number
vnum = 1 !visit number
gnum = 1 !group visit
spseq = 1 !parallel sequence. (1=prime, 2-5=parallel)
anumb = 1 !activity number
enum = 1 !exposure number
enumos = 1 !exposure number for oversampling
detectorname = 'NISRAPID' !convert this 
prodtype='cal'

!Default is having no transiting planets.
do i=1,nplanetmax
   pmodelfile(i)='null'
enddo
nplanet=0 !count number of planets.

i=1 !count line number
!Read in parameters..
do
   read(nunit,500,iostat=filestatus) command
   500 format(A200) !command much be contained within first 80 characters

   if(filestatus == 0) then !valid read..
   
      if((command(1:1).ne.'#').and.(command(1:1).ne.' '))then !ignore comment and blank lines
         ic=scan(command,' ')-1
         icp2=ic+2
         read(command(1:ic),*) keyword
         keyword=to_upper(keyword) !convert keyword to CAPS
         !write(0,*) 'Keyword: ',keyword(1:ic)
         select case(keyword)
         case('TSTART')
            read(command(icp2:),*) dvalue
            tstart=dvalue
         case('TEND')
            read(command(icp2:),*) dvalue
            tend=dvalue
         case('EXPTIME')
            read(command(icp2:),*) dvalue
            exptime=dvalue
         case('DEADTIME')
            read(command(icp2:),*) dvalue
            deadtime=dvalue
         case('RHOSTAR')
            read(command(icp2:),*) dvalue
            sol(1)=dvalue
         case('STARMODEL')
            commandadj=adjustl(command(icp2:))
            ic2=scan(commandadj,' ')-1
            read(commandadj(1:ic2),'(a)') modelfile
         case('STARTYPE')
            read(command(icp2:),*) dvalue
            nmodeltype=int(dvalue)
            if(nmodeltype.ne.2)then
               write(0,*) "Error: Only ATLAS-9 models are currently support (STARTYPE=2)"
               stop
            endif
         case('VSINI')
            read(command(icp2:),*) dvalue
            vsini=dvalue
         case('XOUT')
            read(command(icp2:),*) dvalue
            xout=int(dvalue)
         case('YOUT')
            read(command(icp2:),*) dvalue
            yout=int(dvalue)
         case('XCOO')
            read(command(icp2:),*) dvalue
            xcoo=dvalue
         case('YCOO')
            read(command(icp2:),*) dvalue
            ycoo=dvalue
         case('ROLL')
            read(command(icp2:),*) dvalue
            roll=dvalue
         case('XCEN')
            read(command(icp2:),*) dvalue
            xcen=dvalue
         case('YCEN')
            read(command(icp2:),*) dvalue
            ycen=dvalue
         case('XJIT')
            read(command(icp2:),*) dvalue
            xjit=dvalue
         case('YJIT')
            read(command(icp2:),*) dvalue
            yjit=dvalue
         case('ROLLJIT')
            read(command(icp2:),*) dvalue
            rolljit=dvalue
         case('OVERSAMPLE')
            read(command(icp2:),*) dvalue
            noversample=int(dvalue)
         case('SATURATION')
            read(command(icp2:),*) dvalue
            saturation=dvalue
         case('NGROUP')
            read(command(icp2:),*) dvalue
            ngroup=int(dvalue)
         case('PID')
            read(command(icp2:),*) dvalue
            pid=int(dvalue)
         case('ONUM')
            read(command(icp2:),*) dvalue
            onum=int(dvalue)
         case('VNUM')
            read(command(icp2:),*) dvalue
            vnum=int(dvalue)
         case('GNUM')
            read(command(icp2:),*) dvalue
            gnum=int(dvalue)
         case('SPSEQ')
            read(command(icp2:),*) dvalue
            spseq=int(dvalue)
         case('ANUMB')
            read(command(icp2:),*) dvalue
            anumb=int(dvalue)
         case('ENUM')
            read(command(icp2:),*) dvalue
            enum=int(dvalue)
         case('ENUMOS')
            read(command(icp2:),*) dvalue
            enumos=int(dvalue)
         case('DETECTOR')
            commandadj=adjustl(command(icp2:))
            ic2=scan(commandadj,' ')-1
            read(commandadj(1:ic2),'(a)') detectorname
            !write(6,*) "detectorname ",detectorname
         case('PRODTYPE')
            commandadj=adjustl(command(icp2:))
            ic2=scan(commandadj,' ')-1
            read(commandadj(1:ic2),'(a)') prodtype
            !write(6,*) "prodtype ",prodtype
         case default
            !handle multiplanet systems.
            if(keyword(1:ic-1).eq.'RPRSFILE')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  commandadj=adjustl(command(icp2:))
                  if(commandadj.ne.'null')then
                     nplanet=max(nplanet,np) !keep track of how many planets we have.
                     ic2=scan(commandadj,' ')-1
                     read(commandadj(1:ic2),'(a)') pmodelfile(np)
                  !else
                  !   write(0,*) 'Skipping pmodelfile entry'
                  endif
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif(keyword(1:ic-1).eq.'EMISFILE')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  commandadj=adjustl(command(icp2:))
                  ic2=scan(commandadj,' ')-1
                  read(commandadj(1:ic2),'(a)') emisfile(np)
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif(keyword(1:ic-1).eq.'TTVFILE')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  commandadj=adjustl(command(icp2:))
                  ic2=scan(commandadj,' ')-1
                  read(commandadj(1:ic2),'(a)') ttvfile(np)
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'EP')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+2)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'PE')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+3)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'BB')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+4)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'ES')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+5)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'EC')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+6)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'RV')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+7)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'AL')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+8)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            elseif (keyword(1:ic-1).eq.'EL')then
               read(keyword(ic:ic),*) np !get planet number
               if((np.le.9).and.(np.gt.0))then
                  read(command(icp2:),*) dvalue
                  sol(8*(np-1)+9)=dvalue
               else
                  write(0,*) trim(command)
                  write(0,*) 'Error: Planet number is Invalid ',np 
               endif
            else
               write(0,*) 'Warning: Invalid KEYWORD: ', keyword(1:ic)
            endif
         end select

      endif

      i=i+1
      cycle
   elseif(filestatus == -1) then
      exit
   else
      write(0,*) "File Error!!"
      write(0,900) "iostat: ",filestatus
      900 format(A8,I3)
      stop
   endif
enddo

!convert all times to days
tstart=tstart/24.0d0 !hours -> days
tend=tend/24.0d0     !hours -> days
exptime=exptime/86400.0   !seconds -> days
deadtime=deadtime/86400.0 !seconds -> days

!check if modelfile is null.  If so, exit, because there is nothing to do.

return
end

