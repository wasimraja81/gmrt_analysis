chelp+
      !-----------------------------------------------------------
      ! This is a robot to read and interpret the output of UVFND, 
      ! run with, say OPCODE 'CLIP', and to determine the FLAGGING 
      ! STRING to be written to an AIPS readable FLAG FILE. 
      !
      ! This code outputs a "compact" statement of bad baselines, 
      ! and a summary (over all times) of bad antenna-baselines in 
      ! the form of an ANTE-BASE matrix. 
      !                                  -- wr, 20 July, 2011
      ! Caveats: 
      !     1) AIPS requires a flag line to not exceed 200 chars. 
      !        Our compactifying the flag commands, encounters 
      !        cases where the length of the line exceeds limit. 
      !        Hence this code now does away with the "REASON" 
      !        string, the SOURCE string as well as the extra 
      !        formatting-related SPACE characters. 
      !                                  --wr, 25 Jul, 2011
      !-----------------------------------------------------------
chelp-



      implicit none

      integer*4      maxant, max_badsamp, maxbase, maxscan 
      parameter      (max_badsamp = 8196, maxscan = 20)
      parameter      (maxant=32, maxbase = maxant*(maxant+1)/2)
      character      infile*172,outfile*172, data_tag*64,parfile*200 , 
     -               summary_file*172, stokes_tag*4, outdir*132, 
     -               indir*132, base_all*180
      integer*4      visnum
      integer*4      dd, hh, mm, ss, 
     -               dd1, hh1, mm1, ss1, 
     -               dd2, hh2, mm2, ss2, 
     -               ant1, ant2, tsamp_sec, tol_sec 
      real*4         U, V, W, amp
      integer*4      phas_deg, wt, chan_num
      integer*4      ant_mat(maxant,maxant), nante, nbase
      integer*4      ant_mat3(maxant,maxant,maxscan)
      integer*4      nchar
      integer*4      i, j, k, icnt, jcnt, itmp , nflags, isamp, iscan 
      integer*4      tdiff, i1, i2, icomp , iall, ibad, itimes 
      integer*4      nbase_now,time_val_sec(maxbase,max_badsamp), 
     -               n_badsamp_tot(maxbase), 
     -               n_badsamp_scan(maxbase,maxscan)
      integer*4      time_scan_sec(maxbase,maxscan,max_badsamp) 
      character      junk_char*1, templine*220
      integer*4      maxbad_allowed

      character      ante_str*16, base_str*160,time_str*64,
     -               stokes_str*16,
     -               bchan_str*12, echan_str*12, opcod_str*16,
     -               reason_str*64, flag_str*220, source_str*64
      integer*4      ante(maxbase), base(maxbase)

      integer*4       dd_0(maxscan), hh_0(maxscan), mm_0(maxscan), 
     -                ss_0(maxscan), t0(maxscan),  
     -                nscans 
      integer*4       ilim, iante, ibase, nsamp_per_scan 
      real*4          scan_len 
      integer*4       ibad_ante(maxant), ibad_scan
      integer*4       bad_scan(maxscan), bad_ante(maxscan,maxant)
      ! The scan intervals: 
!      DATA    dd_0(1),dd_0(2),dd_0(3),dd_0(4),dd_0(5)/ 1, 1, 1, 1, 1/
!      DATA    hh_0(1),hh_0(2),hh_0(3),hh_0(4),hh_0(5)/10,12,13,14,15/
!      DATA    mm_0(1),mm_0(2),mm_0(3),mm_0(4),mm_0(5)/30, 0, 0, 0,30/
!      DATA    ss_0(1),ss_0(2),ss_0(3),ss_0(4),ss_0(5)/ 0, 0, 0, 0, 0/
!
!      DATA    dd_0(6),dd_0(7),dd_0(8),dd_0(9),dd_0(10)/ 1, 1, 1, 1, 1/
!      DATA    hh_0(6),hh_0(7),hh_0(8),hh_0(9),hh_0(10)/16,18,19,20,22/
!      DATA    mm_0(6),mm_0(7),mm_0(8),mm_0(9),mm_0(10)/30, 0, 0, 0,20/
!      DATA    ss_0(6),ss_0(7),ss_0(8),ss_0(9),ss_0(10)/ 0, 0, 0, 0, 0/
!
      !--------------------------------------------------
      if(iargc().lt.4)then
              write(*,*)"Usage: "
              write(*,*)" my_uvflg <srcname> <stokes_tag> <chan_num> <pa
     -rfile>"
              write(*,*)" "
              write(*,*)"1)<srcname> should match its name in AIPS. "
              write(*,*)"     The output filename of UVFND should as "
              write(*,*)"     such be carefully named to satisfy the "
              write(*,*)"     above requirement of srcname match."
              write(*,*)" "
              write(*,*)"2)<stokes_tag> indicates the Stokes based on "
              write(*,*)"     which UVFND printed the visibilities to "
              write(*,*)"     be used for flagging."
              write(*,*)" "
              write(*,*)"3)<chan_num> is the channel for which the "
              write(*,*)"     flag file is to be written. "
              write(*,*)" "
              write(*,*)"4)<parfile> File containing parmaeters."
              write(*,*)"     [This file must be present in PAR/]"
              write(*,*)" "
              stop
      endif
      call getarg(1,data_tag)
      call getarg(2,stokes_tag)  ! Flagging based on which Stokes? 
      call getarg(3,templine)
      read(templine,*)chan_num
      infile = data_tag(1:nchar(data_tag))//'_'//
     -         stokes_tag(1:nchar(stokes_tag))//'_'//
     -         templine(1:nchar(templine))//'.UVFND'
       call getarg(4,parfile)
       parfile = '../PAR/'//parfile(1:nchar(parfile))
       open(41,file=parfile,status='old',err=104)
       goto 105
104    write(*,*)"Error opening the parfile: ",
     -                 parfile(1:nchar(parfile))
       write(*,*)" Quitting now... "
       write(*,*)" "
       stop
105    continue
       read(41,*)junk_char
       read(41,*)junk_char
       read(41,*)junk_char
       read(41,*)nscans
       read(41,*)scan_len   ! mins
       read(41,*)(dd_0(i),i=1,nscans)
       read(41,*)(hh_0(i),i=1,nscans)
       read(41,*)(mm_0(i),i=1,nscans)
       read(41,*)(ss_0(i),i=1,nscans)
       read(41,*)junk_char
       read(41,*)tsamp_sec
       read(41,*)nante
       read(41,*)maxbad_allowed
       read(41,*)tol_sec
       read(41,'(a)')templine
       indir = templine(1:index(templine,';')-1)
       indir = indir(1:nchar(indir))
       read(41,'(a)')templine
       outdir = templine(1:index(templine,';')-1)
       outdir = outdir(1:nchar(outdir))
       templine = "mkdir "//outdir(1:nchar(outdir))
       call system(templine(1:nchar(templine)))
       close(41)

       nbase = nante*(nante+1)/2
       nsamp_per_scan = int(scan_len*60/tsamp_sec)*nbase ! total vis. (per chan)
                                                         ! per scan


!      ! Set some paramters here (use command line/parfile 
!      !                          later):
!      nante = 30
!      maxbad_allowed = 5 ! If this many samples for a baseline-pair 
!                          ! are found bad, the entire baseline-pair 
!                          ! for all time will be regarded bad for the 
!                          ! source(s) at hand and for the channel in 
!                          ! conseideration. 
!      tsamp_sec = 2
!      nscans = 10           ! No. of scans (Cal + target)

      do i = 1,nscans
         t0(i) = dd_0(i)*86400 + hh_0(i)*3600 + mm_0(i)*60 + ss_0(i)
      enddo

      if(chan_num.gt.0.and.chan_num.lt.10)then
              write(outfile,'(I1)')chan_num
      else if(chan_num.ge.10.and.chan_num.lt.100)then
              write(outfile,'(I2)')chan_num
      else if(chan_num.ge.100.and.chan_num.lt.1000)then
              write(outfile,'(I3)')chan_num
      else
              write(*,*)"channel number exceeds max chan!!"
              write(*,*)"Quitting now..."
              stop
      endif


      outfile = outdir(1:nchar(outdir))//'/'//
     -          infile(1:index(infile,'UVFND')-1)//'FLG'
      summary_file = outdir(1:nchar(outdir))//'/'//
     -          infile(1:index(infile,'UVFND')-1)//'SUM'
      infile = indir(1:nchar(indir))//'/'//infile(1:nchar(infile))

      open(11,file=outfile,status='unknown') 
      open(31,file=summary_file,status='unknown') 


      !source_str = "'3C468.1'"
      source_str = "'"//data_tag(1:nchar(data_tag))//"'"
      reason_str = ' REASON '//"''"
      !write(bchan_str,'(I3)')chan_num
      !write(echan_str,'(I3)')chan_num
      write(bchan_str,*)chan_num
      write(echan_str,*)chan_num
      !--------------------------------------------------

      !source_str = 'SOURCES '//source_str(1:nchar(source_str))
      source_str = 'SOURCES '//"''"
      stokes_str = ' STOKES '//"''"
      bchan_str = ' BCHAN'//bchan_str(1:nchar(bchan_str))
      echan_str = ' ECHAN'//echan_str(1:nchar(echan_str))
      opcod_str = ' OPCODE '//"'FLAG'"

      open(21,file=infile,status='old',err=101) 
      goto 102
101   write(*,*)'FILE NOT FOUND: ',infile(1:nchar(infile))
      write(*,*)"Quitting now..."
      stop
102   continue

      do i = 1,nante
         do j = 1,nante
            ant_mat(i,j) = 0
         enddo
      enddo
      ! Store the ante-base corrsponding to the 
      ! baseline num: 
      icnt = 0
      do i = 1,nante
         do j = i,nante
            icnt = icnt + 1
            ante(icnt) = i
            base(icnt) = j
         enddo
      enddo

      write(*,*)"icnt: ",icnt
      ! Some initialisations:
      do i = 1,nante
         do j = 1,nante
            do k = 1,nscans
               ant_mat3(i,j,k) = 0
            enddo
         enddo
      enddo
      ! Array to store the number of bad points 
      ! per baseline: 
      do i = 1,maxbase
         n_badsamp_tot(i) = 0
         do iscan = 1,maxscan
            n_badsamp_scan(i,iscan) = 0
         enddo
      enddo
      isamp = 0
      do while (.true.)
         read(21,'(a)',end=201)templine
         if(nchar(templine).le.0)then 
                 goto 200
         endif
         !if(index(templine,'Points found').gt.0)then
         !        goto 200
         !endif

         read(templine,fmt=1140,err=200,end=201)visnum,dd,
     -                    junk_char,hh,
     -                    junk_char,mm,junk_char,ss,
     -                    ant1,junk_char,ant2,U,V,W,
     -                    amp,phas_deg,wt
         isamp = isamp + 1
         !write(*,'(a)')templine(1:nchar(templine))

         ant_mat(ant1,ant2) = ant_mat(ant1,ant2) + 1
         ! Compute the baseline number given the antennas:
         nbase_now = nante*(ant1-1) - (ant1-2)*(ant1-1)/2 + 
     -                     (ant2 - ant1 + 1)
         n_badsamp_tot(nbase_now) = n_badsamp_tot(nbase_now) + 1
         icnt = n_badsamp_tot(nbase_now)

         time_val_sec(nbase_now,icnt) = dd*86400 + hh*3600 + mm*60 + ss
         itmp = time_val_sec(nbase_now,icnt)
         ! Distribute the bad vis. into the time bins (cal-scan +
         ! target-scan): 
         do iscan = 1,nscans-1
            if(itmp.ge.t0(iscan).and.itmp.lt.t0(iscan+1))then
                    n_badsamp_scan(nbase_now,iscan) =
     -                       n_badsamp_scan(nbase_now,iscan) + 1
                    jcnt = n_badsamp_scan(nbase_now,iscan)
                    ant_mat3(ant1,ant2,iscan)=ant_mat3(ant1,ant2,iscan)
     -                                              + 1
                    time_scan_sec(nbase_now,iscan,jcnt) = itmp
            endif
         enddo
200      continue
      enddo

201   continue
      close(21)

      !---------------------------------------------
      ! Write the bad ante-base matrix for each scan
      !
      do iscan = 1,nscans-1
         dd1 = int(t0(iscan)/86400)
         itmp = t0(iscan) - dd1*86400 ! remaining secs

         hh1 = int(itmp/3600) 
         itmp = itmp - hh1*3600 ! remaining secs

         mm1 = int(itmp/60)
         ss1 = itmp - mm1*60 ! remaining secs

         dd2 = int(t0(iscan+1)/86400)
         itmp = t0(iscan+1) - dd2*86400 ! remaining secs

         hh2 = int(itmp/3600) 
         itmp = itmp - hh2*3600 ! remaining secs

         mm2 = int(itmp/60)
         ss2 = itmp - mm2*60 ! remaining secs

         write(*,*)"  "
         write(*,1150)iscan,dd1,hh1,mm1,ss1,dd2,hh2,mm2,ss2
         write(*,fmt=1143)(i,i=1,nante)
         write(*,1144)
         write(31,*)"  "
         write(31,1150)iscan,dd1,hh1,mm1,ss1,dd2,hh2,mm2,ss2
         write(31,fmt=1143)(i,i=1,nante)
         write(31,1144)
         do iante = 1,nante
            write(*,fmt=1142)iante,
     -                       (ant_mat3(iante,ibase,iscan),ibase=1,nante)
            write(31,fmt=1142)iante,
     -                       (ant_mat3(iante,ibase,iscan),ibase=1,nante)
         enddo
         write(*,1144)
      enddo
      write(*,1144)
      write(31,1144)
      !---------------------------------------------
      ! The matrix for the entire time (Cumulative): 
      write(*,*)" Cumulative Summary: "
      write(*,fmt=1143)(iante,iante=1,nante)
      write(*,1144)
      ! Note the summary in a file: 
      write(31,*)" Cumulative Summary: "
      write(31,fmt=1143)(iante,iante=1,nante)
      write(31,1144)
      
      do iante = 1,nante
         write(*,fmt=1142)iante,(ant_mat(iante,ibase),ibase=1,nante)
         write(31,fmt=1142)iante,(ant_mat(iante,ibase),ibase=1,nante)
      enddo
      !---------------------------------------------
      ! Sniff the hopelessly bad SCANS and ANTENNAS: 
      ! 1) SCANS: 
      write(*,*)"Figuring out the hopelessly bad scans..."
      do iscan = 1,nscans-1
         ibad_scan = 0
         do iante = 1,nante
            do ibase = iante,nante
               itmp = ant_mat3(iante,ibase,iscan)
               if(itmp.gt.0)then
                       ibad_scan = ibad_scan + itmp 
               endif
            enddo
         enddo
         ! Now decide if the entire scan was bad:
         ! [flag scan if bad for more than 70% visibilities]
         ilim = int(0.7*nbase*nsamp_per_scan) ! for a single channel
         if(ibad_scan.ge.ilim)then
                 bad_scan(iscan) = 1 ! Scan is bad
         else
                 bad_scan(iscan) = 0 ! Some part good
         endif
      enddo
      ! 2) ANTENNAS: 
      write(*,*)"Figuring out the hopelessly bad antennas..."
      do iscan = 1,nscans - 1
         do iante = 1,nante
            ibad_ante(iante) = 0
         enddo
         do iante = 1,nante
            do ibase = 1,iante - 1
               itmp = ant_mat3(ibase,iante,iscan)
               if(itmp.gt.0)then
                       ibad_ante(iante) = ibad_ante(iante) +
     -                                                     itmp 
               endif
            enddo
            do ibase = iante,nante
               itmp = ant_mat3(iante,ibase,iscan)
               if(itmp.gt.0)then
                       ibad_ante(iante) = ibad_ante(iante) +
     -                                                     itmp 
               endif
            enddo
         enddo
         ! Now decide if the ante was bad throughout the scan:
         ! [flag ante if bad with more than 60% baselines]
         ilim = int(0.6*nante*scan_len*60.0/real(tsamp_sec)) ! for a single channel
         do iante = 1,nante
            if(ibad_ante(iante).ge.ilim)then
                 bad_ante(iscan,iante) = 1 ! Scan is bad
            else
                 bad_ante(iscan,iante) = 0 ! Some part good
            endif
         enddo
         ! Another way of finding out a BAD ANTE is to see if 
         ! it remains bad with all baselines for imore than 
         ! max_allowed number of points within the scan: 
         do iante = 1,nante
            itimes = 0
            do ibase = 1,iante-1
               itmp = ant_mat3(ibase,iante,iscan)
               if(itmp.gt.maxbad_allowed)then
                   itimes = itimes + 1
               endif
            enddo
            do ibase = iante,nante
               itmp = ant_mat3(iante,ibase,iscan)
               if(itmp.gt.maxbad_allowed)then
                   itimes = itimes + 1
               endif
            enddo
            ilim = int(0.5*nante)
            if (itimes .gt. ilim)then
                    bad_ante(iscan,iante) = 1
            endif
         enddo
      enddo
      !---------------------------------------------
      ! Write the Flag file now: 
      ! 
      nflags = 0
      do iscan = 1,nscans-1
         ! ------------------------------------------
         ! Mere comments in flag file: 
         dd1 = int(t0(iscan)/86400)
         itmp = t0(iscan) - dd1*86400 ! remaining secs

         hh1 = int(itmp/3600) 
         itmp = itmp - hh1*3600 ! remaining secs

         mm1 = int(itmp/60)
         ss1 = itmp - mm1*60 ! remaining secs

         dd2 = int(t0(iscan+1)/86400)
         itmp = t0(iscan+1) - dd2*86400 ! remaining secs

         hh2 = int(itmp/3600) 
         itmp = itmp - hh2*3600 ! remaining secs

         mm2 = int(itmp/60)
         ss2 = itmp - mm2*60 ! remaining secs
         write(11,1151)iscan,dd1,hh1,mm1,ss1,dd2,hh2,mm2,ss2
         if(dd1.gt.0.and.dd1.lt.10)then
                 write(time_str,'(I1,A1)')dd1,','
         else if(dd1.ge.10.and.dd1.lt.100)then
                 write(time_str,'(I2,A1)')dd1,','
         else if(dd1.ge.100.and.dd1.lt.1000)then
                 write(time_str,'(I3,A1)')dd1,','
         else 
                 write(*,*)"dd1 unreasonably high!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         if(hh1.ge.0.and.hh1.lt.10)then
                 write(templine,'(I1,A1)')hh1,','
         else if(hh1.ge.10.and.hh1.lt.100)then
                 write(templine,'(I2,A1)')hh1,','
         else 
                 write(*,*)"hh unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(mm1.ge.0.and.mm1.lt.10)then
                 write(templine,'(I1,A1)')mm1,','
         else if(mm1.ge.10.and.mm1.lt.100)then
                 write(templine,'(I2,A1)')mm1,','
         else 
                 write(*,*)"mm unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(ss1.ge.0.and.ss1.lt.10)then
                 write(templine,'(I1,A1)')ss1,','
         else if(ss1.ge.10.and.ss1.lt.100)then
                 write(templine,'(I2,A1)')ss1,','
         else 
                 write(*,*)"ss unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(dd2.gt.0.and.dd2.lt.10)then
                 write(templine,'(I1,A1)')dd2,','
         else if(dd2.ge.10.and.dd2.lt.100)then
                 write(templine,'(I2,A1)')dd2,','
         else if(dd2.ge.100.and.dd2.lt.1000)then
                 write(templine,'(I3,A1)')dd2,','
         else 
                 write(*,*)"dd2 unreasonably high!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(hh2.ge.0.and.hh2.lt.10)then
                 write(templine,'(I1,A1)')hh2,','
         else if(hh2.ge.10.and.hh2.lt.100)then
                 write(templine,'(I2,A1)')hh2,','
         else 
                 write(*,*)"hh unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(mm2.ge.0.and.mm2.lt.10)then
                 write(templine,'(I1,A1)')mm2,','
         else if(mm2.ge.10.and.mm2.lt.100)then
                 write(templine,'(I2,A1)')mm2,','
         else 
                 write(*,*)"mm unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         if(ss2.ge.0.and.ss2.lt.10)then
                 write(templine,'(I1)')ss2
         else if(ss2.ge.10.and.ss2.lt.100)then
                 write(templine,'(I2)')ss2
         else 
                 write(*,*)"ss unreasonable!"
                 write(*,*)"Quitting now..."
                 stop
         endif
         time_str = " TIMERANGE "//time_str(1:nchar(time_str))//
     -              templine(1:nchar(templine))
         ! If this is a bad scan, flag the entire scan:
         if(bad_scan(iscan).eq.1)then
                 ! write the flag command
                 ! and skip to the next scan
                 ante_str = "ANTE 0"
                 base_str = "BASE 0"
                 time_str = time_str(1:nchar(time_str))
                 flag_str = source_str(1:nchar(source_str))//
     -           stokes_str(1:nchar(stokes_str))//
     -           ante_str(1:nchar(ante_str))//
     -           base_str(1:nchar(base_str))//
     -           time_str(1:nchar(time_str))//
     -           bchan_str(1:nchar(bchan_str))// 
     -           echan_str(1:nchar(echan_str))// 
     -           opcod_str(1:nchar(opcod_str))// 
     -           reason_str(1:nchar(reason_str))//' /'

                 write(11,*)flag_str(1:nchar(flag_str))
                 nflags = nflags + 1

                 goto 601 
         endif
         ! ------------------------------------------
         nbase_now = 0
         do iante = 1,nante
            if(iante.gt.0.and.iante.lt.10)then
                    write(ante_str,'(I1)')iante
            else if(iante.ge.10.and.iante.lt.100)then
                    write(ante_str,'(I2)')iante
            else if(iante.ge.100.and.iante.lt.1000)then
                    write(ante_str,'(I3)')iante
            else
                    write(*,*)"ante num exceeds maxante!"
                    write(*,*)"Quitting now..."
                    stop
            endif
            ante_str = ' ANTE='//ante_str(1:nchar(ante_str))
            base_all = ' BASE='
            if(bad_ante(iscan,iante).eq.1)then
                    ! write the flag command
                    ! and skip to the next ante
                    dd1 = int(t0(iscan)/86400)
                    itmp = t0(iscan) - dd1*86400 ! remaining secs
           
                    hh1 = int(itmp/3600) 
                    itmp = itmp - hh1*3600 ! remaining secs
           
                    mm1 = int(itmp/60)
                    ss1 = itmp - mm1*60 ! remaining secs
           
                    dd2 = int(t0(iscan+1)/86400)
                    itmp = t0(iscan+1) - dd2*86400 ! remaining secs
           
                    hh2 = int(itmp/3600) 
                    itmp = itmp - hh2*3600 ! remaining secs
           
                    mm2 = int(itmp/60)
                    ss2 = itmp - mm2*60 ! remaining secs
                    if(dd1.gt.0.and.dd1.lt.10)then
                            write(time_str,'(I1,A1)')dd1,','
                    else if(dd1.ge.10.and.dd1.lt.100)then
                            write(time_str,'(I2,A1)')dd1,','
                    else if(dd1.ge.100.and.dd1.lt.1000)then
                            write(time_str,'(I3,A1)')dd1,','
                    else 
                            write(*,*)"dd1 unreasonably high!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    if(hh1.ge.0.and.hh1.lt.10)then
                            write(templine,'(I1,A1)')hh1,','
                    else if(hh1.ge.10.and.hh1.lt.100)then
                            write(templine,'(I2,A1)')hh1,','
                    else 
                            write(*,*)"hh unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(mm1.ge.0.and.mm1.lt.10)then
                            write(templine,'(I1,A1)')mm1,','
                    else if(mm1.ge.10.and.mm1.lt.100)then
                            write(templine,'(I2,A1)')mm1,','
                    else 
                            write(*,*)"mm unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(ss1.ge.0.and.ss1.lt.10)then
                            write(templine,'(I1,A1)')ss1,','
                    else if(ss1.ge.10.and.ss1.lt.100)then
                            write(templine,'(I2,A1)')ss1,','
                    else 
                            write(*,*)"ss unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(dd2.gt.0.and.dd2.lt.10)then
                            write(templine,'(I1,A1)')dd2,','
                    else if(dd2.ge.10.and.dd2.lt.100)then
                            write(templine,'(I2,A1)')dd2,','
                    else if(dd2.ge.100.and.dd2.lt.1000)then
                            write(templine,'(I3,A1)')dd2,','
                    else 
                            write(*,*)"dd2 unreasonably high!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(hh2.ge.0.and.hh2.lt.10)then
                            write(templine,'(I1,A1)')hh2,','
                    else if(hh2.ge.10.and.hh2.lt.100)then
                            write(templine,'(I2,A1)')hh2,','
                    else 
                            write(*,*)"hh unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(mm2.ge.0.and.mm2.lt.10)then
                            write(templine,'(I1,A1)')mm2,','
                    else if(mm2.ge.10.and.mm2.lt.100)then
                            write(templine,'(I2,A1)')mm2,','
                    else 
                            write(*,*)"mm unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(ss2.ge.0.and.ss2.lt.10)then
                            write(templine,'(I1)')ss2
                    else if(ss2.ge.10.and.ss2.lt.100)then
                            write(templine,'(I2)')ss2
                    else 
                            write(*,*)"ss unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = " TIMERANGE "//
     -                         time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    base_str = " BASE 0"
                    time_str =time_str(1:nchar(time_str))
                    flag_str = source_str(1:nchar(source_str))//
     -              stokes_str(1:nchar(stokes_str))//
     -              ante_str(1:nchar(ante_str))//
     -              base_str(1:nchar(base_str))//
     -              time_str(1:nchar(time_str))//
     -              bchan_str(1:nchar(bchan_str))// 
     -              echan_str(1:nchar(echan_str))// 
     -              opcod_str(1:nchar(opcod_str))// 
     -              reason_str(1:nchar(reason_str))//' /'

                    write(11,*)flag_str(1:nchar(flag_str))
                    nflags = nflags + 1
                    nbase_now = nbase_now + (nante-iante+1) ! this many baselines
                                                            ! skipped
                    goto 602
            endif
            iall = 0
            do ibase = iante,nante
              nbase_now = nbase_now + 1               
              ! If baseline formed with bad antenna, skip
              if(bad_ante(iscan,ibase).eq.1)then
                      goto 300
              endif
              if(ant_mat3(iante,ibase,iscan) .eq. 0)then
                      goto 300 ! No flagging required
              else if(ant_mat3(iante,ibase,iscan).ge.maxbad_allowed)then
                      ! Accumulate the baselines that were bad
                      ! for the entire scan length to make the flag
                     ! command compact: 
                      iall = iall + 1
                      if(ibase.gt.0.and.ibase.lt.10)then
                              write(templine,'(I1)')ibase
                      else if(ibase.ge.10.and.ibase.lt.100)then
                              write(templine,'(I2)')ibase
                      else if(ibase.ge.100.and.ibase.lt.1000)then
                              write(templine,'(I2)')ibase
                      else
                              write(*,*)"Antennas exceed maxant!"
                              write(*,*)"Quitting now..."
                              stop
                      endif
                      base_all = base_all(1:nchar(base_all))//
     -                           templine(1:nchar(templine))//','
              else 
                      write(base_str,'(I2)')ibase
                      base_str = ' BASE='//base_str(1:nchar(base_str))
                      ! Write immediately into the flag file:  
                      ibad = ant_mat3(iante,ibase,iscan) ! No. of times the bad
                                                 ! baseline occured 
                      ! Flag smartly :
                      i1 = 1
                      i2 = ibad !n_badsamp_tot(i)
                      icomp = 1
                      do while (icomp .gt. 0)
                         tdiff = time_scan_sec(nbase_now,iscan,i2)-
     -                           time_scan_sec(nbase_now,iscan,i1)
                         if(tdiff.le.(i2-i1)*tsamp_sec+tol_sec)then
                           itmp = time_scan_sec(nbase_now,iscan,i1) - 
     -                                     tsamp_sec
                           dd1 = int(itmp/86400)
                           itmp = itmp - dd1*86400 ! remaining secs

                           hh1 = int(itmp/3600) 
                           itmp = itmp - hh1*3600 ! remaining secs

                           mm1 = int(itmp/60)
                           ss1 = itmp - mm1*60 ! remaining secs

                           itmp = time_scan_sec(nbase_now,iscan,i2) + 
     -                                     tsamp_sec
                           dd2 = int(itmp/86400)
                           itmp = itmp - dd2*86400 ! remaining secs

                           hh2 = int(itmp/3600) 
                           itmp = itmp - hh2*3600 ! remaining secs

                           mm2 = int(itmp/60)
                           ss2 = itmp - mm2*60 ! remaining secs
                           if(dd1.gt.0.and.dd1.lt.10)then
                                   write(time_str,'(I1,A1)')dd1,','
                           else if(dd1.ge.10.and.dd1.lt.100)then
                                   write(time_str,'(I2,A1)')dd1,','
                           else if(dd1.ge.100.and.dd1.lt.1000)then
                                   write(time_str,'(I3,A1)')dd1,','
                           else 
                                   write(*,*)"dd unreasonably high!"
                                   stop
                           endif
                           if(hh1.ge.0.and.hh1.lt.10)then
                                   write(templine,'(I1,A1)')hh1,','
                           else if(hh1.ge.10.and.hh1.lt.100)then
                                   write(templine,'(I2,A1)')hh1,','
                           else 
                                   write(*,*)"hh unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(mm1.ge.0.and.mm1.lt.10)then
                                   write(templine,'(I1,A1)')mm1,','
                           else if(mm1.ge.10.and.mm1.lt.100)then
                                   write(templine,'(I2,A1)')mm1,','
                           else 
                                   write(*,*)"mm unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(ss1.ge.0.and.ss1.lt.10)then
                                   write(templine,'(I1,A1)')ss1,','
                           else if(ss1.ge.10.and.ss1.lt.100)then
                                   write(templine,'(I2,A1)')ss1,','
                           else 
                                   write(*,*)"ss unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(dd2.gt.0.and.dd2.lt.10)then
                                   write(templine,'(I1,A1)')dd2,','
                           else if(dd2.ge.10.and.dd2.lt.100)then
                                   write(templine,'(I2,A1)')dd2,','
                           else if(dd2.ge.100.and.dd2.lt.1000)then
                                   write(templine,'(I3,A1)')dd2,','
                           else 
                                   write(*,*)"dd unreasonably high!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(hh2.ge.0.and.hh2.lt.10)then
                                   write(templine,'(I1,A1)')hh2,','
                           else if(hh2.ge.10.and.hh2.lt.100)then
                                   write(templine,'(I2,A1)')hh2,','
                           else 
                                   write(*,*)"hh unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(mm2.ge.0.and.mm2.lt.10)then
                                   write(templine,'(I1,A1)')mm2,','
                           else if(mm2.ge.10.and.mm2.lt.100)then
                                   write(templine,'(I2,A1)')mm2,','
                           else 
                                   write(*,*)"mm unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))
                           if(ss2.ge.0.and.ss2.lt.10)then
                                   write(templine,'(I1)')ss2
                           else if(ss2.ge.10.and.ss2.lt.100)then
                                   write(templine,'(I2)')ss2
                           else 
                                   write(*,*)"ss unreasonable!"
                                   write(*,*)"Quitting now..."
                                   stop
                           endif
                           time_str = time_str(1:nchar(time_str))//
     -                                templine(1:nchar(templine))

!                           write(time_str,fmt=1145)dd1,',',hh1,',',
!     -                                         mm1,',',ss1-tsamp_sec,
!     -                                             ',',
!     -                                             dd2,',',hh2,',',
!     -                                         mm2,',',ss2+tsamp_sec
                           time_str = ' TIMERANGE='//
     -                                time_str(1:nchar(time_str))
                           flag_str = source_str(1:nchar(source_str))//
     -                        stokes_str(1:nchar(stokes_str))//
     -                        ante_str(1:nchar(ante_str))//
     -                        base_str(1:nchar(base_str))//
     -                        time_str(1:nchar(time_str))//
     -                        bchan_str(1:nchar(bchan_str))// 
     -                        echan_str(1:nchar(echan_str))// 
     -                        opcod_str(1:nchar(opcod_str))// 
     -                        reason_str(1:nchar(reason_str))//' /'

                           write(11,*)flag_str(1:nchar(flag_str))
                           nflags = nflags + 1

                           if (i2.eq.ibad)then
                                   icomp = -1
                           else if (i2.lt.ibad)then
                                   i1 = i2+1
                                   i2 = ibad
                           endif
                         else
                           i2 = i2 -1
                         endif
                      enddo
              endif
300           continue
            enddo  ! End of ibase-loop (baselines)
            if(iall .gt. 0)then
                    ! Re-compute the beg and end times for the scans: 
                    ! The recomputation is done since its only a 
                    ! small overhead (I do not have the mood of defining 
                    ! a fresh set of variables to save computation! )

                    dd1 = int(t0(iscan)/86400)
                    itmp = t0(iscan) - dd1*86400 ! remaining secs

                    hh1 = int(itmp/3600) 
                    itmp = itmp - hh1*3600 ! remaining secs

                    mm1 = int(itmp/60)
                    ss1 = itmp - mm1*60 ! remaining secs

                    dd2 = int(t0(iscan+1)/86400)
                    itmp = t0(iscan+1) - dd2*86400 ! remaining secs

                    hh2 = int(itmp/3600) 
                    itmp = itmp - hh2*3600 ! remaining secs

                    mm2 = int(itmp/60)
                    ss2 = itmp - mm2*60 ! remaining secs
                    if(dd1.gt.0.and.dd1.lt.10)then
                            write(time_str,'(I1,A1)')dd1,','
                    else if(dd1.ge.10.and.dd1.lt.100)then
                            write(time_str,'(I2,A1)')dd1,','
                    else if(dd1.ge.100.and.dd1.lt.1000)then
                            write(time_str,'(I3,A1)')dd1,','
                    else 
                            write(*,*)"dd unreasonably high!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    if(hh1.ge.0.and.hh1.lt.10)then
                            write(templine,'(I1,A1)')hh1,','
                    else if(hh1.ge.10.and.hh1.lt.100)then
                            write(templine,'(I2,A1)')hh1,','
                    else 
                            write(*,*)"hh unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(mm1.ge.0.and.mm1.lt.10)then
                            write(templine,'(I1,A1)')mm1,','
                    else if(mm1.ge.10.and.mm1.lt.100)then
                            write(templine,'(I2,A1)')mm1,','
                    else 
                            write(*,*)"mm unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(ss1.ge.0.and.ss1.lt.10)then
                            write(templine,'(I1,A1)')ss1,','
                    else if(ss1.ge.10.and.ss1.lt.100)then
                            write(templine,'(I2,A1)')ss1,','
                    else 
                            write(*,*)"ss unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(dd2.gt.0.and.dd2.lt.10)then
                            write(templine,'(I1,A1)')dd2,','
                    else if(dd2.ge.10.and.dd2.lt.100)then
                            write(templine,'(I2,A1)')dd2,','
                    else if(dd2.ge.100.and.dd2.lt.1000)then
                            write(templine,'(I3,A1)')dd2,','
                    else 
                            write(*,*)"dd unreasonably high!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(hh2.ge.0.and.hh2.lt.10)then
                            write(templine,'(I1,A1)')hh2,','
                    else if(hh2.ge.10.and.hh2.lt.100)then
                            write(templine,'(I2,A1)')hh2,','
                    else 
                            write(*,*)"hh unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(mm2.ge.0.and.mm2.lt.10)then
                            write(templine,'(I1,A1)')mm2,','
                    else if(mm2.ge.10.and.mm2.lt.100)then
                            write(templine,'(I2,A1)')mm2,','
                    else 
                            write(*,*)"mm unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                         templine(1:nchar(templine))
                    if(ss2.ge.0.and.ss2.lt.10)then
                            write(templine,'(I1)')ss2
                    else if(ss2.ge.10.and.ss2.lt.100)then
                            write(templine,'(I2)')ss2
                    else 
                            write(*,*)"ss unreasonable!"
                            write(*,*)"Quitting now..."
                            stop
                    endif
                    time_str = time_str(1:nchar(time_str))//
     -                                 templine(1:nchar(templine))
!                    write(time_str,fmt=1145)dd1,',',hh1,',',
!     -                                  mm1,',',ss1,
!     -                                      ',',
!     -                                      dd2,',',hh2,',',
!     -                                  mm2,',',ss2
                    time_str = ' TIMERANGE='//
     -                         time_str(1:nchar(time_str))
                    base_str = base_all(1:nchar(base_all)-1) ! remove the "comma" at the end
                    flag_str = source_str(1:nchar(source_str))//
     -                 stokes_str(1:nchar(stokes_str))//
     -                 ante_str(1:nchar(ante_str))//
     -                 base_str(1:nchar(base_str))//
     -                 time_str(1:nchar(time_str))//
     -                 bchan_str(1:nchar(bchan_str))// 
     -                 echan_str(1:nchar(echan_str))// 
     -                 opcod_str(1:nchar(opcod_str))// 
     -                 reason_str(1:nchar(reason_str))//' /'

                    write(11,*)flag_str(1:nchar(flag_str))
                    nflags = nflags + 1
            endif
602         continue
         enddo     ! End of i-loop (antennas)
601      continue
      enddo        ! End of scans
      !---------------------------------------------

      write(*,1144)
      write(*,*)"Total # bad vis.: ",isamp
      write(*,*)" # Flags written: ",nflags
      write(31,1144)
      write(31,*)"Total # bad vis.: ",isamp
      write(31,*)" # Flags written: ",nflags

      close(31)







1140  FORMAT(I7,I3,A1,I2,A1,I2,A1,I2,I3,A1,I2,F9.2,F9.2,F9.2,F8.3,I4,I3)
1141  FORMAT(I7,I3,'/',I2,':',I2,':',I2,I3,'-',I2,F9.2,F9.2,F9.2,F8.3,I4
     -,I3)
1142  FORMAT(I2,'|',30(I3,1X))
1143  FORMAT(3X,30(I2,2X))
1144  FORMAT(3X,30('----'))
1145  FORMAT(I1,A1,I2,A1,I2,A1,I2,A1,I1,A1,I2,A1,I2,A1,I2)
1150  FORMAT('Scan #: ',I2,', TIMERAN: ',I3,':',I2,':',I2,':',I2,' --',
     -       I3,':',I2,':',I2,':',I2)
1151  FORMAT('!Scan #: ',I2,', TIMERAN: ',I3,':',I2,':',I2,':',I2,' --',
     -       I3,':',I2,':',I2,':',I2)
      end

      !include '/usr/lib/subroutine_lib/nchar.f'
      Integer function NCHAR(string)
C
C  Routine to count the number of characters in the
C  input string. Looks for the last occurrence of 
C  non-(null, blank or tab character)
C
C
C
      Implicit none
      integer*4 i,ipos
      character*(*)  string
      character blank,tab,null,c

C      data blank,tab,null/' ',9,0/
      blank=' '
        tab=char(9)
        null=char(0)

      ipos = 0
      i      = len(string)
      do while (i.gt.0.and.ipos.eq.0)
         c = string(i:i)
         if (c.ne.blank.and.c.ne.tab.and.c.ne.null) ipos = i
         i = i - 1
      end do

      NCHAR = ipos
      return
      end
