#!/bin/bash
#--------------------------------------------------------
#
# Shell script to process all the channel-files output 
# by AIPS task UVFND and write out AIPS readable FLAG 
# files. 
#                           -- wr, 23 Jul, 2011
#--------------------------------------------------------

# Set some variables here (modify suitably):
begchan=11 
endchan=246
srcname=MY_3C468.1
stokes_tag=I
parfile=my_uvflg_$srcname.par

bindir=/home/wasim/Desktop/CURR_DEVEL/FLAG_COMMAND_GENERATION/FLGCMD_GEN/EXEC/
executable=my_uvflg
for (( ichan=$begchan;ichan<=$endchan;ichan++ ));
do
	$bindir$executable $srcname $stokes_tag $ichan $parfile
done
