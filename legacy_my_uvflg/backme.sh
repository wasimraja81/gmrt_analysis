#!/bin/bash

# This script will keep tarred back-ups of all the MY_A2303/ subdirs 
# except the large DATA/  and WORK/ directories. 

# The tar will be kept in the directory called MY_A2303_BACKUP/ inside 
# the parent A2303_ANALYSIS/ directory.

# -- wasim, 19 OCT, 2009

export date_today=`date +_ason_%Y%b%d_%H-%M_hrs`
tarfile=flgcmd_gen_codes_backup_from_`hostname`$date_today
backup_dir=FLGCMD_GEN_BACKUP
code_tag=FLAG_COMMAND_GENERATION
mkdir $tarfile
#cp -ra CMD/ $tarfile/ 
cp -ra EXEC/ $tarfile/
cp -ra MAKE/ $tarfile/ 
cp -ra PAR/ $tarfile/ 
#cp -ra PLOTS/ $tarfile/ 
#cp -ra README/ $tarfile/ 
#cp -ra INSTALL/ $tarfile/ 
cp -ra SOURCE/ $tarfile/ 
cp -ra WORK/ $tarfile  
#cp -ra SCRIPTS/ $tarfile  
#cp -ra CONFIG/ $tarfile  
tar cvf $tarfile.tar $tarfile
rm -rf $tarfile
#create if required:
mkdir ../$backup_dir 
mv $tarfile.tar ../$backup_dir 
echo " "
echo " "
echo "------------------------------------ "
echo "Congratulations!!"
echo "You have successfully backed"
echo "up your $code_tag codes..."
echo "The back-up is contained "
echo "in the tar: "
echo " "
echo $tarfile".tar"
echo "kept in the directory ../$backup_dir "
echo "------------------------------------ "
