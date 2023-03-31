#!/bin/bash
#This file run nsight supported that start a single process. JOB_ID = 0, 1, 2. 
#Please refer to src/tester_nsight.py about the model behind each JOB_ID.
#Controlled version, JOB 0 gives way to JOB 1 or 2.
cleanup() {
	# sudo killall python
	pkill python
}
trap cleanup SIGINT SIGTERM SIGKILL

[[ -z $NUM_APPS ]] 		&& NUM_APPS=1
[[ -z $ALLOCS ]] 		&& ALLOCS=99
[[ -z $EXP_LENGTH ]] 	&& EXP_LENGTH=120
[[ -z $JOB_ID ]] 	&& JOB_ID=1

IFS=: read -ra ALLOCS_ARR <<<"$ALLOCS"

# pkill python 2>/dev/null

# start resource manager in background
#source /home/cc/miniconda3/etc/profile.d/conda.sh
#conda activate torch

CUR_CGROUP_DIR="../alnair${i}"
[[ -d $CUR_CGROUP_DIR ]] ||  mkdir $CUR_CGROUP_DIR # sudo has been deleted: sudo mkdir $CUR_CGROUP_DIR
CUR_ALLOC=${ALLOCS_ARR[0]}
LD_PRELOAD="../intercept-lib/build/lib/libcuinterpose.so"\
	ALNAIR_VGPU_COMPUTE_PERCENTILE=$CUR_ALLOC\
	CGROUP_DIR=$CUR_CGROUP_DIR ID=$JOB_ID\
	UTIL_LOG_PATH="sched_tester2_sm_util.log"\
	python -u src/tester_nsight.py $JOB_ID $DEVICE_ID --control > out_$JOB_ID.txt & 
	#echo "$!" |  tee $CUR_CGROUP_DIR/tasks # sudo has been deleted: sudo tee $CUR_CGROUP_DIR/tasks

sleep $EXP_LENGTH
cleanup
