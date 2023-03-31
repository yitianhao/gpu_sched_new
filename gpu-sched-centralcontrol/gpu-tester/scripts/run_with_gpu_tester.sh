#!/bin/bash
cleanup() {
	# sudo killall python
	killall python
}
trap cleanup SIGINT SIGTERM SIGKILL

[[ -z $NUM_APPS ]] 		&& NUM_APPS=1
[[ -z $ALLOCS ]] 		&& ALLOCS=99
[[ -z $EXP_LENGTH ]] 	&& EXP_LENGTH=120

IFS=: read -ra ALLOCS_ARR <<<"$ALLOCS"

killall python 2>/dev/null

# start resource manager in background
#source /home/cc/miniconda3/etc/profile.d/conda.sh
#conda activate torch
for i in $(seq 1 $NUM_APPS); do
	# CUR_CGROUP_DIR="/sys/fs/cgroup/cpu/alnair${i}"
	CUR_CGROUP_DIR="../alnair${i}"
	[[ -d $CUR_CGROUP_DIR ]] ||  mkdir $CUR_CGROUP_DIR # sudo has been deleted: sudo mkdir $CUR_CGROUP_DIR
	CUR_ALLOC=${ALLOCS_ARR[$((i - 1))]}
	LD_PRELOAD="../intercept-lib/build/lib/libcuinterpose.so"\
		ALNAIR_VGPU_COMPUTE_PERCENTILE=$CUR_ALLOC\
		CGROUP_DIR=$CUR_CGROUP_DIR ID=$((i-1))\
		UTIL_LOG_PATH="sched_tester${i}_sm_util.log"\
		python -u src/tester.py $((i-1)) $DEVICE_ID &
	# python -u src/tester.py $((i-1)) &
	echo "$!" #|  tee $CUR_CGROUP_DIR/tasks # sudo has been deleted: sudo tee $CUR_CGROUP_DIR/tasks
done

sleep $EXP_LENGTH
cleanup
