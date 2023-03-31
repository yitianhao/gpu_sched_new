#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1

CGROUP_DIR1="/sys/fs/cgroup/cpu/alnair1"
CGROUP_DIR2="/sys/fs/cgroup/cpu/alnair2"
CGROUP_DIR3="/sys/fs/cgroup/cpu/alnair3"

cleanup() {
	sudo killall python
	sudo nvidia-smi -i 0 -c DEFAULT
}
trap cleanup SIGINT SIGTERM SIGKILL

killall python 2>/dev/null
sudo bash -c 'fuser -k 10001/tcp'; sudo bash -c 'fuser -k 10002/tcp'

# start resource manager in background
source /home/cc/miniconda3/etc/profile.d/conda.sh
conda activate torch
MAX_BW=200000 MI=100000 python -u api/concierge.py &

if [[  $(ps aux | grep "nvidia-cuda-mps-control -d" | wc -l) -eq 2 ]]; then
	echo quit | nvidia-cuda-mps-control
fi

rm -f /tmp/mps_log/server.log
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
ejport MPS_SERVER_PID=$(echo "get_server_list" | nvidia-cuda-mps-control)
# export LD_PRELOAD=/home/cc/intercept-lib/build/lib/libcuinterpose.so

sleep 1
CUDA_MPS_PIPE_DIRECTORY=/tmp/mps\
	CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1\
	CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70\
	ALNAIR_VGPU_COMPUTE_PERCENTILE=99 CGROUP_DIR=$CGROUP_DIR1\
	HOSTNAME="SCHED_TESTER1" CTRL_PRT=5001 APP_IDX=1\
	python -u app/gpu_tester/scheduler_tester.py &
PYTHON_PID1=$!

CUDA_MPS_PIPE_DIRECTORY=/tmp/mps\
	CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1\
	CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30\
	ALNAIR_VGPU_COMPUTE_PERCENTILE=99 CGROUP_DIR=$CGROUP_DIR2\
	HOSTNAME="SCHED_TESTER2" CTRL_PRT=5002 APP_IDX=2\
	python -u app/gpu_tester/scheduler_tester.py &
PYTHON_PID2=$!

# CUDA_MPS_PIPE_DIRECTORY=/tmp/mps\
# 	CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1\
# 	CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33\
# 	python -u app/gpu_tester/scheduler_tester.py &

sleep 120 
kill $PYTHON_PID1 $PYTHON_PID2
echo quit | nvidia-cuda-mps-control
cleanup
