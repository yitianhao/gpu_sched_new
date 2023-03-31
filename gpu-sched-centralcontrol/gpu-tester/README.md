## To Run the 2x Experiment 
1. Compile and run `expcontorller` once under `gpu-sched/pytcppexp/` to delete stale boost files and create new SharedMemory, Mutex, and conditional variable files.\
`./make.sh`\
`./expcontorller`\
It does not matter is the expcontorller is running after it has been exectuted once before the experiment.\
2. Make sure the three boost files are fresh.\ 
`ls -lt /dev/shm/named_cnd3`\
`ls -lt /dev/shm/MySharedMemory3`\
`ls -lt /dev/shm/sem.named_mutex3`\
3. Configure control or no control under `gpu-sched/intercept-lib/`\
With controller: `CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK`\
Without controller: `CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR`\
Compile: `./make.sh`\
4. Under `gpu-sched/gpu-tester/`, run the experiment with FastRCNN and DeepLab sharing the GPU.
`./run_test.sh`\
## To Run the 2x Experiment with NSight supports
1. Compile and run `expcontorller` once under `gpu-sched/pytcppexp/` to delete stale boost files and create new SharedMemory, Mutex, and conditional variable files.\
`./make.sh`\
`./expcontorller`\
It does not matter is the expcontorller is running after it has been exectuted once before the experiment.\
2. Make sure the three boost files are fresh.\ 
`ls -lt /dev/shm/named_cnd3`\
`ls -lt /dev/shm/MySharedMemory3`\
`ls -lt /dev/shm/sem.named_mutex3`\
3. Configure control or no control under `gpu-sched/intercept-lib/`\
With controller: `CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR -D_SCHEDULER_LOCK`\
Without controller: `CXXFLAGS += -fPIC -O3 -D_RECORD_UTIL -D_DYN_ADJUST_FR`\
Compile: `./make.sh`\
4. Under `gpu-sched/gpu-tester/`, run the experiment with FastRCNN and DeepLab sharing the GPU.
`./run_test_nsight.sh`\
Run the experiment with FastRCNN and DeepLab sharing the GPU under scheduler.\
`./run_test_nsight_control.sh`\
Run the experiment with FastRCNN exclusivly on GPU. \
`./run_test_nsight_only1.sh`\
Run the experiment with DeepLab exclusivly on GPU. \
`./run_test_nsight_only0.sh`\
5.On my local machine, under `~/Documents/code/remote/` run `./JCT_ziyi_cp.sh` to get experiments JCT results.
