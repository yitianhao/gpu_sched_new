#!/bin/bash
#Run JOB 1 and JOB0 with controller
cd ../gpu-tester #Assume you are runing nsight profiling from docker-k8s folder
EXP_LENGTH=30 DEVICE_ID=0 JOB_ID=0 bash scripts/run_with_gpu_tester_nsight_control.sh &
EXP_LENGTH=30 DEVICE_ID=0 JOB_ID=1 bash scripts/run_with_gpu_tester_nsight_control.sh 

