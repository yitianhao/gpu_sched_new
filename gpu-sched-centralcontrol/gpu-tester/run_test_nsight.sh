#!/bin/bash
#Run JOB 1 and JOB 0 sharing GPU without controller. Job 1: 1 infer/sec Job 0: continuously run
cd ../gpu-tester/
EXP_LENGTH=15 DEVICE_ID=0 JOB_ID=0 bash scripts/run_with_gpu_tester_nsight.sh &
EXP_LENGTH=15 DEVICE_ID=0 JOB_ID=1 bash scripts/run_with_gpu_tester_nsight.sh 
