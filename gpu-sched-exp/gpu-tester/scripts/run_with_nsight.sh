#!/usr/bin/bash

CONFIG=exp_configs/input.json
SAVE_FOLDER=.
mkdir -p $SAVE_FOLDER

# create nsight_report.nsys-rep
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu \
    -o ${SAVE_FOLDER}/nsight_report -f true -e --cudabacktrace=true -x true \
    python run_exp.py -f ${CONFIG}

# convert nsight_report.nsys-rep into csv file
# csv header definition:
# /opt/nvidia/nsight-systems/2022.4.2/host-linux-x64/reports/kernelexectrace.py: Line 22
nsys stats -r kernexectrace --format csv -o ${SAVE_FOLDER}/nsight_report \
    ${SAVE_FOLDER}/nsight_report.nsys-rep
