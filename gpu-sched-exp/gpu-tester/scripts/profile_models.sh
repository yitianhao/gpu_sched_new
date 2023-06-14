#!/usr/bin/bash
# workload=detection
workload=segmentation
# workload=classification
profile_configs=$(ls ../profile_configs/$workload/*.json)
# profile_configs=$(ls profile_configs/$workload/ssd300_vgg16_1440x2560_sleep_time_0.json)
# profile_configs=$(ls profile_configs/$workload/fasterrcnn_resnet50_fpn_720x1280_sleep_time_1.json)
# profile_configs='./profile_configs/batch.json'
device=1

for profile_config in $profile_configs; do
    echo $profile_config
    # timeout -s SIGINT 30 python src/run_model.py $profile_config $device
    name=$(basename $profile_config)
    name="${name%.*}"

    save_folder=../profiles/$workload/${name}
    mkdir -p $save_folder
    # echo $save_folder
    # measure gpu utilization, etc
    nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used \
        --format=csv -l 1 -f ${save_folder}/smi_report.csv & pid=$!
    nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s process-tree \
        -o ${save_folder}/nsight_report -f true -e --cudabacktrace=true -x true \
    kill $pid
    nsys stats -r kernexectrace,nvtxpptrace --format csv --force-export true \
            --force-overwrite true -o ${save_folder}/nsight_report \
        ${save_folder}/nsight_report.nsys-rep
    python src/plot_nsys_report.py \
        -f ${save_folder}/nsight_report_kernexectrace.csv \
        -o ${save_folder}
    python src/run_model.py $profile_config $device 30
done
