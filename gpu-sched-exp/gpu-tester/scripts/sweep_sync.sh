config_file=exp_configs/input_controlsync.json
# tmp_config_file=/tmp/input_controlsync.json
# exp_name=verify_queue_delay/ssd300_vgg16_1440_batch_4_vs_ssd300_vgg16_fpn

model_B_names=(fasterrcnn_resnet50_fpn) # fcos_resnet50_fpn)
# fcn_resnet50
#                retinanet_resnet50_fpn ssd300_vgg16)
model_B_weights=(FasterRCNN_ResNet50_FPN_Weights)
# FCOS_ResNet50_FPN_Weights
#                  FCN_ResNet50_Weights RetinaNet_ResNet50_FPN
#                  SSD300_VGG16_Weights)

# model_A_name=maskrcnn_resnet50_fpn_v2
# model_A_weight=MaskRCNN_ResNet50_FPN_V2_Weights
# model_A_name=alexnet
model_A_name=codegen
model_A_weight=codegen-350M-mono
model_A_batch_size=8
python_path=/mnt/data/zxxia/CodeGen/.venv/bin/python
repo_path=/mnt/data/zxxia/CodeGen
python_path=/dataheart/zxxia/CodeGen/.venv/bin/python
repo_path=/dataheart/zxxia/CodeGen
sleep_time=2
model_B_batch_size=1

for index in "${!model_B_names[@]}"; do
    echo $index ${model_B_names[$index]} ${model_B_weights[$index]}

    exp_name=verify_queue_delay/${model_A_name}_batch_${model_A_batch_size}_vs_${model_B_names[$index]}_debug

    for sync in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
        out_path="../results/${exp_name}/sync_${sync}"
        mkdir -p $out_path
        exp_config_file=${out_path}/exp_config.json
        # jq .models[0].control.queue_limit.event_group=$sync ${config_file}  | \
        jq .models[0].control.queue_limit.sync=$sync ${config_file}  | \
        jq .exp_dur=90  | \
        jq .device_id=1  | \
        jq --arg model_A_name "${model_A_name}" '.models[0].model_name=$model_A_name'  | \
        jq --arg model_A_weight "${model_A_weight}" '.models[0].model_weight=$model_A_weight'  | \
        jq --arg python_path "${python_path}" '.models[0].python_path=$python_path'  | \
        jq --arg repo_path "${repo_path}" '.models[0].repo_path=$repo_path'  | \
        jq '.models[0].resize_size=[720, 1280]'  | \
        jq .models[0].batch_size=$model_A_batch_size  | \
        jq .models[0].sleep_time=0 | \
        jq --arg model_B_name "${model_B_names[$index]}" '.models[1].model_name=$model_B_name'  | \
        jq --arg model_B_weight "${model_B_weights[$index]}" '.models[1].model_weight=$model_B_weight'  | \
        jq --arg python_path "${python_path}" '.models[1].python_path=$python_path'  | \
        jq --arg repo_path "${repo_path}" '.models[1].repo_path=$repo_path'  | \
        jq .models[1].sleep_time=${sleep_time} | \
        jq .models[1].batch_size=${model_B_batch_size} | \
        jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
        jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${exp_config_file}

        # nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used \
        #     --format=csv -l 1 -f ${out_path}/smi_report.csv & pid=$!
        nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s process-tree \
            -o ${out_path}/nsight_report -f true -e --cudabacktrace=true -x true \
        python src/run_exp.py -f $exp_config_file
        # kill $pid
        nsys stats -r kernexectrace,nvtxpptrace --format csv --force-export true \
            --force-overwrite true -o ${out_path}/nsight_report \
            ${out_path}/nsight_report.nsys-rep
        python src/plot_nsys_report.py \
            -f ${out_path}/nsight_report_kernexectrace.csv \
            -p ${out_path}/models_pid.json \
            -o ${out_path}
    done
done
done
