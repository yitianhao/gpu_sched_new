config_file=exp_configs/input_controlsync.json
sync_level=layer
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
# model_B_names=(codegen)
# model_B_weights=(codegen-350M-mono)
model_B_batch_size=1
model_B_python_path=/dataheart/yhao/gpu-sched/CodeGen/.venv/bin/python
model_B_repo_path=/dataheart/yhao/gpu-sched/CodeGen

# model_A_name=deeplabv3_mobilenet_v3_large
# model_A_weight=DeepLabV3_MobileNet_V3_Large_Weights
# model_A_name=codegen
# model_A_weight=codegen-350M-mono
# model_A_batch_size=24
# model_A_name=fasterrcnn_resnet50_fpn
# model_A_weight=FasterRCNN_ResNet50_FPN_Weights
# model_A_name=fcn_resnet50
# model_A_weight=FCN_ResNet50_Weights
model_A_name=bloom
model_A_weight=bloom-560m
model_A_batch_size=16
model_A_python_path=/dataheart/yhao/gpu-sched/CodeGen/.venv/bin/python
model_A_repo_path=/dataheart/yhao/gpu-sched/CodeGen

for index in "${!model_B_names[@]}"; do
    echo $index ${model_B_names[$index]} ${model_B_weights[$index]}

    exp_name=${model_A_name}_batch_${model_A_batch_size}_vs_${model_B_names[$index]}_batch_${model_B_batch_size}_${sync_level}_sync

    for sync in 1000000 1 2 4 8 10 20 50 100 500 1000 5000 10000; do
        out_path="../results/${exp_name}/sync_${sync}"
        mkdir -p $out_path
        exp_config_file=${out_path}/exp_config.json
        jq .models[0].control.queue_limit.sync=$sync ${config_file}  | \
        jq .exp_dur=300  | \
        jq .device_id=1  | \
        jq --arg model_A_name "${model_A_name}" '.models[0].model_name=$model_A_name'  | \
        jq --arg model_A_weight "${model_A_weight}" '.models[0].model_weight=$model_A_weight'  | \
        jq --arg python_path "${model_A_python_path}" '.models[0].python_path=$python_path'  | \
        jq --arg repo_path "${model_A_repo_path}" '.models[0].repo_path=$repo_path'  | \
        jq '.models[0].resize_size=[1440, 2560]'  | \
        jq .models[0].batch_size=$model_A_batch_size  | \
        jq --arg sync_level "${sync_level}" '.models[0].sync_level=$sync_level'  | \
        jq --arg model_B_name "${model_B_names[$index]}" '.models[1].model_name=$model_B_name'  | \
        jq --arg model_B_weight "${model_B_weights[$index]}" '.models[1].model_weight=$model_B_weight'  | \
        jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
        jq --arg python_path "${model_B_python_path}" '.models[1].python_path=$python_path'  | \
        jq --arg repo_path "${model_B_repo_path}" '.models[1].repo_path=$repo_path'  | \
        jq '.models[1].resize_size=[720, 1280]'  | \
        jq .models[1].batch_size=$model_B_batch_size  | \
        jq --arg sync_level "${sync_level}" '.models[1].sync_level=$sync_level'  | \
        jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${exp_config_file}

        python src/run_exp.py -f $exp_config_file
        # nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used \
        #     --format=csv -l 1 -f ${out_path}/smi_report.csv & pid=$!
        # nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s process-tree \
        #     -o ${out_path}/nsight_report -f true -e --cudabacktrace=true -x true \
        # python src/run_exp.py -f $exp_config_file
        # # kill $pid
        # nsys stats -r kernexectrace,nvtxpptrace --format csv --force-export true \
        #     --force-overwrite true -o ${out_path}/nsight_report \
        #     ${out_path}/nsight_report.nsys-rep
        # python src/plot_nsys_report.py \
        #     -f ${out_path}/nsight_report_kernexectrace.csv \
        #     -p ${out_path}/models_pid.json \
        #     -o ${out_path}
    done
done




    # jq '.models[1].model_name="ssd300_vgg16"'  | \
    # jq '.models[1].model_weight="SSD300_VGG16_Weights"'  | \
# python src/plot_jct_scatter.py \
#     --log-dir \
#         ../results/$exp_name \
#     --model-A-profile \
#         ./profiles/detection/fasterrcnn_resnet50_fpn_1440x2560_sleep_time_0.csv \
#     --model-B-profile \
#         ./profiles/detection/fasterrcnn_resnet50_fpn_720x1280_sleep_time_1.csv

# config_file=exp_configs/input_controlevent.json
# tmp_config_file=/tmp/input_controlevent.json
# for event_group in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
#     round=0
#     out_path="../results/fcn_controlevent/event_group_${event_group}/round_${round}"
#     jq .models[0].control.queue_limit.event_group=$event_group ${config_file}  | \
#     jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
#     jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${tmp_config_file}
#     python src/run_exp.py -f /tmp/input_controlevent.json
# done


# config_file=exp_configs/input_controlsync.json
# tmp_config_file=/tmp/input_controlsync.json
# for sync in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
#     # for round in $(seq 0 1 1); do
#     round=0
#         # echo $sync $round
#     out_path="../results/controlsync_resnext101_resnet101/sync_${sync}/round_${round}"
#     jq .models[0].control.queue_limit.sync=$sync ${config_file}  | \
#     jq '.models[0].model_name="resnext101_32x8d"'  | \
#     jq '.models[0].model_weight="ResNeXt101_32X8D_Weights"'  | \
#     jq '.models[0].resize="true"'  | \
#     jq '.models[0].resize_size=[720, 1280]'  | \
#     jq '.models[1].model_name="resnet101"'  | \
#     jq '.models[1].model_weight="Resnet101_Weights"'  | \
#     jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
#     jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${tmp_config_file}
#     python src/run_exp.py -f $tmp_config_file
#     # jq '.models[1].model_name="resnet50"'  | \
#     # jq '.models[1].model_weight="ResNet50_Weights"'  | \

#     # jq '.models[0].model_name="resnet50"'  | \
#     # jq '.models[0].model_weight="ResNet50_Weights"'  | \
#     # jq '.models[0].resize="true"'  | \
#     # jq '.models[0].resize_size=[1440, 2560]'  | \
#     # jq '.models[1].model_name="resnet50"'  | \
#     # jq '.models[1].model_weight="ResNet50_Weights"'  | \
#         # mv ../results/fcn_new/fcn_new_sync_${sync} ../results/fcn_new/sync_$sync
#     # done
# done
