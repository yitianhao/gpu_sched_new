config_file=exp_configs/input_controlsync.json
sync_level=kernel
# tmp_config_file=/tmp/input_controlsync.json
# exp_name=verify_queue_delay/ssd300_vgg16_1440_batch_4_vs_ssd300_vgg16_fpn

model_B_names=(fasterrcnn_resnet50_fpn) # fcos_resnet50_fpn)
model_B_weights=(FasterRCNN_ResNet50_FPN_Weights)
# FCOS_ResNet50_FPN_Weights
#                  FCN_ResNet50_Weights RetinaNet_ResNet50_FPN
#                  SSD300_VGG16_Weights)

# model_A_name=maskrcnn_resnet50_fpn_v2
# model_A_weight=MaskRCNN_ResNet50_FPN_V2_Weights
# model_A_name=alexnet
# model_B_names=(bloom)
# model_B_weights=(bloom-560m)
# model_B_names=(codegen)
# model_B_weights=(codegen-350M-mono)
# model_B_names=(retinanet_resnet50_fpn)
# model_B_weights=(RetinaNet_ResNet50_FPN_Weights)
# model_B_names=(fcos_resnet50_fpn)
# model_B_weights=(FCOS_ResNet50_FPN_Weights)
model_B_batch_size=1
model_B_input_size=720
model_B_sync=21
model_B_python_path=/dataheart/yhao/gpu-sched/CodeGen/.venv/bin/python
model_B_repo_path=/dataheart/yhao/gpu-sched/CodeGen

# model_A_name=deeplabv3_mobilenet_v3_large
# model_A_weight=DeepLabV3_MobileNet_V3_Large_Weights
# model_A_name=codegen
# model_A_weight=codegen-350M-mono
# model_A_batch_size=24
model_A_name=fasterrcnn_resnet50_fpn
model_A_weight=FasterRCNN_ResNet50_FPN_Weights
# model_A_name=fcn_resnet50
# model_A_weight=FCN_ResNet50_Weights
# model_A_name=bloom
# model_A_weight=bloom-560m
# model_A_name=retinanet_resnet50_fpn
# model_A_weight=RetinaNet_ResNet50_FPN_Weights
# model_A_name=maskrcnn_resnet50_fpn_v2
# model_A_weight=MaskRCNN_ResNet50_FPN_V2_Weights
# model_A_name=ssd300_vgg16
# model_A_weight=SSD300_VGG16_Weights
# model_A_name=deeplabv3_resnet50
# model_A_weight=DeepLabV3_ResNet50_Weights
model_A_batch_size=1
model_A_input_size=180
model_A_sync=20
model_A_python_path=/dataheart/yhao/gpu-sched/CodeGen/.venv/bin/python
model_A_repo_path=/dataheart/yhao/gpu-sched/CodeGen

for index in "${!model_B_names[@]}"; do
    echo $index ${model_B_names[$index]} ${model_B_weights[$index]}

    exp_name=${model_A_name}_batch_${model_A_batch_size}_${model_A_input_size}p_vs_${model_B_names[$index]}_batch_${model_B_batch_size}_${model_B_input_size}p_${sync_level}_sync

    out_path="../shepherd/${exp_name}"
    mkdir -p $out_path
    exp_config_file=${out_path}/exp_config.json
    jq .models[0].control.queue_limit.sync=$model_A_sync ${config_file}  | \
    jq .models[1].control.queue_limit.sync=$model_B_sync | \
    jq .models[1].control.controlsync=true | \
    jq .exp_dur=60  | \
    jq .device_id=0  | \
    jq --arg model_A_name "${model_A_name}" '.models[0].model_name=$model_A_name'  | \
    jq --arg model_A_weight "${model_A_weight}" '.models[0].model_weight=$model_A_weight'  | \
    jq --arg python_path "${model_A_python_path}" '.models[0].python_path=$python_path'  | \
    jq --arg repo_path "${model_A_repo_path}" '.models[0].repo_path=$repo_path'  | \
    jq '.models[0].resize_size=[180, 320]'  | \
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
done
