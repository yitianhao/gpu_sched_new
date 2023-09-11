config_file=exp_configs/input_controlsync.json
sync_level=""
# sync_layer=layer # uncomment this to simulate PipeSwitch

model_A_name=fasterrcnn_resnet50_fpn
model_A_weight=FasterRCNN_ResNet50_FPN_Weights
model_A_batch_size=1

model_B_name=fasterrcnn_resnet50_fpn
model_B_weight=FasterRCNN_ResNet50_FPN_Weights
model_B_batch_size=1
sleep_time=1

exp_name=sweep_sync_example
for sync in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
    out_path="../results/${exp_name}/sync_${sync}"
    mkdir -p $out_path
    exp_config_file=${out_path}/exp_config.json
    jq .models[0].control.queue_limit.sync=$sync ${config_file}  | \
    jq .exp_dur=90  | \
    jq .device_id=0  | \
    jq --arg model_A_name "${model_A_name}" '.models[0].model_name=$model_A_name'  | \
    jq --arg model_A_weight "${model_A_weight}" '.models[0].model_weight=$model_A_weight'  | \
    jq '.models[0].resize_size=[1440, 2560]'  | \
    jq .models[0].batch_size=$model_A_batch_size  | \
    jq --arg sync_level "${sync_level}" '.models[0].sync_level=$sync_level'  | \
    jq --arg model_B_name "${model_B_name}" '.models[1].model_name=$model_B_name'  | \
    jq --arg model_B_weight "${model_B_weight}" '.models[1].model_weight=$model_B_weight'  | \
    jq .models[1].sleep_time=${sleep_time} | \
    jq .models[1].batch_size=${model_B_batch_size} | \
    jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
    jq '.models[1].resize_size=[720, 1280]'  | \
    jq .models[1].batch_size=$model_B_batch_size  | \
    jq --arg sync_level "${sync_level}" '.models[1].sync_level=$sync_level'  | \
    jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${exp_config_file}
    # jq --arg python_path "${model_B_python_path}" '.models[1].python_path=$python_path'  | \
    # jq --arg repo_path "${model_B_repo_path}" '.models[1].repo_path=$repo_path'  | \
    # jq --arg repo_path "${model_A_repo_path}" '.models[0].repo_path=$repo_path'  | \
    # jq --arg python_path "${model_A_python_path}" '.models[0].python_path=$python_path'  | \
    # jq --arg python_path "${python_path}" '.models[1].python_path=$python_path'  | \
    # jq --arg repo_path "${repo_path}" '.models[1].repo_path=$repo_path'  | \

    python src/run_exp.py -f $exp_config_file
done
