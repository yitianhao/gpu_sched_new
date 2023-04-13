config_file=exp_configs/input_controlsync.json
tmp_config_file=/tmp/input_controlsync.json
for sync in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
    # for round in $(seq 0 1 1); do
    round=0
        # echo $sync $round
    # out_path="..\/results\/larger_size_sync_${sync}\/round_${round}"
    # out_path="..\/results\/fcn_new\/sync_${sync}\/round_${round}"
    # sed -e "s/\"sync\": 1/\"sync\": ${sync}/g" \
    #     -e "s/\"output_file_path\": \".\/logs\"/\"output_file_path\": \"$out_path\"/g" \
    #     exp_configs/input_controlevent.json > /tmp/input_controlevent.json
    out_path="../results/controlsync/sync_${sync}/round_${round}"
    jq .models[0].control.queue_limit.sync=$sync ${config_file}  | \
    jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
    jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${tmp_config_file}
    python run_exp.py -f /tmp/input_controlevent.json
        # mv ../results/fcn_new/fcn_new_sync_${sync} ../results/fcn_new/sync_$sync
    # done
done

config_file=exp_configs/input_controlevent.json
tmp_config_file=/tmp/input_controlevent.json
for event_group in 1 2 4 8 10 20 50 100 500 1000 5000 10000 1000000; do
    round=0
    out_path="../results/fcn_controlevent/event_group_${event_group}/round_${round}"
    jq .models[0].control.queue_limit.event_group=$event_group ${config_file}  | \
    jq --arg out_path "${out_path}" '.models[0].output_file_path=$out_path' | \
    jq --arg out_path "${out_path}" '.models[1].output_file_path=$out_path' > ${tmp_config_file}
    python run_exp.py -f /tmp/input_controlevent.json
done
