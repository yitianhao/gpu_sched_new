import copy
import os
# import random
from itertools import product
from generate_profile_config import MODEL_NAMES, MODEL_WEIGHTS, TEMPLATE
from utils import write_json_file, get_configs_name

# random.seed(42)
model_A_batch_sizes = [1, 8]
model_B_batch_sizes = [1, 4]
sleep_time = [2]
model_names = MODEL_NAMES['detection'][:-1] + MODEL_NAMES['nlp']
model_weights = MODEL_WEIGHTS['detection'][:-1] + MODEL_WEIGHTS['nlp']
syncs = [1, 50, 500]

indices = list(range(len(model_names)))

pairs = list(product(indices, model_A_batch_sizes, indices, model_B_batch_sizes, sleep_time))
# pairs = [pair for pair in pairs if pair[1] >= pair[3]]
# print(len(pairs))

for idx, pair in enumerate(pairs):
    for sync in syncs:
        model_configs = []
        model_A_idx, model_A_batch, model_B_idx, model_B_batch, model_B_sleep = pair
        config = copy.deepcopy(TEMPLATE)
        config["model_name"] = model_names[model_A_idx]
        config["model_weight"] = model_weights[model_A_idx]
        config['control']['control'] = True
        config['control']['controlsync'] = True
        config['batch_size'] = model_A_batch
        config['control']['queue_limit']['sync'] = sync
        config['output_file_name'] = 'model_A'
        if 'codegen' in config['model_name']:
            config['repo_path'] = '/dataheart/zxxia/CodeGen'
            config['python_path'] = '/dataheart/zxxia/CodeGen/.venv/bin/python'
        else:
            w, h = 720, 1280
            config["resize_size"] = [w, h]
            config["resize"] = False

        model_configs.append(config)

        config = copy.deepcopy(TEMPLATE)
        config["model_name"] = model_names[model_B_idx]
        config["model_weight"] = model_weights[model_B_idx]
        config['control']['control'] = True
        config['control']['controlsync'] = False
        config['batch_size'] = model_B_batch
        config['sleep_time'] = model_B_sleep
        config['priority'] = 1
        config['output_file_name'] = 'model_B'
        if 'codegen' in config['model_name']:
            config['repo_path'] = '/dataheart/zxxia/CodeGen'
            config['python_path'] = '/dataheart/zxxia/CodeGen/.venv/bin/python'
        else:
            w, h = 720, 1280
            config["resize_size"] = [w, h]
            config["resize"] = False

        model_configs.append(config)

        exp_config = {'models': model_configs, 'exp_dur': 90, 'device_id': idx % 3}
        print(idx, idx % 3)
        name = get_configs_name(exp_config)
        exp_config['models'][0]['output_file_path'] = f'../results/datamirror/controlsync/model_pairs/{name}/sync_{sync}'
        exp_config['models'][1]['output_file_path'] = f'../results/datamirror/controlsync/model_pairs/{name}/sync_{sync}'
        print(name)

        save_dir = os.path.join("../model_pair_configs", f"device_{idx % 3}", f'{name}', f'sync_{sync}')
        os.makedirs(save_dir, exist_ok=True)
        write_json_file(os.path.join(save_dir, f'{name}.json'), exp_config)
