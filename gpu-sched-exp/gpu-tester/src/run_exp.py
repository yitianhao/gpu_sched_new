import argparse
import json
import subprocess
import signal
import sys
import os
import tempfile
import uuid
from ctypes import cdll
from time import sleep
from utils import get_config_name, read_json_file, write_json_file

def createModelTempFile(models):
    model_files = []
    for model in models:
        # Create temporary file
        model_file = tempfile.NamedTemporaryFile(mode='w+', prefix=f"{model['model_name']}_", suffix=".json")
        # Write model to temporary file
        json.dump(model, model_file, indent=4)
        model_file.flush()
        model_files.append(model_file)
    return model_files

def destroyModelTempFile(model_files):
    for model_file in model_files:
        model_file.close()

def main():
    # Get input file path
    parser = argparse.ArgumentParser(description="Core GPU Sharing Experiment")
    parser.add_argument('-f', '--file', metavar="FILEPATH", required=True,
                        help="Specifies the path to the experiment configuration file")
    args = parser.parse_args()
    filename = args.file

    # Parse Experiment Configuration JSON file
    try:
        experiment_config = read_json_file(filename)
    except FileNotFoundError:
        print(f"Input Experiment Config file: [{filename}] not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Input Experiment Config file: [{filename}] invalid.", file=sys.stderr)
        sys.exit(1)

    # Initialize share mem
    lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))
    suffix = uuid.uuid4().hex
    lib.create_shared_mem_and_locks(suffix.encode())

    #Run each model
    models = experiment_config.get('models', [])
    model_pids = []
    model_processes = []
    log_files = []
    model_files = createModelTempFile(models)
    for i, model_file in enumerate(model_files):
        # Run src/run_model.py with current model configuration
        model = models[i]
        # logging
        logfilename = os.path.join(
            model['output_file_path'], f"{model['output_file_name']}.log")
        os.makedirs(model['output_file_path'], exist_ok=True)
        logfile = open(logfilename, "w", 1)
        log_files.append(logfile)
        # hooks configuration
        process_env = os.environ.copy()
        control = model['control']['control']
        controlSync = model['control']['controlsync']
        controlEvent = model['control']['controlEvent']
        if (control):
            # process_env['ALNAIR_VGPU_COMPUTE_PERCENTILE'] = "99"
            # process_env['CGROUP_DIR'] = "../alnair"
            process_env['ID'] = str(model['priority'])
            process_env['SUFFIX'] = suffix
            # process_env['UTIL_LOG_PATH'] = "sched_tester2_sm_util.log"
            # process_env['LD_PRELOAD'] = os.path.abspath(os.path.join(
            #     "../intercept-lib/build/lib/libcuinterpose.so"))
        if (control and controlSync):
            process_env['LD_PRELOAD'] = os.path.abspath(
                "../intercept-lib/build/lib/libcuinterpose_sync.so")
            process_env['SYNC_KERNELS'] = str(model['control']['queue_limit']['sync'])
        if (control and controlEvent):
            process_env['LD_PRELOAD'] = os.path.abspath(
                "../intercept-lib/build/lib/libcuinterpose_event.so")
            process_env['EVENT_GROUP_SIZE'] = str(model['control']['queue_limit']['event_group'])

        # run each model as a process
        if 'codegen' in model['model_name']:
            cmd = [model['python_path'],
                   '-m', 'jaxformer.hf.sample',
                   '--model', model['model_weight'],
                   '--context', '"def hello_world():"',
                   '--batch-size', str(model['batch_size']),
                   '--max-length', '2',
                   '--device', 'cuda:' + str(experiment_config["device_id"]),
                   '--output_file_path',
                   os.path.abspath(model['output_file_path']),
                   '--output_file_name', model['output_file_name'], '--control', '--priority', str(model['priority'])]
            cwd = model['repo_path']
        else:
            cmd = ['python', 'src/run_model.py', model_file.name,
                   str(experiment_config.get('device_id'))]
            cwd = '.'
        model_process = subprocess.Popen(
            cmd, cwd=cwd, stdout=logfile,
            stderr=logfile, env=process_env)
        # record model's PID
        model_name = get_config_name(model)
        model_pids.append((model_name, model_process.pid))
        model_processes.append(model_process)

    #Stop each model at given experiment duration exhausted
    exp_duration = experiment_config.get('exp_dur')
    sleep(exp_duration)
    for i, p in enumerate(model_processes):
        # p.wait()
        logfile = log_files[i]
        logfile.flush()
        p.send_signal(signal.SIGKILL)
        logfile.close()
    destroyModelTempFile(model_files)

    # Write out.log
    write_json_file(
        os.path.join(model['output_file_path'], 'models_pid.json'), model_pids)
    print("PID summary log saved as [models_pid.json]")
    lib.remove_shared_mem_and_locks(suffix.encode())


if __name__ == '__main__':
    main()
