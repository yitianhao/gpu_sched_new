import argparse
import csv
import json
import os
import select
import signal
import sys
from time import perf_counter_ns, sleep
import torch
from utils import read_json_file
from vision_model import VisionModel
from transformer_model import TransformerModel
from utils import print_time

# Wenqing: Import an ad-hoc iFPC injected c++ set mem
from ctypes import cdll


RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    RUNNING = False

def debug_print(msg):
    print(msg, file=sys.stderr, flush=True)


class SchedulerTester():
    def __init__(self, control, config, device_id, sync_model_load) -> None:
        self.config = config
        self.device_id = device_id
        self.control = control
        self.priority = config['priority']
        if self.control and self.priority > 0:
            # only load library when needed
            self.lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))
        else:
            self.lib = None

        try:
            self.model = VisionModel(self.config, self.device_id)
        except (ValueError, KeyError):
            self.model = None
        if self.model is None:
            try:
                self.model = TransformerModel(self.config, self.device_id)
            except:
                raise ValueError("Unsupported Model")

        os.makedirs(self.config['output_file_path'], exist_ok=True)
        csv_filename = os.path.join(
            self.config['output_file_path'],
            self.config['output_file_name'] + ".csv")
        self.csv_fh = open(csv_filename, 'w', 1)
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')

        # https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        self.csv_writer.writerow(
            ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
             'max_allocated_gpu_memory_allocated_byte',
             'max_reserved_gpu_memory_byte'])
        with print_time("dummy run", file=sys.stderr):
            self.infer() # dummy run to warm up the gpu
        if sync_model_load:
            print("model loaded", file=sys.stdout, flush=True)
            poll_result = select.select([sys.stdin], [], [], 120)[0]
            if poll_result:
                msg = sys.stdin.readline().rstrip()
                debug_print(msg)
            else:
                # continue to run anyway
                return

    def __del__(self):
        self.csv_fh.flush()
        self.csv_fh.close()

    def infer(self):
        start_t: int = perf_counter_ns()
        res = None # res :torch.Tensor
        if self.lib is not None:
            try:
                suffix = os.getenv("SUFFIX", None)
                assert suffix is not None
                self.lib.setMem(1, suffix.encode())
                self.lib.waitForEmptyGPU()
            except Exception as e:
                print(e)
        assert self.model is not None
        res = self.model()
        torch.cuda.synchronize()
        end_t: int = perf_counter_ns()
        if self.lib is not None:
            try:
                suffix = os.getenv("SUFFIX", None)
                assert suffix is not None
                self.lib.setMem(0, suffix.encode())
            except Exception as e:
                debug_print(e)
        max_alloc_mem_byte = torch.cuda.max_memory_allocated(self.device_id)
        max_rsrv_mem_byte = torch.cuda.max_memory_reserved(self.device_id)
        self.csv_writer.writerow([
            start_t, end_t, (end_t - start_t) / 1000000, max_alloc_mem_byte,
            max_rsrv_mem_byte])
        self.csv_fh.flush()
        return res

    def run(self):
        sleep_dur = self.config['sleep_time']
        while RUNNING:
            torch.cuda.nvtx.range_push("regionTest")
            self.infer()
            sys.stdout.flush()
            torch.cuda.nvtx.range_pop()
            sleep(sleep_dur)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    # Get input model configuration file path
    parser = argparse.ArgumentParser(description="Run a model's inference job")
    parser.add_argument('filename', type=str,
                        help="Specifies the path to the model JSON file")
    parser.add_argument('deviceid', type=int, help="Specifies the gpu to run")
    parser.add_argument('--sync-model-load', action="store_true",
                        help="Load model and wait until run message is "
                        "received to start execution. Default: do not wait.")
    args = parser.parse_args()
    filename = args.filename
    device_id = args.deviceid

    debug_print(f"proc {os.getpid()}, set Device {device_id}")
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")
    # set the cuda device to use
    torch.cuda.set_device(device_id)
    try:
        data = read_json_file(filename)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        debug_print(f"Invalid input file.")
        sys.exit(1)

    tester: SchedulerTester = SchedulerTester(
        data['control']['control'], data, device_id, args.sync_model_load)
    sleep(1)
    tester.run()


if __name__ == "__main__":
    main()
