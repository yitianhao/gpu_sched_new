import argparse
import csv
import json
import os
import signal
import sys
from time import perf_counter_ns, sleep
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from utils import read_json_file

# Wenqing: Import an ad-hoc iFPC injected c++ set mem
from ctypes import cdll


RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    RUNNING = False


def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)


class SchedulerTester():
    def __init__(self, control, config, device_id, resize=False, resize_size=(1440, 2560)) -> None:
        self.device_id = device_id
        self.control = control
        self.priority = config['priority']
        if self.control and self.priority > 0:
            # only load library when needed
            self.lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))
        self.config = config
        self.resize = config['resize']
        resize_size_list = config['resize_size']
        self.resize_size = tuple(resize_size_list)
        #for miniconda3/envs/torch/lib/python3.9/site-packages/torchvision/models/detection/transform.py"
        if self.resize == False:
            os.environ['RESIZE'] = "false"
        else:
            os.environ['RESIZE'] = "true"
        model_name = self.config['model_name']
        model_weight = self.config['model_weight']
        if (getattr(torchvision.models.segmentation, model_weight, False)):
            # a model from torchvision.models.segmentation
            self.weights= getattr(torchvision.models.segmentation, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models.segmentation, model_name)(weights=self.weights).eval().cuda()
        elif(getattr(torchvision.models.detection, model_weight, False)):
            # a model from torchvision.models.detection
            self.weights= getattr(torchvision.models.detection, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models.detection, model_name)(weights=self.weights).eval().cuda()
        else:
            # a model from torchvision.models or terminated with an excepton
            self.weights= getattr(torchvision.models, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models, model_name)(weights=self.weights).eval().cuda()

        if self.resize == False:
            #No resize FasterRCNN_ResNet50 720
            self.model_preprocess = self.weights.transforms()
            img_path = self.config['input_file_path']
            self.img: torch.Tensor =\
                self.model_preprocess(read_img(img_path)).unsqueeze(0).cuda()
        else:
            self.model_preprocess = self.weights.transforms()
            img_path = self.config['input_file_path']
            i = read_img(img_path).unsqueeze(0)
            i = F.interpolate(i, self.resize_size)
            self.img: torch.Tensor = i.cuda()
        os.makedirs(self.config['output_file_path'], exist_ok=True)
        csv_filename = os.path.join(
            self.config['output_file_path'],
            self.config['output_file_name'] + ".csv")
        self.csv_fh = open(csv_filename, 'w')
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')

        # https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        self.csv_writer.writerow(
            ['start_timestamp_ns', 'end_timestamp_ns', 'jct_ms',
             'max_allocated_gpu_memory_allocated_byte',
             'max_reserved_gpu_memory_byte'])

    def __del__(self):
        self.csv_fh.flush()
        self.csv_fh.close()

    def infer(self):
        res = None # res :torch.Tensor
        if self.control and self.priority > 0:
            try:
                self.lib.setMem(1)
            except Exception as e:
                print(e)
        start_t: int = perf_counter_ns()
        res = self.model(self.img)
        torch.cuda.synchronize()
        end_t: int = perf_counter_ns()
        if self.control and self.priority > 0:
            try:
                self.lib.setMem(0)
            except Exception as e:
                print(e)
            # read and print shared memory's current value
            # lib.printCurr()
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
    args = parser.parse_args()
    filename = args.filename
    device_id = args.deviceid

    print("Device", device_id)
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")
    # set the cuda device to use
    torch.cuda.set_device(device_id)
    try:
        print(f"run_model.py: parsing file: {filename}")
        data = read_json_file(filename)
        print(f"Model: {data['model_name']}")
        print("Fields:")
        print(json.dumps(data, indent=4))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print(f"Invalid input file.", file=sys.stderr)
        sys.exit(1)

    tester: SchedulerTester = SchedulerTester(
        data['control']['control'], data, device_id)
    sleep(1)
    tester.run()


if __name__ == "__main__":
    main()
