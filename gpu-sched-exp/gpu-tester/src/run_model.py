import argparse
import csv
import json
import os
import signal
import sys
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns, sleep
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from utils import read_json_file

# Wenqing: Import an ad-hoc iFPC injected c++ set mem
from ctypes import cdll
lib = cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so"))


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
    def __init__(self, control, config, resize=False, resize_size=(1440, 2560)) -> None:
        self.control = control
        self.priority = config['priority']
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
        csv_filename = os.path.join(
            self.config['output_file_path'],
            self.config['output_file_name'] + ".csv")
        self.csv_fh = open(csv_filename, 'w')
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')
        self.csv_writer.writerow(['timestamp_ns', 'jct_ms'])

    def __del__(self):
        self.csv_fh.flush()
        self.csv_fh.close()

    def infer(self):
        st: int = perf_counter_ns()
        # print(f"starting inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        res = None # res :torch.Tensor
        if self.control and self.priority > 0:
            try:
                lib.setMem(1)
            except Exception as e: print(e)
            print(f"{int(clock_gettime_ns(CLOCK_REALTIME) / 1000)} kernelGroupStart\n")
            res = self.model(self.img)
            print(f"{int(clock_gettime_ns(CLOCK_REALTIME) / 1000)} kernelGroupEnd\n")
            try:

                lib.setMem(0)
            except Exception as e: print(e)
            # read and print shared memory's current value
            # lib.printCurr()
        else:
            res = self.model(self.img)
        torch.cuda.synchronize()
        self.csv_writer.writerow(
            [clock_gettime_ns(CLOCK_REALTIME),
             (perf_counter_ns() - st) / 1000000])
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
    signal.signal(signal.SIGTERM, signal_handler)
    # Get input model configuration file path
    parser = argparse.ArgumentParser(description="Run a model's inference job")
    parser.add_argument('filename', help="Specifies the path to the model JSON file")
    parser.add_argument('deviceid', help="Specifies the gpu to run")
    args = parser.parse_args()
    filename = args.filename
    device_id = args.deviceid

    print("Device", int(device_id))
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")
    # set the cuda device to use
    torch.cuda.set_device(int(device_id))
    try:
        print(f"run_model.py: parsing file: {filename}")
        torch.cuda.set_device(int(sys.argv[2]))
        data = read_json_file(filename)
        print(f"Model: {data['model_name']}")
        print("Fields:")
        print(json.dumps(data, indent=4))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print(f"Invalid input file.", file=sys.stderr)
        sys.exit(1)

    # tester: SchedulerTester = SchedulerTester(0)
    tester: SchedulerTester = SchedulerTester(data['control']['control'], data)
    sleep(1)
    tester.run()


if __name__ == "__main__":
    main()
