import os
import argparse
import json
import collections
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns, sleep
import random
from typing import List
import time
import datetime;
import sys
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F

#Wenqing: Import an ad-hoc iFPC injected c++ set mem 
from ctypes import cdll
# lib = cdll.LoadLibrary('./libgeek.so')i
lib =cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so")) 
 

def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)

class SomeModel():
    def __init__(self, model_size) -> None:
        self.layers = []
        self.model_size = model_size
        self.layer = torch.rand((1500, 1500)).cuda()
        # for i in range(50):
            # self.layers.append(torch.rand((2000, 2000)).cuda())
        self.input = torch.rand((20, 1500)).cuda()
    
    def evaluate(self):
        res = self.input
        for j in range(self.model_size):
            res = torch.matmul(res, self.layer)
        return res
        
        

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

    def infer(self):
        st: int = perf_counter_ns()
        # print(f"starting inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        res: torch.Tensor = None
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
        # print(f"{res}")
        print(f"JCT {clock_gettime_ns(CLOCK_REALTIME)}"
                     f" {self.priority} {(perf_counter_ns() - st) / 1000000}\n")
        return res

    def run(self):
        # t_end = time.time() + 15
        # while time.time() < t_end:
        sleep_dur = self.config['sleep_time']
        while True:
            torch.cuda.nvtx.range_push("regionTest")
            self.infer()
            sys.stdout.flush()
            torch.cuda.nvtx.range_pop()
            sleep(sleep_dur)


def main():  
    # print("Process", int(sys.argv[1]))
    # print("Device", int(sys.argv[2]))
    # # read control switch
    # doControl = False
    # if len(sys.argv) > 3:
    #     print("Control?", sys.argv[3])
    #     doControl = True if sys.argv[3] == "--control" else False
    


    #Get input model configuration file path
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
        with open(filename, 'r') as file_input:
            data = json.load(file_input, object_pairs_hook=collections.OrderedDict)
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
