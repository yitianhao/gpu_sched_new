import os
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
# from torch.profiler import profile, record_function, ProfilerActivity
# import torch.cuda.profiler as profiler
# import pyprof
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
    # added
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
    SSD300_VGG16_Weights,
    ssd300_vgg16,
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large
)
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from torchvision.models import (
    regnet_y_128gf,
    RegNet_Y_128GF_Weights,
    mnasnet0_5,
    MNASNet0_5_Weights,
    squeezenet1_1, 
    SqueezeNet1_1_Weights,
    vit_h_14,
    ViT_H_14_Weights,
    shufflenet_v2_x0_5,
    ShuffleNet_V2_X0_5_Weights,
    vit_l_32,
    ViT_L_32_Weights,
    densenet201,
    DenseNet201_Weights,
    densenet121,
    DenseNet121_Weights,
)

from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)
#Wenqing: Import an ad-hoc iFPC injected c++ set mem 
from ctypes import cdll
# lib = cdll.LoadLibrary('./libgeek.so')i
lib =cdll.LoadLibrary(os.path.abspath("../pytcppexp/libgeek.so")) 
 

BATCH_SZ: int = int(os.getenv("BATCH_SZ") or 1)
DATA_SET: str = os.getenv("DATA_SET") or "../data-set/rene/"

resize_id = int(os.environ['RESIZE_ID'])
resize_size = int(os.environ['SIZE'])

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
    def __init__(self, idx, control) -> None:
        self.idx = idx
        self.control = control
        if self.idx == 0:
            self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
            self.model: torch.nn.Module = deeplabv3_resnet101(weights=self.weights).eval().cuda()
            # self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
        elif self.idx == 1:
            self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
            # self.weights = DenseNet121_Weights.DEFAULT
            # self.model: torch.nn.Module = densenet121(weights=self.weights).eval().cuda()
            # self.weights =  RetinaNet_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = retinanet_resnet50_fpn(weights=self.weights).eval().cuda()
        elif self.idx == 2:
            self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.model: torch.nn.Module = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights).eval().cuda()
        elif self.idx == 3:
            self.weights = DenseNet121_Weights.DEFAULT
            self.model: torch.nn.Module = densenet121(weights=self.weights).eval().cuda()
        elif self.idx == 4:
            self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
         
        # if self.idx == 0:
        #     self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        #     self.model: torch.nn.Module = keypointrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
        # elif self.idx == 1:
        #     self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        #     self.model: torch.nn.Module = maskrcnn_resnet50_fpn_v2(weights=self.weights).eval().cuda()
        # elif self.idx == 2:
        #     self.weights =  RetinaNet_ResNet50_FPN_Weights.DEFAULT
        #     self.model: torch.nn.Module = retinanet_resnet50_fpn(weights=self.weights).eval().cuda()
        # elif self.idx == 3:
        #     self.weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        #     self.model: torch.nn.Module = fcos_resnet50_fpn(weights=self.weights).eval().cuda()
        # elif self.idx == 4:
        #     self.weights = SSD300_VGG16_Weights.DEFAULT
        #     self.model: torch.nn.Module = ssd300_vgg16(weights=self.weights).eval().cuda()
        # elif self.idx == 5:
        #     self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        #     self.model: torch.nn.Module = ssdlite320_mobilenet_v3_large(weights=self.weights).eval().cuda()
        # elif self.idx == 6:
        #     self.weights = MobileNet_V3_Small_Weights.DEFAULT
        #     self.model: torch.nn.Module = mobilenet_v3_small(weights=self.weights).eval().cuda()
        # elif self.idx == 7:
        #     self.weights = MNASNet0_5_Weights.DEFAULT
        #     self.model: torch.nn.Module =  mnasnet0_5(weights=self.weights).eval().cuda()
        # elif self.idx == 8:
        #     self.weights = SqueezeNet1_1_Weights.DEFAULT
        #     self.model: torch.nn.Module =  squeezenet1_1(weights=self.weights).eval().cuda()
        # elif self.idx == 9:
        #     self.weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
        #     self.model: torch.nn.Module =  shufflenet_v2_x0_5(weights=self.weights).eval().cuda()
        # elif self.idx == 10:
        #     self.weights = ViT_L_32_Weights.DEFAULT
        #     self.model: torch.nn.Module =  vit_l_32(weights=self.weights).eval().cuda()
        # elif self.idx == 11:
        #     self.weights = DenseNet201_Weights.DEFAULT
        #     self.model: torch.nn.Module =  densenet201(weights=self.weights).eval().cuda()



            

        if os.environ['RESIZE'] == "false":
            #No resize FasterRCNN_ResNet50 720
            self.model_preprocess = self.weights.transforms()
            self.fid: int = 99
            img_path = f"../data-set/rene/{self.fid:010}.png"
            self.img: torch.Tensor =\
                self.model_preprocess(read_img(img_path)).unsqueeze(0).cuda()
        else:
            self.model_preprocess = self.weights.transforms()
            self.fid: int = 99
            img_path = f"../data-set/rene/{self.fid:010}.png"
            i = read_img(img_path).unsqueeze(0)
            def size_to_full(size):
                if size == 180: return (180, 320)
                elif size == 1440: return (1440, 2560)
                elif size == 720: return (720, 1280)
                elif size == 1080: return (1080, 1920)
            if self.idx == resize_id:
                i = F.interpolate(i, size_to_full(resize_size)) if resize_size != 0 else i
            self.img: torch.Tensor = i.cuda()

    def infer(self):
        st: int = perf_counter_ns()
        # print(f"starting inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        res: torch.Tensor = None
        if self.control and (self.idx == 1 or self.idx == 2):
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
                     f" {self.idx} {(perf_counter_ns() - st) / 1000000}\n")
        return res

    def run(self):
        while True:
            torch.cuda.nvtx.range_push("regionTest")
            self.infer()
            torch.cuda.nvtx.range_pop()
            if self.idx == 1 or self.idx == 2:
                sleep(1)


def main():  
    print("Process", int(sys.argv[1]))
    print("Device", int(sys.argv[2]))
    # read control switch
    doControl = False
    if len(sys.argv) > 3:
        print("Control?", sys.argv[3])
        doControl = True if sys.argv[3] == "--control" else False
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")
    # set the cuda device to use
    torch.cuda.set_device(int(sys.argv[2]))
    
    # tester: SchedulerTester = SchedulerTester(0)    
    tester: SchedulerTester = SchedulerTester(int(sys.argv[1]), doControl)

    sleep(1)

    tester.run()



    # print(sys.argv[1])


if __name__ == "__main__":
    # pyprof.init()
    main()
