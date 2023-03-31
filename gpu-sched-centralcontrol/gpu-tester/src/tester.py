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
    def __init__(self, idx) -> None:
        # self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        # self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(
                # weights=self.weights).eval().cuda()
        self.idx = idx
        if self.idx == 1:
            # self.weights = MNASNet0_5_Weights.DEFAULT
            # self.model: torch.nn.Module = mnasnet0_5(weights=self.weights).eval().cuda()
            # self.weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
            # self.model: torch.nn.Module = shufflenet_v2_x0_5(weights=self.weights).eval().cuda()
            # self.weights = SqueezeNet1_1_Weights.DEFAULT
            # self.model: torch.nn.Module = squeezenet1_1(weights=self.weights).eval().cuda()
            # self.weights = MobileNet_V3_Small_Weights.DEFAULT
            # self.model: torch.nn.Module = mobilenet_v3_small(weights=self.weights).eval().cuda()
            # self.model = SomeModel(800)
            # self.weights = DenseNet121_Weights.DEFAULT
            # self.model: torch.nn.Module = densenet121(weights=self.weights).eval().cuda()
            self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
        elif self.idx == 2:
            self.weights = ViT_L_32_Weights.DEFAULT
            self.model: torch.nn.Module = vit_l_32(weights=self.weights).eval().cuda()
        elif self.idx == 0:
            self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
            self.model: torch.nn.Module = deeplabv3_resnet101(weights=self.weights).eval().cuda()
            # self.weights = RegNet_Y_128GF_Weights.DEFAULT
            # self.model: torch.nn.Module = regnet_y_128gf(weights=self.weights).eval().cuda()
            # self.weights = ViT_H_14_Weights.DEFAULT
            # self.model: torch.nn.Module = vit_h_14(weights=self.weights).eval().cuda()
            # self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
            # self.weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = retinanet_resnet50_fpn(weights=self.weights).eval().cuda()
            # self.weights = FCOS_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = fcos_resnet50_fpn(weights=self.weights).eval().cuda()
            # self.weights = SSD300_VGG16_Weights.DEFAULT
            # self.model: torch.nn.Module = ssd300_vgg16(weights=self.weights).eval().cuda()
            # self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            # self.model: torch.nn.Module = ssdlite320_mobilenet_v3_large(weights=self.weights).eval().cuda()
            # self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            # self.model: torch.nn.Module = keypointrcnn_resnet50_fpn(weights=self.weights).eval().cuda()
            # self.model = SomeModel(3000)
            # self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            # self.model: torch.nn.Module = maskrcnn_resnet50_fpn_v2(weights=self.weights).eval().cuda()

        if True:
            self.model_preprocess = self.weights.transforms()
            self.fid: int = 99
            img_path = f"../data-set/rene/{self.fid:010}.png"
            # i = read_img(img_path).unsqueeze(0)
            self.img: torch.Tensor =\
                self.model_preprocess(read_img(img_path)).unsqueeze(0).cuda()
            # self.img: self.img if self.idx == 0 else F.interpolate(self.img, (180, 320))
            # i = F.interpolate(i, (720, 1280)) if self.idx == 1 else F.interpolate(i, (520, 924))
            # self.img: torch.Tensor = i.cuda()
            # self.img = F.interpolate(self.img, (360, 640))
            # self.imgs: List[torch.Tensor] = [
            #     self.model_preprocess(read_img(os.path.join(
            #         DATA_SET, f"{int(random.randrange(0, 100)):010}.png"))).cuda()
            #     for _ in range(0, BATCH_SZ)]
            # self.imgs = torch.stack(self.imgs)
            # self.imgs = F.interpolate(torch.stack(self.imgs), (360, 640))
            # self.imgs = F.interpolate(torch.stack(self.imgs), (360, 640)) if self.idx == 1 else torch.stack(self.imgs)
            # self.imgs = F.interpolate(torch.stack(self.imgs), (720, 1280)) if self.idx == 1 else F.interpolate(torch.stack(self.imgs), (520, 924))

    def infer(self):
        # ct = datetime.datetime.now().strftime("%H:%M:%S")
        # print(f"{ct} pyinfer{self.idx} enter device{torch.cuda.current_device()}"); 
        st: int = perf_counter_ns()
        # print(f"starting inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        res: torch.Tensor = None
        # if self.idx == 0 or self.idx == 1:
        if False:
            res = self.model.evaluate()
        else:
            if self.idx == 1:
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
            # ct = datetime.datetime.now().strftime("%H:%M:%S")
            # print(f"{ct} pyinfer{self.idx} exit\n")
        # print(torch.sum(res))
        # print(torch.sum(res))
        torch.cuda.synchronize()
        # print(f"{res}")

        # print(f"ending inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        print(f"{clock_gettime_ns(CLOCK_REALTIME)}"
                     f" idx: {self.idx}, infer latency: {(perf_counter_ns() - st) / 1000000}\n")
        return res

    def run(self):
        # for i in range(5):
        # # while True:
        #     self.infer()
        #     if self.idx == 1:
        #         sleep(1)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # print(f"{self.idx}: {self.img.size()}")
        while True:
            self.infer()
            if self.idx == 1:
                sleep(1)
            # if self.idx == 0:
                # sleep(1)
                # sleep(1) 
        # prof.export_chrome_trace("trace.json")


def main():  
    print("Process", int(sys.argv[1]))
    print("Device", int(sys.argv[2]))
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")
    # set the cuda device to use
    torch.cuda.set_device(int(sys.argv[2]))
    
    # tester: SchedulerTester = SchedulerTester(0)    
    tester: SchedulerTester = SchedulerTester(int(sys.argv[1]))

    sleep(1)

    tester.run()



    # print(sys.argv[1])


if __name__ == "__main__":
    # pyprof.init()
    main()
