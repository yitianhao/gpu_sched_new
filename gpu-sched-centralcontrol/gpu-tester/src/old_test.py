import os
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns, sleep
import random
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,

)
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

BATCH_SZ: int = int(os.getenv("BATCH_SZ") or 1)
DATA_SET: str = os.getenv("DATA_SET") or "/home/cc/data-set/rene"


def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)


class SchedulerTester():
    def __init__(self) -> None:
        # self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        # self.model: torch.nn.Module = fasterrcnn_resnet50_fpn(
                # weights=self.weights).eval().cuda()
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model: torch.nn.Module = mobilenet_v3_small(weights=self.weights).eval().cuda()
        self.model_preprocess = self.weights.transforms()
        self.fid: int = 0
        img_path = f"/home/cc/data-set/rene/{self.fid:010}.png"
        self.img: torch.Tensor =\
            self.model_preprocess(read_img(img_path)).unsqueeze(0).cuda()
        self.imgs: List[torch.Tensor] = [
            self.model_preprocess(read_img(os.path.join(
                DATA_SET, f"{int(random.randrange(0, 100)):010}.png"))).cuda()
            for _ in range(0, BATCH_SZ)]
        self.imgs = torch.stack(self.imgs)

    def infer(self):

        st: int = perf_counter_ns()
        res: torch.Tensor = self.model(self.imgs)
        print(f"{clock_gettime_ns(CLOCK_REALTIME)}"
                     f" infer latency: {perf_counter_ns() - st}")
        return res

    def run(self):
        while True:
            self.infer()
            sleep(0.5)

def main():

    tester: SchedulerTester = SchedulerTester()
    sleep(1)
    tester.run()


if __name__ == "__main__":
    main()
