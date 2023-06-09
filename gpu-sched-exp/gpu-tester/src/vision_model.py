import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from utils import print_time

def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)

class VisionModel:
    # https://github.com/netx-repo/PipeSwitch/blob/f321d399e501b79ad51da13074e2aecda36cb06a/pipeswitch/worker_common.py#L40
    def insert_layer_level_sync(self, mod):
        def hook_terminate(mod, input, output):
            torch.cuda.synchronize()
            print("added sync")
        if len(list(mod.children())) == 0:
            mod.register_forward_hook(hook_terminate)
        else:
            for child in mod.children():
                self.insert_layer_level_sync(child)

    def __init__(self, config, device_id) -> None:
        self.config = config
        model_name = self.config['model_name']
        model_weight = self.config['model_weight']

        batch_size = self.config['batch_size'] if 'batch_size' in self.config else 1
        # for miniconda3/envs/torch/lib/python3.9/site-packages/torchvision/models/detection/transform.py"
        self.resize = config['resize']
        if self.resize:
            os.environ['RESIZE'] = "true"
        else:
            os.environ['RESIZE'] = "false"
        sync_level = self.config.get('sync_level', "")

        if getattr(torchvision.models.segmentation, model_weight, False):
            # a model from torchvision.models.segmentation
            self.weights = getattr(torchvision.models.segmentation, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.segmentation, model_name)
        elif getattr(torchvision.models.detection, model_weight, False):
            # a model from torchvision.models.detection
            self.weights = getattr(torchvision.models.detection, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.detection, model_name)
        elif getattr(torchvision.models, model_weight, False):
            # a model from torchvision.models or terminated with an exception
            self.weights = getattr(torchvision.models, model_weight).DEFAULT
            model_cls = getattr(torchvision.models, model_name)
        else:
            print(f"Unrecognized model weight {model_weight} and model name "
                  f"{model_name} in torchvision.", file=sys.stderr, flush=True)
            raise ValueError("Unrecognized model weight and model name in "
                             "torchvision.")
        with print_time('loading parameters', sys.stderr):
            self.model: torch.nn.Module = model_cls(weights=self.weights).eval().cuda()
        if sync_level == "layer":
            self.insert_layer_level_sync(self.model)
        self.resize_size = tuple(config['resize_size'])
        self.model_preprocess = self.weights.transforms()
        img_path = self.config['input_file_path']
        if self.resize:
            img = read_img(img_path).unsqueeze(0)
            img = F.interpolate(img, self.resize_size)
        else:
            # No resize FasterRCNN_ResNet50 720
            img = self.model_preprocess(read_img(img_path)).unsqueeze(0)
        img = torch.cat([img] * batch_size)
        self.img: torch.Tensor = img.cuda()

    def __call__(self):
        return self.model(self.img)
