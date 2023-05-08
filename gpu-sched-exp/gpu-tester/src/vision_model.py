import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)

class VisionModel:

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

        if getattr(torchvision.models.segmentation, model_weight, False):
            # a model from torchvision.models.segmentation
            self.weights = getattr(torchvision.models.segmentation, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.segmentation, model_name)
        elif getattr(torchvision.models.detection, model_weight, False):
            # a model from torchvision.models.detection
            self.weights = getattr(torchvision.models.detection, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.detection, model_name)
        elif getattr(torchvision.models, model_weight, False):
            # a model from torchvision.models or terminated with an excepton
            self.weights = getattr(torchvision.models, model_weight).DEFAULT
            model_cls = getattr(torchvision.models, model_name)
        else:
            raise ValueError("Unrecognized model weight and model name in "
                             "torchvision.")
        self.model: torch.nn.Module = model_cls(weights=self.weights).eval().cuda()
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
