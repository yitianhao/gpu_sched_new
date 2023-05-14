import csv
import glob
import math
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader


def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)
    # image = Image.open(img_path).convert("RGB")
    # return torchvision.transforms.ToTensor()(image)

class Video(VisionDataset):
    def __init__(self, root: str, fps: int = 30):
        super(Video, self).__init__(root)
        self.img_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.fps = 30


        # , sample_list_name: Union[str, List[str]],
        #          sample_list_root: str = None, subsample_idxs: List = None,
        #          transform: callable = None, target_transform: callable = None,
        #          resize_res: int = 32, coco: bool = False,
        #          merge_label: bool = True, segment_indices: List = None,
        #          label_type: str = 'human', **kwargs):
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return read_img(img_path)

    def __len__(self):
        return len(self.img_paths)



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
        self.csv_fh = open(os.path.join(config['output_file_path'], 'vision_results.csv'), 'w')
        self.csv_writer = csv.writer(self.csv_fh, lineterminator='\n')

        if getattr(torchvision.models.segmentation, model_weight, False):
            # a model from torchvision.models.segmentation
            self.weights = getattr(torchvision.models.segmentation, model_weight).DEFAULT
            model_cls = getattr(torchvision.models.segmentation, model_name)
        elif getattr(torchvision.models.detection, model_weight, False):
            self.csv_writer.writerow(['frame_id', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
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
        if os.path.isdir(img_path):
            self.img = None
            self.video = Video(img_path)
            self.dataloader = DataLoader(self.video, batch_size=batch_size)
            self.dataloader_iter = iter(self.dataloader)
            self.idx = 0
        else:
            if self.resize:
                img = read_img(img_path).unsqueeze(0)
                img = F.interpolate(img, self.resize_size)
            else:
                # No resize FasterRCNN_ResNet50 720
                img = self.model_preprocess(DataLoader(img_path)).unsqueeze(0)
            img = torch.cat([img] * batch_size)
            self.img: torch.Tensor = img.cuda()
            self.video = None

        # dummpy run
        if self.img is not None:
            self.model(self.img)
        elif self.video:
            img = self.model_preprocess(self.video[self.idx]).unsqueeze(0).cuda()
            self.model(img)

    def __del__(self):
        self.csv_fh.flush()
        self.csv_fh.close()

    def __call__(self):
        if self.img is not None:
            return self.model(self.img), None
        elif self.video:
            # for i in range(30):
            #     img = self.model_preprocess(self.video[self.idx]).unsqueeze(0).cuda()
            #     start_t = time.time()
            #     results = self.model(img)
            #     dur = time.time() - start_t
            #     boxes, labels, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
            #     for (x0, y0, x1, y1), label, score in zip(boxes, labels, scores):
            #         self.csv_writer.writerow([self.idx+1, float(x0), float(y0),
            #                                   float(x1), float(y1), int(label),
            #                                   float(score)])
            #     self.idx += 1
            # # self.idx += math.ceil(dur * self.video.fps)
            # return results
            # inputs = next(self.dataloader_iter)
            results = None
            start_t = time.time()
            for _, inputs in zip(range(4), self.dataloader_iter):
                dur = time.time() - start_t
                if dur * 1000 >= 700:
                    self.idx += 3
                    continue
                results = self.model(inputs.cuda())
                dur = time.time() - start_t
                if dur * 1000 >= 700:
                    self.idx += 3
                    continue
                for result in results:
                    boxes, labels, scores = result['boxes'], result['labels'], result['scores']
                    for (x0, y0, x1, y1), label, score in zip(boxes, labels, scores):
                        self.csv_writer.writerow([self.idx+1, float(x0), float(y0),
                                                  float(x1), float(y1), int(label),
                                                  float(score)])
                    self.idx += 1
            return results, dur
