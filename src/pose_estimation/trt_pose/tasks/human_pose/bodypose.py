import os
import sys
import csv
import json
import torch
import cv2
import time
import numpy as np

# import torch2trt
import trt_pose.coco
import trt_pose.models
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

from pathlib import Path

WIDTH = 224
HEIGHT = 224
DSIZE = (WIDTH, HEIGHT)
MODEL_WEIGHTS = str(
    Path(__file__).parent.parent.parent
    / "resnet18_baseline_att_224x224_A_epoch_249.pth"
)
OPTIMIZED_MODEL = "/home/ultraviolet/bin/pose_estimation/trt_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
DEVICE = torch.device("cpu")
MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, DSIZE)
    frame = PIL.Image.fromarray(frame)
    frame = transforms.functional.to_tensor(frame).to(DEVICE)
    frame = frame.sub_(MEAN[:, None, None]).div_(STD[:, None, None])
    return frame[None, ...]


def get_keypoints(
    image, human_pose, topology, object_counts, objects, normalized_peaks
):
    height = image.shape[0]
    width = image.shape[1]
    keypoints = {}
    K = topology.shape[0]
    count = int(object_counts[0])

    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                keypoints[human_pose["keypoints"][j]] = (x, y)
    return keypoints


def load_pose_json():
    with (Path(__file__).parent / "human_pose.json").open() as f:
        human_pose = json.load(f)
    return human_pose


class BodyPoseModel(object):
    def __init__(self):
        self.human_pose = load_pose_json()
        self.num_parts = len(self.human_pose["keypoints"])
        self.num_links = len(self.human_pose["skeleton"])
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)
        self.model = self.load_torch_model()
        # self.model = self.load_tensorrt_model()

    def load_torch_model(self):
        # load pytorch model
        model = trt_pose.models.resnet18_baseline_att(
            self.num_parts, 2 * self.num_links
        ).eval()
        model.load_state_dict(
            torch.load(MODEL_WEIGHTS, map_location=torch.device("cpu"))
        )
        return model

    def load_tensorrt_model(self):
        # load tensorrt model
        # model = torch2trt.TRTModule()
        # model.load_state_dict(torch.load(OPTIMIZED_MODEL))
        return 0

    def detect_pose(self, frames):
        outs = []
        # starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        start = time.time()
        for frame in frames:
            t_frame = preprocess(frame)  # preprocess frame for input to the model
            cmap, paf = self.model(t_frame)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
            frame = cv2.resize(frame, DSIZE)
            keypoints = get_keypoints(
                frame, self.human_pose, self.topology, counts, objects, peaks
            )
            outs.append(keypoints)
            # self.draw_objects(frame, counts, objects, peaks)
        end = time.time()
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)/1000
        print("body pose inference_time:", end - start)
        return outs


# bp = BodyPoseModel()
