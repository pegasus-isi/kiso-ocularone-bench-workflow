from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import trt_pose.coco
import math
import os
import numpy as np
import traitlets
import pickle
import trt_pose.models
import torch

# import torch2trt
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torchvision.transforms as transforms
import PIL.Image
import sys
from pose_estimation.trt_pose_hand.preprocessdata import Preprocessdata
import time

WIDTH = 224
HEIGHT = 224
DSIZE = (WIDTH, HEIGHT)
MODEL_WEIGHTS = "/home/ultraviolet/bin/pose_estimation/trt_pose_hand/model/hand_pose_resnet18_att_244_244.pth"
OPTIMIZED_MODEL = "model/hand_pose_resnet18_att_244_244_trt.pth"
DEVICE = torch.device("cpu")
MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])
SVM_TRAIN = False


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, DSIZE)
    frame = PIL.Image.fromarray(frame)
    frame = transforms.functional.to_tensor(frame).to(DEVICE)
    frame = frame.sub_(MEAN[:, None, None]).div_(STD[:, None, None])
    return frame[None, ...]


def draw_joints(image, joints):
    count = 0
    for i in joints:
        if i == [0, 0]:
            count += 1
    if count >= 3:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, (255, 0, 255), 1)
    for i in hand_pose["skeleton"]:
        if joints[i[0] - 1][0] == 0 or joints[i[1] - 1][0] == 0:
            break
        cv2.line(
            image,
            (joints[i[0] - 1][0], joints[i[0] - 1][1]),
            (joints[i[1] - 1][0], joints[i[1] - 1][1]),
            (0, 255, 0),
            1,
        )


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def load_pose_json():
    with open(
        "/home/ultraviolet/bin/pose_estimation/trt_pose_hand/preprocess/hand_pose.json",
        "r",
    ) as f:
        hand_pose = json.load(f)
    return hand_pose


class HandPoseModel(object):
    def __init__(self):
        self.hand_pose = load_pose_json()
        self.topology = trt_pose.coco.coco_category_to_topology(self.hand_pose)
        self.num_parts = len(self.hand_pose["keypoints"])
        self.num_links = len(self.hand_pose["skeleton"])
        self.parse_objects = ParseObjects(
            self.topology, cmap_threshold=0.12, link_threshold=0.15
        )
        self.draw_objects = DrawObjects(self.topology)
        self.model = self.load_torch_model()
        self.clf, self.gesture_type, self.preprocessdata = self.load_classifier()

    def load_tensorrt_model(self):
        # load tensorrt model
        # print("loading tensorrt torch model")
        # model_trt = torch2trt.TRTModule()
        # model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        return 0

    def load_torch_model(self):
        # load pytorch model
        model = trt_pose.models.resnet18_baseline_att(
            self.num_parts, 2 * self.num_links
        ).eval()
        model.load_state_dict(
            torch.load(MODEL_WEIGHTS, map_location=torch.device("cpu"))
        )
        return model

    def load_classifier(self):
        # load svm
        print("loading svm classifier")
        clf = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf"))
        preprocessdata = Preprocessdata(self.topology, self.num_parts)
        if SVM_TRAIN:
            pass
            # clf, predicted = preprocess.trainsvm(clf, joints_train, joints_test,\
            #                                   hand.labels_train, hand.labels_test)
            # filename = 'svmmodel.sav'
            # pickle.dump(clf, open(filename, 'wb'))
        else:
            filename = (
                "/home/ultraviolet/bin/pose_estimation/trt_pose_hand/svmmodel.sav"
            )
            clf = pickle.load(open(filename, "rb"))

        with open(
            "/home/ultraviolet/bin/pose_estimation/trt_pose_hand/preprocess/gesture.json",
            "r",
        ) as f:
            gesture = json.load(f)
            gesture_type = gesture["classes"]
        return clf, gesture_type, preprocessdata

    def detect_handpose(self, frames):
        outs = []
        start = time_sync()
        print(len(frames))
        for image in frames:
            print("started inference")
            data = preprocess(image)
            cmap, paf = self.model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
            image = cv2.resize(image, DSIZE)
            joints = self.preprocessdata.joints_inference(image, counts, objects, peaks)
            # draw_joints(image, joints)
            # draw_objects(image, counts, objects, peaks)
            dist_bn_joints = self.preprocessdata.find_distance(joints)
            gesture = self.clf.predict(
                [dist_bn_joints, [0] * self.num_parts * self.num_parts]
            )
            # print("inference plus classifier time:", end_classifier-start)
            # gesture_joints = gesture[0]
            # preprocessdata.prev_queue.append(gesture_joints)
            # preprocessdata.prev_queue.pop(0)
            # preprocessdata.print_label(image, preprocessdata.prev_queue, gesture_type)
            print("ended inference")
        end = time_sync()
        print("handpose inference time:", end - start)
        return gesture


# hp = HandPoseModel()
