import os
import sys
import time
import csv
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
#FILE = Path(__file__).resolve()
#ROOT = FILE.parents[0]  # YOLOv5 root directory
#if str(ROOT) not in sys.path:
 #   sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import csv
#from detection.yolov5.models.common import DetectMultiBackend
#from detection.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
#from detection.yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
#                                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
#from detection.yolov5.utils.plots import Annotator, colors, save_one_box
#from load_model import Loadv5Model
from ultralytics import YOLO
# sys.path.append('.') ## add current directory path to python path
# from load_model import LoadYolov8mModel

from detection.yolov8m.load_model import LoadYolov8mModel

import subprocess
import ast

class DetectYolov8m(object):
    def __init__(self):
        #self.model = model
        # self.model = Yolov8mModel.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect(self, model, batch_frame, dnn_model):    
            # rel_depth = execute monodepth file
            # pass yolov8 and rel_depth to lt_avg script
            
            # res  = 1685.119 * lt_avg + 54.86
            # dummy_data = torch.zeros(len(batch_frame),3,640,640).to(self.device)
        #print(batch_frame.shape)
        #starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()
        #print('Reached here',len(batch_frame))
        #outs = self.model(batch_frame,size=640)
        #print('outs',outs)
        #ender.record()
        #torch.cuda.synchronize()
        #inference_time = starter.elapsed_time(ender)
        #print("inference time of yolov5s model:",inference_time/1000)
        #return outs

        # if dnn_model == "DISTANCE_ESTIMATION_OBJECT":
        #    output = self.model.predict(batch_frame[0], conf = 0.5)
        #    bboxes = output[0].boxes
        #    return bboxes

        bboxes_class_lefttop_rightdown_list = []
        if dnn_model == "CROWD_DENSITY":
            output = model.predict(batch_frame[0], classes=[0], conf = 0.5)
        else:
            output = model.predict(batch_frame[0], conf = 0.5)
        bboxes = output[0].boxes
        for box in bboxes:
            bboxes_class_lefttop_rightdown_list.append([box.cls.squeeze().tolist(), box.xyxy.squeeze().tolist()]) 
        result = bboxes_class_lefttop_rightdown_list

        return result
 
#yolov5_model = Loadv5Model().load_model()
#detect = Detectv5(yolov5_model)
#for i in range(10):
    #detect.detect([1])




