import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import trt_pose.coco
import math
import os
import numpy as np
import traitlets
import trt_pose.models
import torch
import torch2trt
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torchvision.transforms as transforms
import PIL.Image
from preprocessdata import preprocessdata
from gesture_classifier import gesture_classifier
import csv
import multiprocessing
from jtop import jtop

WIDTH = 224
HEIGHT = 224
MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pth'
OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
DEVICE = torch.device('cuda')


def jetson_logging(file_path):
    #log jetson stats as the inference is going on
    print("jetson logging...")
    with jtop() as jetson:
        with open(file_path, 'w') as csv_file:
            stats = jetson.stats
            writer = csv.DictWriter(csv_file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
            while jetson.ok():
                stats = jetson.stats
                writer.writerow(stats)


def torch_to_trt():
    with open('preprocess/hand_pose.json', 'r') as f:
        hand_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(hand_pose)
    num_parts = len(hand_pose['keypoints'])
    num_links = len(hand_pose['skeleton'])
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_batch_size=32, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

if __name__ == "__main__":

    #convert torch model to trt
    #print("Converting model to tensorRT...")
    #torch_to_trt()
    

    inference_log_csv = 'logs/handpose_final.csv'
    jetson_log_csv = 'logs/handpose_jetson_log.csv'
    #add headers to inference csv file
    with open(inference_log_csv,'a') as csv_file:
        dict_writer = csv.DictWriter(csv_file, delimiter=',', \
                            fieldnames=['fps','inference_time','batch_size'])
        dict_writer.writeheader()

    p2 = multiprocessing.Process(target=jetson_logging, args=(jetson_log_csv,))
    p2.start()
    try:
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

        chunk_path = "~/Desktop/video_datasets/drone_1/5frames/test-0000.mp4"
        cap = cv2.VideoCapture(chunk_path)
        for batch in range(1,11):
           # _,frame = cap.read()

            dummy_data = torch.zeros((batch, 3, HEIGHT, WIDTH)).cuda()
            for i in range(20):
                starter,ender = torch.cuda.Event(enable_timing=True),\
                                torch.cuda.Event(enable_timing=True)
                starter.record()
                model_trt(dummy_data)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                inf_dict = {'fps':1/curr_time,'inference_time':curr_time, 'batch_size':batch}
                print(inf_dict)
                with open(inference_log_csv,'a',newline='') as csv_file:
                    dict_writer = csv.DictWriter(csv_file,inf_dict.keys())
                    dict_writer.writerow(inf_dict)
        cap.release()
    except Exception as e:
        print(e)
        p2.terminate()
    p2.terminate()

