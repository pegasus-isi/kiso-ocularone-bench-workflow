# -*- coding:utf-8 -*-
import sys

# sys.path.append('FaceMaskDetection/utils')
import cv2
import time
import csv
import argparse
import numpy as np
from PIL import Image
import torch
from FaceMaskDetection.utils.anchor_generator import generate_anchors
from FaceMaskDetection.utils.anchor_decode import decode_bbox
from FaceMaskDetection.utils.nms import single_class_non_max_suppression
from FaceMaskDetection.load_model.pytorch_loader import (
    load_pytorch_model,
    pytorch_inference,
)
import glob
from pathlib import Path

# model = load_pytorch_model('models/face_mask_detection.pth');
# model = load_pytorch_model('models/model360.pth')#model
# anchor configuration
# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: "Mask", 1: "NoMask"}


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def load_model():
    model = load_pytorch_model("models/model360.pth")
    return model


class LoadMaskModel(object):
    def __init__(self):
        self.model_path = str(Path(__file__).parent / "models/model360.pth")
        # self.model = load_pytorch_model(self.model_path)

    def load_model(self):
        return load_pytorch_model(self.model_path)


class DetectMask(object):
    def __init__(self, model):
        self.model = model

    def inference(
        self,
        frames,
        conf_thresh=0.5,
        iou_thresh=0.4,
        target_shape=(160, 160),
        draw_result=True,
        show_result=False,
    ):
        """
        Main function of detection inference
        :param image: 3D numpy array of image
        :param conf_thresh: the min threshold of classification probabity.
        :param iou_thresh: the IOU threshold of NMS
        :param target_shape: the model input size.
        :param draw_result: whether to daw bounding box to the image.
        :param show_result: whether to display the image.
        :return:
        """

        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        anchor_sizes = [
            [0.04, 0.056],
            [0.08, 0.11],
            [0.16, 0.22],
            [0.32, 0.45],
            [0.64, 0.72],
        ]
        anchor_ratios = [[1, 0.62, 0.42]] * 5

        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)

        # id2class = {0: 'Mask', 1: 'NoMask'}
        # image = np.copy(image)
        res = []
        s = time_sync()
        # model = load_pytorch_model('/home/ultraviolet/bin/FaceMaskDetection/models/model360.pth')
        for image in frames:
            count_mask = 0
            output_info = []
            height, width, _ = image.shape
            print(image.shape, height, width)
            image_resized = cv2.resize(image, target_shape)
            image_np = image_resized / 255.0  # 归一化到0~1
            image_exp = np.expand_dims(image_np, axis=0)

            image_transposed = image_exp.transpose((0, 3, 1, 2))
            print("image_transposed = ", image_transposed.shape)
            y_bboxes_output, y_cls_output = pytorch_inference(
                self.model, image_transposed
            )
            # y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
            print("y_bboxes_output = ", y_bboxes_output)
            print("y_bboxes_output shape = ", y_bboxes_output.shape)
            # e = time_sync()
            # print("mask detection inference time:",e-s)

            # remove the batch dimension, for batch is always 1 for inference.
            print(anchors_exp)
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]
            # To speed up, do single class NMS, not multiple classes NMS.
            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)

            # keep_idx is the alive bounding box after nms.
            keep_idxs = single_class_non_max_suppression(
                y_bboxes,
                bbox_max_scores,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
            )

            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                # if(id2class[class_id] == 'Mask'):
                #    count_mask = count_mask + 1
                bbox = y_bboxes[idx]
                # clip the coordinate, avoid the value exceed the image boundary.
                xmin = max(0, int(bbox[0] * width))
                ymin = max(0, int(bbox[1] * height))
                xmax = min(int(bbox[2] * width), width)
                ymax = min(int(bbox[3] * height), height)

                # if draw_result:
                #    if class_id == 0:
                #        color = (0, 255, 0)
                #    else:
                #        color = (255, 0, 0)
                #    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                #    cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

                # if(count_mask >= 1):
                #    print("Mask detected")
                output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
            return output_info
            # else:
            #    print("No mask detected")
        # return 0


"""
def run_on_video(video_paths, output_video_name, conf_thresh):
    video_paths = glob.glob(video_paths+'/*.mp4')
    for video_path in video_paths:
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        status = True
        idx = 0
        while status:
            start_stamp = time.time()
            status, img_raw = cap.read()
            #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            read_frame_stamp = time.time()
            if (status):
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
                #cv2.imshow('image', img_raw[:, :, ::-1])
                #cv2.waitKey(1)
                inference_stamp = time.time()
                # writer.write(img_raw)
                write_frame_stamp = time.time()
                idx += 1
                #print("%d of %d" % (idx, total_frames))
                #print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                 #                                                  inference_stamp - read_frame_stamp,
                  #                                                  write_frame_stamp - inference_stamp))
                inf_dict = {'inference_time':inference_stamp-read_frame_stamp,'batch_size':1}
                print(inf_dict)
                #with open('mask_log.csv','a',newline='') as csv_file:
                 #   dict_writer = csv.DictWriter(csv_file,inf_dict.keys())
                  #  dict_writer.writerow(inf_dict)
            else:
                break
        cap.release()
        # writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, default='img/demo2.jpg', help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model = load_model()
        inference(model, img, show_result=False, target_shape=(360, 360))
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)
"""
