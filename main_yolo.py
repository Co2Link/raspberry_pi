from __future__ import division
from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from server.streamCapture import StreamCapture_http

cap = StreamCapture_http('http://192.168.137.164:8000/stream.mjpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="yolov3/data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="yolov3/config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="yolov3/weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="yolov3/data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode


    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    ret = True

    while ret:
        prev_time = time.time()
        ret, frame = cap.read()

        frame_gray = np.zeros_like(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame_gray[:,:,0] = gray
        frame_gray[:,:,1] = gray
        frame_gray[:,:,2] = gray

        img = transforms.ToTensor()(frame_gray)
        img, _ = pad_to_square(img, 0)
        img = resize(img, 416)
        input = Variable(img.type(Tensor))

        with torch.no_grad():
            detections = model(input[None,:,:,:])
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
        
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Inference Time: %s" % (inference_time))

        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, frame.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1
                
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # draw bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                # draw text
                cv2.rectangle(frame,(x1,y1),(x2,y1+30),color,cv2.FILLED)
                cv2.putText(frame, classes[int(cls_pred)],(x1,y1+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
        cv2.imshow('a',frame)
        if cv2.waitKey(1) == 27:
            exit(0)