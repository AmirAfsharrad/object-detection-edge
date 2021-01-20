import cv2
import numpy as np
import subprocess as sp
import time
import atexit
import yaml
import argparse
import torch
import torch.jit

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

class Detector:
    def __init__(self, cfg, detection_name = 'test.jpg'):
        self.detection_name = detection_name
        self.cfg = cfg
        self.imgsz = cfg['Model']['img-size']  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.weights = cfg['Model']['weights_path']
        self.device = torch_utils.select_device(device='cpu')
        
        # Initialize model
        self.model = Darknet(cfg['Model']['cfg_path'], self.imgsz)

        # Load weights
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:
            load_darknet_weights(self.model, self.weights)

        # Eval mode
        self.model.to(self.device).eval()

        # Get names and colors
        self.names = load_classes(cfg['Model']['names_path'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, img0):
        boxes = []

        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]
        if len(img.shape) == 2:
            img_new = np.expand_dims(img, -1)
            img_new = np.tile(img_new, (1,1,3))
            img = img_new
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.cfg['Model']['conf-thres'], self.cfg['Model']['iou-thres'],
                                   multi_label=False)

        # Process detections
        im0 = img0
        for i, det in enumerate(pred):  # detections for image i
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    temp = []
                    for bb in xyxy:
                        temp.append(bb.detach().numpy())
                    boxes.append(temp)
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

            # Save results (image with detections)

        return im0, boxes

##################################################################################################################
# config_path = "config_edge.yaml"

# with open(config_path) as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

# detection_boxes = []


# shot_frame = cv2.imread("zidane.jpg")
# detector = Detector(config)
# start = time.time()
# im_detections, detection_boxes = detector.detect(shot_frame)
# print(time.time() - start)
# cv2.imshow("Shot", im_detections)
# cv2.waitKey()

def init(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    detector = Detector(config)
    return detector

def detect_function(image, detector, visualize = True):
    im_detections, detection_boxes = detector.detect(image)
    if visualize:
        cv2.imshow("Shot", im_detections)
        # if cv2.waitKey(2) & 0xFF == ord('q'):
        #     break
