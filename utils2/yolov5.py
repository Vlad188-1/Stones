import cv2
import copy

import numpy as np
import torch

from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords


def inference_yolo(
        model, orgimg, device,
        img_size=640, conf_thres=0.05, iou_thres=0.5
):
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        if img_size==1280:
            img0 = cv2.resize(img0, (int(768), int(1280)), interpolation=interp)
        else:
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)



    # imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    imgsz = check_img_size(img_size, s=model.stride)  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # if len(img.shape) == 3:
    img = img[None]  # expand for batch dim

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orgimg.shape).round()
    return pred.cpu().numpy()

    
