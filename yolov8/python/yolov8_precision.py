#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import json
import torch
import random
import numpy as np
import time
from rknnlite.api import RKNNLite
from pycocotools.coco import COCO

def compute_iou(box1, box2):
    # box: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# ===============================================================
#               CONFIGURATION
# ===============================================================
MODEL_PATH = '/userdata/yolov8/models/yolov8_int8.rknn'
IMAGE_FOLDER_PATH = '/userdata/yolov8/datasets/val2017'
ANNOTATION_JSON_PATH = '/userdata/yolov8/datasets/annotations/instances_val2017.json'

OBJ_THRESH, NMS_THRESH = 0.4, 0.65 
IMG_SIZE = (640, 640)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light", "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant", "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite", "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ", "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa", "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse ","remote ","keyboard ","cell phone","microwave ", "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    return boxes[_class_pos], classes[_class_pos], scores

def nms_boxes(boxes, scores):
    x,y,w,h = boxes[:,0],boxes[:,1],boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]
    areas=w*h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i=order[0]
        keep.append(i)
        xx1=np.maximum(x[i],x[order[1:]])
        yy1=np.maximum(y[i],y[order[1:]])
        xx2=np.minimum(x[i]+w[i],x[order[1:]]+w[order[1:]])
        yy2=np.minimum(y[i]+h[i],y[order[1:]]+h[order[1:]])
        w1=np.maximum(0.0,xx2-xx1+1e-5)
        h1=np.maximum(0.0,yy2-yy1+1e-5)
        inter=w1*h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)
        inds=np.where(ovr <= NMS_THRESH)[0]
        order=order[inds+1]
    return np.array(keep)

def dfl(pos):

    x=torch.tensor(pos)
    n,c,h,w=x.shape
    p_num=4
    mc=c//p_num
    y=x.reshape(n,p_num,mc,h,w).softmax(2)
    acc_metrix=torch.arange(mc, dtype=torch.float).reshape(1,1,mc,1,1)
    return (y*acc_metrix).sum(2).numpy()

def box_process(position, origin_h, origin_w):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    grid = np.concatenate((col.reshape(1,1,grid_h,grid_w), row.reshape(1,1,grid_h,grid_w)), axis=1)
    stride_w = IMG_SIZE[0] / grid_w
    stride_h = IMG_SIZE[1] / grid_h
    stride = np.array([stride_w, stride_h, stride_w, stride_h]).reshape(1,4,1,1)
    
    position = dfl(position)
    box_xy = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy_640 = np.concatenate((box_xy[:,0:1], box_xy[:,1:2], box_xy2[:,0:1], box_xy2[:,1:2]), axis=1) * stride
    
    gain = min(IMG_SIZE[1]/origin_h, IMG_SIZE[0]/origin_w)
    pad_w = (IMG_SIZE[0] - origin_w * gain) / 2
    pad_h = (IMG_SIZE[1] - origin_h * gain) / 2
    
    xyxy_orig = xyxy_640.copy()
    xyxy_orig[:, [0, 2]] -= pad_w
    xyxy_orig[:, [1, 3]] -= pad_h
    xyxy_orig /= gain
    
   xyxy_orig[:, [0, 2]] = np.clip(xyxy_orig[:, [0, 2]], 0, origin_w)
    xyxy_orig[:, [1, 3]] = np.clip(xyxy_orig[:, [1, 3]], 0, origin_h)
    return xyxy_orig

def post_process(input_data, origin_h, origin_w):
    boxes, scores, classes_conf = [], [], []
    num_heads = 3
    for i in range(num_heads):
        box_tensor = input_data[i*num_heads]
        class_tensor = input_data[i*num_heads+1]
        
        boxes.append(box_process(box_tensor, origin_h, origin_w))
        classes_conf.append(class_tensor)
        
        score_tensor = input_data[i*num_heads + 2]
        scores.append(score_tensor)

    def sp_flatten(_in):
        ch=_in.shape[1]
        _in=_in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])
    
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds=np.where(classes==c)
        b,cl,s=boxes[inds],classes[inds],scores[inds]
        keep = nms_boxes(b, s)
        if len(keep)>0: 
            nboxes.append(b[keep])
            nclasses.append(cl[keep])
            nscores.append(s[keep])
            
    if not nboxes: 
        return None, None, None
        
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    shape=im.shape[:2]
    r=min(new_shape[0]/shape[0],new_shape[1]/shape[1])
    new_unpad=int(round(shape[1]*r)), int(round(shape[0]*r))
    dw,dh=new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw/=2
    dh/=2
    if shape[::-1]!=new_unpad: 
        im=cv2.resize(im,new_unpad,interpolation=cv2.INTER_LINEAR)
    top,bottom=int(round(dh-0.1)),int(round(dh+0.1))
    l,r=int(round(dw-0.1)),int(round(dw+0.1))
    return cv2.copyMakeBorder(im,top,bottom,l,r,cv2.BORDER_CONSTANT,value=color)

if __name__ == '__main__':
    model = RKNNLite()
    print(f"Loading RKNN model: {MODEL_PATH}")
    model.load_rknn(MODEL_PATH)
    print("Initializing runtime...")
    model.init_runtime()
    print("Runtime initialized.")

    coco_gt = COCO(ANNOTATION_JSON_PATH)
    cat_id_map = {v['id']: v['name'] for k, v in coco_gt.cats.items()}

    all_images = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"Error: No images found in {IMAGE_FOLDER_PATH}")
        sys.exit(1)

    # Randomly sample 5000 images from the dataset
    random.seed(42)
    sample_images = random.sample(all_images, 5000)

    total_TP, total_FP, total_FN = 0, 0, 0
    iou_thresh = 0.5
    total_infer_time = 0.0

    for idx, image_to_debug in enumerate(sample_images):
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_to_debug)
        img_src = cv2.imread(image_path)
        if img_src is None:
            print(f"[Warning] Skipping unreadable image: {image_path}")
            continue
            
        origin_h, origin_w = img_src.shape[:2]
        img_letterboxed = letterbox(img_src.copy(), IMG_SIZE)
        input_data = np.expand_dims(cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB), 0)
        
        start_time = time.time()
        outputs = model.inference(inputs=[input_data])
        infer_time = time.time() - start_time
        total_infer_time += infer_time
        
        pred_boxes, pred_classes, pred_scores = post_process(outputs, origin_h, origin_w)
        
        image_id = int(os.path.splitext(image_to_debug)[0])
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        if pred_boxes is None:
            pred_list = []
        else:
            pred_list = []
            for i in range(len(pred_boxes)):
                p_box = pred_boxes[i]
                p_cls_idx = int(pred_classes[i])
                p_s = pred_scores[i]
                p_box_xywh = [p_box[0], p_box[1], p_box[2] - p_box[0], p_box[3] - p_box[1]]
                pred_list.append({'bbox': p_box_xywh, 'class_index': p_cls_idx, 'score': p_s})

        gt_flags = [False] * len(gt_anns)
        pred_flags = [False] * len(pred_list)

        for pi, pred in enumerate(pred_list):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_anns):
                if gt_flags[gi]:
                    continue
                

                gt_cat_id = gt['category_id']
                pred_class_name = CLASSES[pred['class_index']]
                gt_class_name = cat_id_map.get(gt_cat_id, "N/A")

                if pred_class_name != gt_class_name:
                    continue

                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_thresh and best_gi != -1 and not gt_flags[best_gi]:
                gt_flags[best_gi] = True
                pred_flags[pi] = True
                
        TP = sum(pred_flags)
        FP = len(pred_flags) - TP
        FN = len(gt_flags) - sum(gt_flags)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        if (idx + 1) % 100 == 0 or (idx + 1) == len(sample_images):
            print(f"Processed {idx+1}/{len(sample_images)} images...")

    avg_infer_time = total_infer_time / len(sample_images) if len(sample_images) > 0 else 0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    
    print("\n================== Summary over {} images ==================".format(len(sample_images)))
    print(f"Total True Positives (TP): {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print(f"Average inference time per image: {avg_infer_time*1000:.2f} ms")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    model.release()