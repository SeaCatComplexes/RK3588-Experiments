#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch 
import os
import cv2
import sys
import json
import random
import numpy as np
import time
from rknnlite.api import RKNNLite
from pycocotools.coco import COCO

# ===============================================================
#               CONFIGURATION
# ===============================================================
MODEL_PATH = '../models/yolov10_fp32.rknn'
IMAGE_FOLDER_PATH = '../datasets/val2017'
ANNOTATION_JSON_PATH = '../datasets/annotations/instances_val2017.json'

OBJ_THRESH = 0.4 
IMG_SIZE = (640, 640) 

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light", "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant", "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite", "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ", "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa", "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse ","remote ","keyboard ","cell phone","microwave ", "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) for two bounding boxes.
    Box format: [x, y, w, h]
    """
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

def letterbox(im, new_shape=(640,640), color=(0,0,0)):
    """
    Resizes and pads an image to a target size.
    """
    shape = im.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    l, r = int(round(dw-0.1)), int(round(dw+0.1))
    return cv2.copyMakeBorder(im, top, bottom, l, r, cv2.BORDER_CONSTANT, value=color)


def dfl(position):
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w).softmax(2)
    acc_metrix = torch.arange(mc, dtype=torch.float).reshape(1,1,mc,1,1)
    return (y*acc_metrix).sum(2).numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    grid = np.concatenate((col.reshape(1,1,grid_h,grid_w), row.reshape(1,1,grid_h,grid_w)), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    position = dfl(position)
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def post_process_yolov10(input_data, origin_h, origin_w):
    if not input_data: return None, None, None
    max_det, nc = 300, len(CLASSES)
    boxes, scores = [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        scores.append(input_data[pair_per_branch * i + 1])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = torch.from_numpy(np.expand_dims(np.concatenate(boxes), axis=0))
    scores = torch.from_numpy(np.expand_dims(np.concatenate(scores), axis=0))
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))
    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    preds = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    mask = preds[..., 4] > OBJ_THRESH
    preds = preds[0][mask[0]]
    if preds.numel() == 0: return None, None, None
    boxes_np = preds[..., :4].numpy()
    scores_np = preds[..., 4].numpy()
    classes_np = preds[..., 5].numpy().astype(np.int64)
    gain = min(IMG_SIZE[1] / origin_h, IMG_SIZE[0] / origin_w)
    pad_w = (IMG_SIZE[0] - origin_w * gain) / 2
    pad_h = (IMG_SIZE[1] - origin_h * gain) / 2
    boxes_np[:, [0, 2]] -= pad_w
    boxes_np[:, [1, 3]] -= pad_h
    boxes_np /= gain
    boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, origin_w)
    boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, origin_h)
    return boxes_np, classes_np, scores_np


if __name__ == '__main__':
    model = RKNNLite()
    print(f"Loading RKNN model: {MODEL_PATH}")
    model.load_rknn(MODEL_PATH)
    print("Initializing runtime environment...")
    model.init_runtime()
    print("Runtime initialized.")

    coco_gt = COCO(ANNOTATION_JSON_PATH)
    cat_id_map = {v['id']: v['name'] for k, v in coco_gt.cats.items()}

    all_images = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"Error: No images found in {IMAGE_FOLDER_PATH}.")
        sys.exit(1)

    random.seed(42) 
    if len(all_images) > 1000:
        sample_images = random.sample(all_images, 5000)
    else:
        sample_images = all_images
    print(f"Will randomly sample {len(sample_images)} images from the dataset for evaluation...")
    
    total_TP, total_FP, total_FN = 0, 0, 0
    total_infer_time = 0.0
    iou_thresh = 0.5 

    for idx, image_filename in enumerate(sample_images):
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_filename)
        img_src = cv2.imread(image_path)
        if img_src is None:
            print(f"[Warning] Could not read image, skipping: {image_path}")
            continue
            
        origin_h, origin_w = img_src.shape[:2]
        img_letterboxed = letterbox(img_src.copy(), IMG_SIZE)
        input_data = np.expand_dims(cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB), 0)
        
        start_time = time.time()
        outputs = model.inference(inputs=[input_data])
        infer_time = time.time() - start_time
        total_infer_time += infer_time
        
        pred_boxes, pred_classes, pred_scores = post_process_yolov10(outputs, origin_h, origin_w)
        
        image_id = int(os.path.splitext(image_filename)[0])
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        gt_anns = coco_gt.loadAnns(ann_ids)

        pred_list = []
        if pred_boxes is not None:
            for i in range(len(pred_boxes)):
                p_box = pred_boxes[i]
                p_box_xywh = [p_box[0], p_box[1], p_box[2] - p_box[0], p_box[3] - p_box[1]]
                pred_list.append({'bbox': p_box_xywh, 'class_index': int(pred_classes[i]), 'score': pred_scores[i]})

        # Start matching to calculate TP, FP, FN
        gt_flags = [False] * len(gt_anns)
        pred_flags = [False] * len(pred_list)

        for pi, pred in enumerate(pred_list):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_anns):
                if gt_flags[gi]: continue 

                pred_class_name = CLASSES[pred['class_index']]
                gt_class_name = cat_id_map.get(gt['category_id'], "N/A")
                if pred_class_name != gt_class_name:
                    continue

                # Calculate IoU
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_thresh and best_gi != -1:
                if not gt_flags[best_gi]:
                    gt_flags[best_gi] = True
                    pred_flags[pi] = True
                
        TP = sum(pred_flags)
        FP = len(pred_flags) - TP
        FN = len(gt_flags) - sum(gt_flags)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        # Print progress
        if (idx + 1) % 100 == 0 or (idx + 1) == len(sample_images):
            print(f"Processed {idx+1}/{len(sample_images)} images...")

    avg_infer_time_ms = (total_infer_time / len(sample_images)) * 1000 if sample_images else 0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    
    # Print summary report
    print("\n================== Summary over {} images ==================".format(len(sample_images)))
    print(f"Total True Positives (TP): {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print(f"Average inference time per image: {avg_infer_time_ms:.2f} ms")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Release model resources
    model.release()