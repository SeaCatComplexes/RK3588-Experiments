#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import cv2
import sys
import json
import random
import numpy as np
from rknnlite.api import RKNNLite
from pycocotools.coco import COCO

# --- CONFIGURATION ---
MODEL_PATH = '../models/yolov10_int8.rknn'
IMAGE_FOLDER_PATH = '../datasets/val2017' 
ANNOTATION_JSON_PATH = '../datasets/annotations/instances_val2017.json' 

OBJ_THRESH = 0.4
IMG_SIZE = (640, 640)
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light", "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant", "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite", "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ", "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa", "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse ","remote ","keyboard ","cell phone","microwave ", "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def dfl(position):
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.arange(mc).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
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


def letterbox(im, new_shape=(640,640), color=(0,0,0)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    l, r = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, l, r, cv2.BORDER_CONSTANT, value=color)


if __name__ == '__main__':
    model = RKNNLite()
    model.load_rknn(MODEL_PATH)
    model.init_runtime()

    coco_gt = COCO(ANNOTATION_JSON_PATH)
    cat_id_map = {v['id']: v['name'] for k, v in coco_gt.cats.items()}

    all_images = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"Error: No images found in {IMAGE_FOLDER_PATH}")
        sys.exit(1)

    img_src, image_to_debug = None, None
    while img_src is None:
        image_to_debug = random.choice(all_images)
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_to_debug)
        img_src = cv2.imread(image_path)
        if img_src is None:
            print(f"[Warning] Skipping unreadable image: {image_path}")

    # Preprocess image
    origin_h, origin_w = img_src.shape[:2]
    img_letterboxed = letterbox(img_src.copy(), IMG_SIZE)
    input_image = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, 0) # Add batch dimension

    # Inference
    outputs = model.inference(inputs=[input_image])
    
    # Post-process
    pred_boxes, pred_classes, pred_scores = post_process_yolov10(outputs, origin_h, origin_w)

    # Get Ground Truth
    image_id = int(os.path.splitext(image_to_debug)[0])
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    gt_anns = coco_gt.loadAnns(ann_ids)

    print("=" * 80)
    print(f"Visual Debug for: {image_to_debug}")
    print("=" * 80)

    print("\n--- GROUND TRUTH ---")
    if not gt_anns:
        print("No GT annotations for this image.")
    else:
        for i, ann in enumerate(gt_anns):
            class_name = cat_id_map.get(ann['category_id'], 'Unknown')
            bbox_xywh = np.round(ann['bbox'], 2).tolist()
            print(f"  GT {i+1}: Class='{class_name}', Bbox(xywh)={bbox_xywh}")

    print("\n--- PREDICTIONS ---")
    if pred_boxes is None:
        print("Model detected no objects.")
    else:
        for i in range(len(pred_boxes)):
            p_box = pred_boxes[i]
            p_cls_idx = int(pred_classes[i])
            p_score = pred_scores[i]
            
            # Convert xyxy to xywh for printing
            p_box_xywh = [p_box[0], p_box[1], p_box[2] - p_box[0], p_box[3] - p_box[1]]
            class_name = CLASSES[p_cls_idx]
            
            print(f"  Pred {i+1}: Class='{class_name}', Score={p_score:.4f}, Bbox(xywh)={np.round(p_box_xywh, 2).tolist()}")
    
    print("=" * 80)

    # --- Visualize and Save Result ---
    vis_img = img_src.copy()
    
    # Draw Ground Truth boxes (in Green)
    for ann in gt_anns:
        x, y, w, h = [int(v) for v in ann['bbox']]
        class_name = cat_id_map.get(ann['category_id'], 'Unknown')
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_img, f"GT:{class_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    # Draw Prediction boxes (in Red)
    if pred_boxes is not None:
        for i in range(len(pred_boxes)):
            x1, y1, x2, y2 = [int(v) for v in pred_boxes[i]]
            class_name = CLASSES[int(pred_classes[i])]
            score = pred_scores[i]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_img, f"Pred:{class_name} {score:.2f}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_path = f"./yolov10_debug_result.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"\nDebug image saved to: {output_path}")
    
    # Release model
    model.release()