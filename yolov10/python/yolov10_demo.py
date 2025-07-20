import torch 
import cv2
import numpy as np
import time
from collections import deque
from rknnlite.api import RKNNLite

# --- Model and Camera Parameters ---
YOLOv10_MODEL_PATH = '../models/yolov10_int8.rknn' 
VIDEO_SOURCE = 21  

OBJ_THRESH = 0.5
IMG_SIZE = (640, 640)  
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse ","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


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
    boxes = preds[..., :4].numpy()
    scores = preds[..., 4].numpy()
    classes = preds[..., 5].numpy().astype(np.int64)
    gain = min(IMG_SIZE[1] / origin_h, IMG_SIZE[0] / origin_w)
    pad_w = (IMG_SIZE[0] - origin_w * gain) / 2
    pad_h = (IMG_SIZE[1] - origin_h * gain) / 2
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= gain
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin_h)
    return boxes, classes, scores


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def draw(image, boxes, scores, classes, fps=None):
    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            left, top, right, bottom = [int(_b) for _b in box]
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if fps:
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def main():
    rknn_lite = RKNNLite()
    print(f'--> Loading RKNN model: {YOLOv10_MODEL_PATH}')
    ret = rknn_lite.load_rknn(YOLOv10_MODEL_PATH)
    if ret != 0:
        print(f'Error: Load RKNN model failed! Ret = {ret}')
        return
    print('Done.')

    print('--> Initializing runtime environment...')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print(f'Error: Init runtime environment failed! Ret = {ret}')
        return
    print('Done.')

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        rknn_lite.release()
        return

    fps_deque = deque(maxlen=30)
    print("\nPress 'q' to quit.")

    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        origin_h, origin_w = frame.shape[:2]

        img = letterbox(frame, IMG_SIZE, color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = np.expand_dims(img, 0)
        
        outputs = rknn_lite.inference(inputs=[img])
        
        boxes, classes, scores = post_process_yolov10(outputs, origin_h, origin_w)
        
        process_time = time.time() - frame_start_time
        fps_deque.append(process_time)
        smooth_fps = 1.0 / (sum(fps_deque) / len(fps_deque)) if fps_deque else 0

        result_image = draw(frame, boxes, scores, classes, smooth_fps)
        cv2.imshow('RKNN YOLOv10 Detection', result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rknn_lite.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
