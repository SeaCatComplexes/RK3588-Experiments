import cv2
import numpy as np
import time
from collections import deque
from rknnlite.api import RKNNLite


OBJ_THRESH = 0.5
NMS_THRESH = 0.5
IMG_SIZE = (640, 640)  
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop ","mouse ","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - x
    h = boxes[:, 3] - y
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position, origin_h, origin_w):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    grid = np.concatenate((col.reshape(1, 1, grid_h, grid_w), row.reshape(1, 1, grid_h, grid_w)), axis=1)
    
    stride_w = IMG_SIZE[0] / grid_w
    stride_h = IMG_SIZE[1] / grid_h
    stride = np.array([stride_w, stride_h, stride_w, stride_h]).reshape(1, 4, 1, 1)

    position = dfl(position)
    
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    
    xyxy = np.concatenate((box_xy[:, 0:1], box_xy[:, 1:2], box_xy2[:, 0:1], box_xy2[:, 1:2]), axis=1) * stride
    
    gain = min(IMG_SIZE[1] / origin_h, IMG_SIZE[0] / origin_w)
    pad_w = (IMG_SIZE[0] - origin_w * gain) / 2
    pad_h = (IMG_SIZE[1] - origin_h * gain) / 2
    
    xyxy[:, [0, 2], :, :] -= pad_w
    xyxy[:, [1, 3], :, :] -= pad_h
    xyxy /= gain
    
    xyxy[:, [0, 2], :, :] = np.clip(xyxy[:, [0, 2], :, :], 0, origin_w)
    xyxy[:, [1, 3], :, :] = np.clip(xyxy[:, [1, 3], :, :], 0, origin_h)
    
    return xyxy

def post_process(input_data, origin_h, origin_w):
    if not input_data: return None, None, None
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch 
    
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i], origin_h, origin_w))
        classes_conf.append(input_data[pair_per_branch * i + 1])

        scores.append(input_data[pair_per_branch * i + 2])


    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    
    if not nboxes: return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

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

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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

def main():

    YOLO_MODEL_PATH = '/userdata/yolo11/models/yolo11_int8.rknn'

    VIDEO_SOURCE = 21

    rknn_lite = RKNNLite()
    print(f'--> Loading RKNN model: {YOLO_MODEL_PATH}')
    ret = rknn_lite.load_rknn(YOLO_MODEL_PATH)
    if ret != 0: 
        print(f'Error: Load RKNN model failed! Ret = {ret}')
        exit(ret)
    print('Done.')

    print('--> Initializing runtime environment...')
    ret = rknn_lite.init_runtime()
    if ret != 0: 
        print(f'Error: Init runtime environment failed! Ret = {ret}')
        exit(ret)
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

        img = letterbox(frame, IMG_SIZE, color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        outputs = rknn_lite.inference(inputs=[img])
        
        boxes, classes, scores = post_process(outputs, origin_h, origin_w)
        
        process_time = time.time() - frame_start_time
        fps_deque.append(process_time)
        
        smooth_fps = 1.0 / (sum(fps_deque) / len(fps_deque)) if fps_deque else 0

        result_image = draw(frame, boxes, scores, classes, smooth_fps)
        cv2.imshow('RKNN DFL YOLOv8 Detection', result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rknn_lite.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()