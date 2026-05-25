import cv2
import cpp_detect
import time
import numpy as np

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    y = np.empty_like(x)           
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    b1 = boxes1[:, np.newaxis, :]   # (N,1,4)
    b2 = boxes2[np.newaxis, :, :]   # (1,M,4)

    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])

    inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
    return inter / (area1[:, np.newaxis] + area2[np.newaxis, :] - inter)


def _nms_cv2(boxes, scores, iou_thres):
    """
    Greedy NMS implementation.
    """
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.intp)

    idx = cpp_detect.nms(boxes.astype(np.float32), scores.astype(np.float32), float(iou_thres))
    return np.array(idx, dtype=np.intp).flatten()


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                        agnostic=False, multi_label=False, labels=(), max_det=300):
    """
    Runs Non-Maximum Suppression (NMS) on inference results.
    Returns list of detections, one (n,6) array per image [xyxy, conf, cls].
    """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}'

    max_wh = 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label = multi_label and nc > 1
    merge = False

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]                              # confidence filter

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + 5), dtype=x.dtype)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 5] = 1.0
            x = np.concatenate((x, v), 0)

        if not x.shape[0]:
            continue

        # obj_conf × cls_conf  (in-place is fine; x is already a copy from the filter)
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:
            # argmax + max in one pass over columns
            cls_idx = x[:, 5:].argmax(axis=1)
            conf    = x[np.arange(len(x)), cls_idx + 5]
            mask    = conf > conf_thres
            x = np.concatenate(
                (box[mask], conf[mask, None], cls_idx[mask, None].astype(np.float32)), 1
            )

        if classes is not None:
            x = x[(x[:, 5:6] == np.asarray(classes)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        # Offset boxes per class so different classes never suppress each other
        c      = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes  = x[:, :4] + c
        scores = x[:, 4]

        #  C++ greedy NMS
        i = _nms_cv2(boxes, scores, iou_thres)

        if i.shape[0] > max_det:
            i = i[:max_det]

        if merge and 1 < n < 3_000:
            iou     = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = (np.dot(weights, x[:, :4]) / weights.sum(1, keepdims=True)).astype(np.float32)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

        if time.time() - t > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    if name_bgr_dict is None:
        name_bgr_dict = {'person': (0, 255, 0)}

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        left, top   = int(box[0]), int(box[1])
        right, bottom = int(box[2]), int(box[3])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        cv2.rectangle(image_boxes, (x, y), (x + text_w + line_thickness, y + text_h + line_thickness + baseline), bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2 * baseline + line_thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes