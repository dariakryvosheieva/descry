import math
from collections import OrderedDict

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from craft import CRAFT


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_detector(filepath, device):
    model = CRAFT()
    net_param = torch.load(filepath, map_location=device)
    model.load_state_dict(copy_state_dict(net_param["craft"]))
    model = model.to(device)
    model.eval()
    return model


def process_for_detection(image, canvas_size=512):
    background_color = image.getpixel((0, 0))
    canvas = Image.new("RGB", (canvas_size, canvas_size), background_color)
    target_size = min(max(image.width, image.height), canvas_size)
    ratio = target_size / max(image.width, image.height)
    image = image.resize((int(image.width * ratio), int(image.height * ratio)))
    canvas.paste(image, (0, 0))
    trans = transforms.Compose([transforms.ToTensor()])
    return torch.cat([trans(canvas).unsqueeze(0)], 0), ratio


def get_det_boxes(textmap, linkmap, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    boxes = []
    mapper = []
    for k in range(1, n_labels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        if np.max(textmap[labels==k]) < text_threshold:
            continue

        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        mapper.append(k)
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        sx = max(sx, 0)
        sy = max(sy, 0)
        ex = min(ex, img_w)
        ey = min(ey, img_h)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(segmap!=0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        boxes.append(box)

    return boxes


def adjust_result_coordinates(boxes, ratio, ratio_net=2):
    if len(boxes) > 0:
        boxes = np.array(boxes)
        for k in range(len(boxes)):
            boxes[k] *= (ratio_net / ratio, ratio_net / ratio)
    return boxes


def test_net(net, image, device):
    image, ratio = process_for_detection(image)
    image = image.to(device)
    with torch.no_grad():
        y, features = net(image)
    boxes_list = []
    for out in y:
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()
        boxes = get_det_boxes(score_text, score_link)
        boxes = adjust_result_coordinates(boxes, ratio)
        boxes_list.append(boxes)
    return boxes_list


def get_textbox(detector, image, device):
    result = []
    boxes_list = test_net(detector, image, device)
    for boxes in boxes_list:
        single_img_result = []
        for box in boxes:
            box = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(box)
        result.append(single_img_result)
    return result
