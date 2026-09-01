import random
import cv2 as cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision.transforms import functional as F
from Custom_Dataset_ssd import CustomDataset
import torchvision.transforms as T
import numpy as np
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision
from torchvision.models.detection import _utils
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from ultralytics import YOLO

# ir ao .venv/lib/python3.12/site-packages/thop/profile.py e substituir:
# from distutils.version import LooseVersion
# por:
# from packaging.version import Version as LooseVersion

# seed for reproducibility
random.seed(42)
np.random.seed(42)

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO("YOLO12n_Esca_AllImages_trained.pt", verbose=False)
#model.load_state_dict(torch.load('best_ssd.pth', map_location=device))
model = model.to(device)
model.eval()

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

dataset = CustomDataset(im_path='Dataset', mode='train', transforms=get_valid_transform())
val_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


def calculate_detection_metrics(preds, targets, iou_threshold=0.7, score_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        gt_boxes = target['boxes']
        gt_labels = target['labels']

        # Filtrar por score
        #keep = pred_scores >= score_threshold
        keep = [idx for idx, score in enumerate(pred_scores) if score >= score_threshold]  
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            FN += len(gt_boxes)
            continue

        if len(gt_boxes) == 0:
            FP += len(pred_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        for i in range(len(pred_boxes)):
            max_iou, j = torch.max(ious[i], dim=0)
            if max_iou >= iou_threshold and j.item() not in matched_gt and pred_labels[i] == gt_labels[j]:
                TP += 1
                matched_gt.add(j.item())
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1

# print model summary
# print(model)

# for sample_count in tqdm(range(10)):
#     random_idx = random.randint(0, len(dataset))
#     im, target = dataset[random_idx]
#
#     output = model(im.unsqueeze(0))
#
#     # plot image
#     im = im.clone().detach().cpu().numpy().transpose(1, 2, 0)
#     #im = (im * 255).astype('uint8')
#     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#     # im = cv2.resize(im, (300, 300))
#
#     gt_im = im.copy()
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     # Saving images with ground truth boxes
#     for idx, box in enumerate(target['boxes']):
#         x1, y1, x2, y2 = box.detach().cpu().numpy()
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#         gt_im = cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
#         text = target['labels'][idx].detach().cpu().item()
#         text_size, _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_PLAIN, 1, 1)
#         text_w, text_h = text_size
#         cv2.putText(gt_im, text=str(target['labels'][idx].detach().cpu().item()),
#                     org=(x1 + 5, y1 + 15),
#                     thickness=1,
#                     fontScale=1,
#                     color=[0, 0, 0],
#                     fontFace=cv2.FONT_HERSHEY_PLAIN)
#
#         cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     boxes = output[0]['boxes'].detach().cpu().numpy()
#     labels = output[0]['labels'].detach().cpu().numpy()
#     scores = output[0]['scores'].detach().cpu().numpy()
#
#     boxes = boxes[scores > 0.5]
#     labels = labels[scores > 0.5]
#     scores = scores[scores > 0.5]
#
#     for idx, box in enumerate(boxes):
#
#         if scores[idx] < 0.5:
#             continue
#
#         x1, y1, x2, y2 = box
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         text = labels[idx]
#         text_size, _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_PLAIN, 1, 1)
#         text_w, text_h = text_size
#         cv2.putText(im, text=str(text),
#                     org=(x1 + 5, y1 + 15),
#                     thickness=1,
#                     fontScale=1,
#                     color=[0, 0, 0],
#                     fontFace=cv2.FONT_HERSHEY_PLAIN)
#     plt.imshow(im)
#     plt.show()

prog_bar = tqdm(val_loader, total=len(val_loader))
target = []
preds = []

for i, data in enumerate(prog_bar):
    images, targets = data

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    images_np = [img.cpu().numpy().transpose(1, 2, 0) for img in images]  # To (H, W, C)

    with torch.no_grad():
        outputs = model(images_np)

    # For mAP calculation using Torchmetrics.
    #####################################
    for i in range(len(images_np)):
        true_dict = dict()
        preds_dict = dict()
        true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
        true_dict['labels'] = targets[i]['labels'].detach().cpu()
        preds_dict['boxes'] = outputs[i].boxes.xyxy.cpu()
        preds_dict['scores'] = torch.tensor(outputs[i].boxes.conf.cpu(), dtype=torch.float32)
        preds_dict['labels'] = torch.tensor(outputs[i].boxes.cls.cpu(), dtype=torch.int)
        preds.append(preds_dict)
        target.append(true_dict)
    #####################################

precision, recall, f1 = calculate_detection_metrics(preds, target, iou_threshold=0.5, score_threshold=0.3)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

metric = MeanAveragePrecision()
metric.update(preds, target)
metric_summary = metric.compute()

print(f'mAP@50: {metric_summary["map_50"]:.3}, mAP@50:95: {metric_summary["map"]:.3}')

# GFLOPs
#from thop import profile, clever_format
#import time

# ir ao .venv/lib/python3.12/site-packages/thop/profile.py e substituir:
# from distutils.version import LooseVersion
# por:
# from packaging.version import Version as LooseVersion

"""
macs, params = profile(model, inputs=(images,))
macs, params = clever_format([macs, params], "%.3f")

print(f"GFLOPs: {macs}")
print(f"Parameters: {params}")

with torch.no_grad():
    for i in range(10):
        start_time = time.time()
        y = model(images)
        elapsed = time.time() - start_time
        print(f'Inference time: {elapsed:.4f} seconds')
"""