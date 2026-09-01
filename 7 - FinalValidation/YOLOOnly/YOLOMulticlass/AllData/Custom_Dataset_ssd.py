import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import cv2 as cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2



class CustomDataset(Dataset):
    def __init__(self, im_path, width=640, height=640, mode='train', transforms=None):

        self.transforms = transforms

        self.images, self.labels = self.load_images(im_path, mode)

        self.images.sort()
        self.labels.sort()

        self.mode = mode

        self.width = width
        self.height = height

    def __getitem__(self, idx):

        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        #image_resized /= 255.0

        file_txt = self.labels[idx]
        with open(file_txt, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []
        for line in lines:
            line = line.split()
            labels.append(int(line[0]))

            # convert yolo format to x1, y1, x2, y2
            x = float(line[1]) * self.width
            y = float(line[2]) * self.height
            w = float(line[3]) * self.width
            h = float(line[4]) * self.height
            x1 = x - w/2
            x2 = x + w/2
            y1 = y - h/2
            y2 = y + h/2

            if x1 < 0:
                x1 = 0
            if x2 > self.width -1:
                x2 = self.width -1
            if y1 < 0:
                y1 = 0
            if y2 > self.height -1:
                y2 = self.height-1

            boxes.append([x1, y1, x2, y2])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        else:
            image_resized = F.to_tensor(image_resized)

        return image_resized, target

    def __len__(self):
        return len(self.images)


    def load_images(self, im_path, mode='train'):

        fnames = glob.glob(os.path.join(im_path, 'images', mode, "*"))
        labels = glob.glob(os.path.join(im_path, 'labels', mode, "*"))

        imgs = []
        labels_list = []
        for fname in tqdm(fnames):
            imgs.append(fname)

        for label in tqdm(labels):
            labels_list.append(label)

        return imgs, labels_list

if __name__ == "__main__":
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def get_valid_transform():
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    dataset = CustomDataset(im_path='dataset', mode='train', transforms=get_valid_transform())
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
