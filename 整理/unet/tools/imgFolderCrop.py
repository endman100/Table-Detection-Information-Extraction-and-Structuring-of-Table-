from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from tqdm import tqdm 
import time
import random

from torchvision import transforms as T
class imgFlooder(Dataset):
    def __init__(self, imgPath="", jsonPath="", transform=None, balance=False):
        self.imgPath = imgPath
        self.jsonPath = jsonPath
        self.balance = balance
        self.imgs = self.loadImgName(imgPath, jsonPath)
        
    def loadImgName(self, imgPath, jsonPath):
        returnList = []
        for i in os.listdir(imgPath):
            returnList.append([imgPath+i, jsonPath+i])
        return returnList
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        try:
            img = cv2.imread(self.imgs[index][0])
            img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)

            target = cv2.imread(self.imgs[index][1], cv2.IMREAD_GRAYSCALE)//128
            target = torch.from_numpy(target).unsqueeze(0)

            startx = random.randint(0, img.shape[1]-250)
            starty = random.randint(0, img.shape[2]-250)
            img = img[:, startx : startx+250, starty : starty+250]
            target = target[:, startx : startx+250, starty : starty+250]

            return img, target
        except:
            print(imgs[index])

# pathImgStore = "../data/train/storeImg/"
# pathTargetStore = "../data/test/storeTarget/"
# dataset = imgFlooder(imgPath=pathImgStore, jsonPath=pathTargetStore)
# trainloader = DataLoader(dataset, batch_size=5,shuffle=True)
# for step, (image, target) in enumerate(trainloader):
#     print("image", image.shape)
#     print("target", target.shape)