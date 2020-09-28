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
        
        img = cv2.imread(self.imgs[index][0])
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)

        target = cv2.imread(self.imgs[index][1], cv2.IMREAD_GRAYSCALE)//128
        target = torch.from_numpy(target).unsqueeze(0)
        

        # angle = int(os.path.basename(self.imgs[index][0]).split(".")[0].split("_")[-1])

        return img, target

# pathImgStore = "../data/test/storeImg/"
# pathTargetStore = "../data/test/storeTarget/"
# dataset = imgFlooder(imgPath=pathImgStore, jsonPath=pathTargetStore)
# trainloader = DataLoader(dataset, batch_size=1,shuffle=True)
# # for step, (image, target) in tqdm(enumerate(trainloader)):
# #     # print("image", image.shape)
# #     # print("target", target.shape)
# #     pass
# for step, (image, target, angle) in enumerate(trainloader):
#     print("image", image.shape)
#     print("target", target.shape)
#     print(angle)
#     pass