import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os
import json
from torchvision import models
import torch.nn as nn
import argparse

from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from torchcontrib.optim import SWA

import tqdm

import shutil

# from model.modelInception import myModel
from model.Unet import *
from tools.imgFolder import imgFlooder
# from tools.imgFolderCrop import imgFlooder
# from tools.FocalLoss import BCEFocalLoss

def test(model):
    count = 0
    loss = 0
    model.eval()
    with torch.no_grad():
        for step, (img, target) in enumerate(testloader):
            img, target= img.cuda(), target.cuda()
            output = model(img)
            loss = lossFunctionBCE(output, target.float())
            ans = (output+0.5).int()
            target = target.int()
            count += (ans == target).float().sum()/target.shape[0]/target.shape[1]/target.shape[2]/target.shape[3]
            print(step, end="\r")
            if step % 200 == 0 and step:
                break
    print("")
    return count.item()/len(testloader), loss.item()
def train(model, optimizer, EPOCH, trainloader):
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        model.train()
        for step, (img, target) in enumerate(trainloader):
            # print(target.shape)
            img, target = img.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(img)
            # print(output.squeeze(0).shape, target.shape)
            loss = lossFunctionBCE(output, target.float())
            loss.backward()
            optimizer.step()
            # print(step)
            if step % 10 == 0 and step:
                ans = (output+0.5).int()
                target = target.int()
                accuracy = (ans == target).float().sum()/target.shape[0]/target.shape[1]/target.shape[2]/target.shape[3]
                print('Train Epoch: {}/{} Loss:{:.6f} Accuracy:{:.6f}'.format(epoch, step, loss.item(), accuracy))
            if step % 1000 == 0 and step:
                temp = test(model)
                print("[acc, loss]=", temp)
                # torch.save(model, './result/version2_newdata_param/epoch-{}_step-{}_acc-{:.6f}.pkl'.format(epoch, step, temp[0]))
                torch.save(model.state_dict(), './result/version2_newdata_param/epoch-{}_step-{}_acc-{:.6f}.pkl'.format(epoch, step, temp[0]))
                model.train()

        imgPath = "./data/train/storeImg/"
        targetPath = "./data/train/storeTarget/"
        dataset = imgFlooder(imgPath=imgPath, jsonPath=targetPath)
        trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
        print("reload DataLoader len = ", len(trainloader))


imgPath = "../data/maskGt/train/img/"
targetPath = "../data/maskGt/train/target/"

imgPathTest = "../data/maskGt/test/img/"
targetPathTest = "../data/maskGt/test/target/"

dataset = imgFlooder(imgPath=imgPath, jsonPath=targetPath)
trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
print(len(trainloader))

testdataset = imgFlooder(imgPath=imgPathTest, jsonPath=targetPathTest)
testloader = DataLoader(testdataset, batch_size=1, shuffle=True)
print(len(testloader))
model = UNet(3, 1).cuda()
# model = torch.load("./result/version2_newdata/step-30000_acc-0.995762.pkl").cuda()



lossFunctionBCE = torch.nn.BCELoss()
# base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
# opt_Adam = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.001)
# lossFunctionMSE = torch.nn.MSELoss() 
opt_Adam = torch.optim.Adam(model.parameters(), lr=0.0001)
train(model, opt_Adam, 4, trainloader)  