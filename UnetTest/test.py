import torch
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from model.Unet import *
from tools.imgFolder import imgFlooder

def test(model, imgPathTest):
      TP = 0
      FN = 0
      FP = 0
      TN = 0
      with torch.no_grad():
        confusionMatrix = np.zeros((2,2))
        for count, i in enumerate(os.listdir(imgPathTest)):
            img = cv2.imread(imgPathTest + i)
            img = torch.from_numpy(img.transpose((2, 0, 1))).float().cuda().div(255)
            output = model(img.unsqueeze(0))

            #print(output)
            output = (output+0.5).int().squeeze(0).squeeze(0)


            target = cv2.imread(targetPathTest+i, cv2.IMREAD_GRAYSCALE)//128
            target = torch.from_numpy(target).cuda()
            
            TPTemp = torch.sum((target == 1) & (output == 1)).item()
            FNTemp = torch.sum((target == 1) & (output == 0)).item()
            FPTemp = torch.sum((target == 0) & (output == 1)).item()
            TNTemp = torch.sum((target == 0) & (output == 0)).item()
            ACC = (TPTemp+TNTemp)/(TPTemp+FNTemp+FPTemp+TNTemp)


            TP += TPTemp
            FN += FNTemp
            FP += FPTemp
            TN += TNTemp

            print("{}/1000 {} {} {} {} {}".format(count, ACC, TP, FN, FP, TN), end="\r")

      P = TP/(TP+FP)
      R = TP/(TP+FN)
      F1 = 2*P*R/(P+R)
      print()
      print("ACC=", (TP+TN)/(TP+FN+FP+TN))
      print("Precision=", P)
      print("Recall=", R)
      print("F1", F1)

    
imgPathTest = "./data/test3/storeImg/"
targetPathTest = "./data/test3/storeTarget/"
testPathStore = "./testImg/"


# for i in os.listdir("./result/"):
#       if len(i)>10:
#             print("./result/"+i)
#             model = torch.load("./result/"+i).cuda()
#             model.eval()
#             test(model, imgPathTest)

model = torch.load("./checkpoint/step-50000_acc-0.996163.pkl").cuda()
model.eval()
test(model, imgPathTest)