import json
import os
import csv
import cv2
import numpy as np
import argparse
import sys
from PIL import Image
import importlib

import drawer
from recognitionVersion1 import recognitionModel11 as recognitionModel1
def grayToBinary(grayImg):
    ret, binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    whiteCount = np.sum(binaryImg[:,:] > 128)
    blackCount = np.sum(binaryImg[:,:] < 128)
    if(blackCount < whiteCount):
        ret, binaryImg = cv2.threshold(binaryImg, 128, 255, cv2.THRESH_BINARY_INV)
    return binaryImg

import shutil
def testImgFlooders(model, opt):
    imgDirs  = os.path.join("../cutImage/", opt.typedetect, opt.typedata)
    resultDir = os.path.join("../CrossTest-Result/", "{}_{}_{}".format(opt.typedetect, opt.modelName, opt.typedata))
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    print("imgFlooder", imgDirs)

    for dirname in os.listdir(imgDirs):
        dirpath = os.path.join(imgDirs, dirname)
        if not os.path.isdir(dirpath):
            continue
        print(dirpath)
        
        jsonPath = dirpath + ".json"
        with open(jsonPath, 'r') as f:
            rects = json.load(f)
        # print(jsonPath, len(rects), len(list(os.listdir(dirpath))))
        imagePath = os.path.join("../../groundtruth/image/", dirname+".jpg")
        image = cv2.imread(imagePath)

        model.initDataLoader(dirpath)
        texts = model.predict()

        for i, text  in enumerate(texts):
            text=text.upper()
            rects[i].append(texts[i])

        #清除預測為空字串的區塊
        clearRects = []
        for rect in rects:
            if len(rect[7]) > 0: clearRects.append(rect)
            
        with open(os.path.join(resultDir, dirname+".json"), 'w') as f:
            json.dump(clearRects, f)


        image = drawer.drawCounters(clearRects, image.copy(), color=(0, 0, 255), predList=True)
        cv2.imwrite(os.path.join(resultDir, dirname+".jpg"), image)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', default="dataColor.NoneResNetBiLSTMCTC", help='what model want test')
    parser.add_argument('--typedetect', default="unet-Result-best", help='what data want test')
    parser.add_argument('--typedata', default="RGB", help='what data want test')
    opt = parser.parse_args()
    cfg = importlib.import_module('cfg.'+opt.modelName)
    optClass = cfg.optClass
    recognition = recognitionModel1.recognitionModel1(optClass())

    testImgFlooders(recognition, opt)

    