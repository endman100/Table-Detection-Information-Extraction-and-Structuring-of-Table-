import numpy as np 
import json
import cv2
import shapely
import os
import argparse
import csv
from shapely.geometry import Polygon,MultiPoint  #多边形
def minDistance(word1, word2):
    if not word1:
        return len(word2 or '') or 0

    if not word2:
        return len(word1 or '') or 0

    size1 = len(word1)
    size2 = len(word2)

    last = 0
    tmp = list(range(size2 + 1))
    value = None

    for i in range(size1):
        tmp[0] = i + 1
        last = i
        # print word1[i], last, tmp
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
                # print(last, tmp[j], tmp[j + 1], value)
            last = tmp[j+1]
            tmp[j+1] = value
        # print tmp
    strLen =  len(word2)
    acc2Temp = (strLen-value)/strLen
    if(acc2Temp < 0):
        acc2Temp = 0
    return acc2Temp
def yoloBoxToCv2(detection, shape):
    height = shape[0]
    width = shape[1]
    center_x = int(detection[1] * width)
    center_y = int(detection[2] * height)
    w = int(detection[3] * width)
    h = int(detection[4] * height)
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    cls = int(detection[0])
    return [y, x, y+h, x+w, cls]
def countIOU(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
def readTxt(path):
    with open(path,'r') as f:  #讀取真實結果
        fStrs = f.readlines()
        fStrs = [fStr.replace("\n", "") for fStr in fStrs]
        polys = [[float(i) for i in fStr.split(" ")] for fStr in fStrs]
    rects = polysTorects(polys)
    return rects
def drawPolys(image, Polys):
    for poly in Polys:
        points = np.array(poly)
        points = points.reshape(4, 2)
        cv2.polylines(img=image, pts=np.int32([points]), isClosed=True, color=(0,0,255), thickness=3)
def polysTorects(polys):
    # print(polys)
    return [[poly[0], poly[1], poly[4], poly[5]] for poly in polys]
def rectsTopolys(rects):
    return [[rect[0], rect[1], rect[2], rect[1],
             rect[2], rect[3], rect[0], rect[3]] for rect in rects]
def drawRects(image, Rects, color=(0, 0, 255), score=False):
    for rect in Rects:
        # print(rect)
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), color, 3)
        if(score):

            cv2.putText(image, str(int(rect[4]*100)), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

def countAcc(PreFlooders, GTFlooders, iouThreshold=0.1):
    for GTFlooderName in os.listdir(GTFlooders):
        print(GTFlooderName, end="\r")
        GTPath = os.path.join(GTFlooders, GTFlooderName)

        ImgPath = os.path.join(GTPath, GTFlooderName + ".jpg")
        GTTxtPath = os.path.join(GTPath, GTFlooderName + ".txt")
        classNamePath = os.path.join(GTPath, "classes.txt")

        predictPath = os.path.join(PreFlooders, GTFlooderName + ".json")
        savePath = os.path.join(PreFlooders, GTFlooderName + ".png")

        image = cv2.imread(ImgPath)

        with open(GTTxtPath,'r') as f:  #讀取真實結果
            fStrs = f.readlines()
            fStrs = [fStr.replace("\n", "") for fStr in fStrs]
            yoloRects = [[float(i) for i in fStr.split(" ")] for fStr in fStrs]
            GTRects = [yoloBoxToCv2(i, image.shape) for i in yoloRects]
            GTRects = [[rect[1], rect[0], rect[3], rect[2], rect[4]] for rect in GTRects]

        with open(predictPath, 'r') as f:
            predictRects = json.load(f)

        with open(classNamePath,'r') as f:  #讀取真實字串
            fStrs = f.readlines()
            GTClassName = [fStr.replace("\n", "") for fStr in fStrs]
        # print(GTRects[0:3])
        # print(GTClassName)
        # print(predictRects[0:3])

        # print(predictPolys, gtPolys)
        TP, FP, FN = 0, 0, 0
        pairs = []

        pres = predictRects.copy()
        gts = GTRects.copy()
        ious = np.zeros((len(pres), len(gts)))
        for i, gt in enumerate(gts):
            for j, pre in enumerate(pres):
                iou = countIOU(gt, pre)
                ious[j, i] = iou

        presUse = np.zeros((len(pres)))+1
        gtsUse = np.zeros((len(gts)))+1
        while(np.sum(ious) > 0 and np.sum(presUse) > 0 and np.sum(gtsUse) > 0):
            i, j = np.unravel_index(np.argmax(ious), ious.shape)
            # print(i, j, pres[i][7], GTClassName[gts[j][4]])
            if(minDistance(pres[i][7].upper(), GTClassName[gts[j][4]]) > 0.8):
                pairs.append([i, ious[i, j]])
                TP += 1

                presUse[i] = 0
                gtsUse[j] = 0
                ious[i,:] = 0
                ious[:,j] = 0
            else:
                ious[i,:] = 0
        # print(np.sum(ious), np.sum(presUse), np.sum(gtsUse))
        # print(np.sum(presUse), np.sum(gtsUse))
        FP = np.sum(presUse)
        FN = np.sum(gtsUse)
        
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        # print("TP:{} FP:{} FN:{} precision:{} recall:{}".format(TP, FP, FN, precision, recall))
        if(precision + recall == 0):
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        # drawRects(image, GTRects     , color=(0, 0, 255))
        # drawRects(image, predictRects, color=(255, 0, 0))

        writer.writerow([TP, FP, FN, precision, recall, F1])
        # cv2.imwrite(savePath, image)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flooder', default='marge_CRNN_BINARY', help='what model want test')
    opt = parser.parse_args()

    PreFlooders = os.path.join("./CrossTest-Result/", opt.flooder)
    GTFlooders   = "../groundtruth/marge/"
    with open(os.path.join("./CrossTest/", opt.flooder+'.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        countAcc(PreFlooders, GTFlooders)