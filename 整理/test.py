import numpy as np 
import json
import cv2
import shapely
import os
import argparse
import csv
from shapely.geometry import Polygon,MultiPoint  #多边形
def countIOU(a, b, debug=False):#[x,y,x,y....], [x,y,x,y....]
    a=np.array(a).reshape(4, 2)   #四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

    b=np.array(b).reshape(4, 2)   #四边形二维坐标表示
    poly2 = Polygon(b).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if intersection != 0 or union != 0:
        iou=float(intersection) / union
    else:
        iou=0
    if debug: print("poly1.area:{} poly2.area:{} union:{} intersection:{} iou:{}".format(poly1.area, poly2.area, union, intersection, iou))
    return iou
def countIOURect(rec1, rec2):
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

def countAcc(PreFlooders, GTFlooders, iouThreshold=0.5):
    for GTFlooderName in os.listdir(GTFlooders):
        print(GTFlooderName, end="\r")
        GTPath = os.path.join(GTFlooders, GTFlooderName)

        ImgPath = os.path.join(GTPath, GTFlooderName + ".jpg")
        GTTxtPath = os.path.join(GTPath, GTFlooderName + ".txt")
        classNamePath = os.path.join(GTPath, "classes.txt")

        predictPath = os.path.join(PreFlooders, GTFlooderName + ".json")
        savePath = os.path.join(PreFlooders, GTFlooderName + ".png")

        image = cv2.imread(ImgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # print(imagePreMask)
        # predictPolys = readTxt(predictPath)
        with open(predictPath, 'r') as f:
            predictRects = json.load(f)
        gtRects = readTxt(GTTxtPath)

        # print(predictPolys, gtPolys)
        TP, FP, FN = 0, 0, 0
        pairs = []

        pres = predictRects.copy()
        gts = gtRects.copy()
        for i, gt in enumerate(gts):
            # print("{}/{}".format(i, len(gts)), end="\r")
            maxIOU = [0, -1]
            for j, pre in enumerate(pres):
                if(pre == -1):
                    continue
                iou = countIOURect(gt, pre)
                # print(gt,pre)
                if(iou > maxIOU[0]):
                    maxIOU[0] = iou
                    maxIOU[1] = j
            if(maxIOU[0] > iouThreshold):
                # print(maxIOU)
                pairs.append([i, maxIOU[1]])
                pres[maxIOU[1]] = -1
                TP += 1
            else:
                FN += 1
        FP = len(predictRects)-len(pairs)
        
        precision = TP/(TP + FP) if TP + FP else 0
        recall = TP/(TP + FN) if TP + FN else 0
        print("TP:{} FP:{} FN:{} precision:{} recall:{}".format(TP, FP, FN, precision, recall))
        if(precision + recall == 0):
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)        
        # drawPolys(image, gtPolys)
        # drawPolys(image, predictPolys)
        # drawRects(image, gtRects     , color=(0, 0, 255))
        # drawRects(image, predictRects, color=(255, 0, 0), score=True)

        writer.writerow([TP, FP, FN, precision, recall, F1])
        # cv2.imwrite(savePath, image)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flooder', default='EAST-Result-rulebase', help='what model want test')
    opt = parser.parse_args()

    PreFlooders = opt.flooder
    GTFlooders   = "../groundtruth/margeBlock/"
    with open('./csv/'+PreFlooders+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        countAcc(PreFlooders, GTFlooders)