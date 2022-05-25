import numpy as np 
import json
import cv2
import shapely
import os
import argparse
import csv
from tools import rotateImg
from tools import splitLine
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
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

def cutImage(PreFlooders):
    for filename in os.listdir(PreFlooders):
        name, ext = os.path.splitext(filename)
        if(ext != ".json"):
            continue
        print(filename, end="\r")

        ImgPath = os.path.join("../groundtruth/image/", name + ".jpg")

        predictPath = os.path.join(PreFlooders, name + ".json")
        saveRGBDir = os.path.join("./cutImage/", PreFlooders, "RGB", name)
        saveGRAYDir = os.path.join("./cutImage/", PreFlooders, "GRAY", name)
        saveBINARYDir = os.path.join("./cutImage/", PreFlooders, "BINARY", name)
        if not os.path.exists(saveRGBDir):
            os.makedirs(saveRGBDir)
        if not os.path.exists(saveGRAYDir):
            os.makedirs(saveGRAYDir)
        if not os.path.exists(saveBINARYDir):
            os.makedirs(saveBINARYDir)

        image = cv2.imread(ImgPath)
        with open(predictPath, 'r') as f:
            rects = json.load(f)
        imgRGBs, imgGRAYs, imgBINARYs, rotateRects = rotateImg.rotateImgs(image, rects)
        strRGBs, strGRAYs, strBINARYs, strRects = splitLine.splitLineImgs(imgRGBs, imgGRAYs, imgBINARYs, rotateRects)

        for i in range(len(strRGBs)):
            savePath = os.path.join(saveRGBDir, str(i).zfill(5)+".png")
            cv2.imwrite(savePath, strRGBs[i])
            savePath = os.path.join(saveGRAYDir, str(i).zfill(5)+".png")
            cv2.imwrite(savePath, strGRAYs[i])
            savePath = os.path.join(saveBINARYDir, str(i).zfill(5)+".png")
            cv2.imwrite(savePath, strBINARYs[i])

        with open(saveRGBDir + ".json", 'w') as f:
            json.dump(strRects, f, cls=NpEncoder)
        with open(saveGRAYDir + ".json", 'w') as f:
            json.dump(strRects, f, cls=NpEncoder)
        with open(saveBINARYDir + ".json", 'w') as f:
            json.dump(strRects, f, cls=NpEncoder)
def onlyCutImage(PreFlooders):
    for filename in os.listdir(PreFlooders):
        name, ext = os.path.splitext(filename)
        if(ext != ".json"):
            continue
        print(filename, end="\r")

        ImgPath = os.path.join("../groundtruth/image/", name + ".jpg")

        predictPath = os.path.join(PreFlooders, name + ".json")
        saveTYPE1Dir = os.path.join("./cutImage/", PreFlooders, "TYPE1", name)
        if not os.path.exists(saveTYPE1Dir):
            os.makedirs(saveTYPE1Dir)

        image = cv2.imread(ImgPath)
        with open(predictPath, 'r') as f:
            rects = json.load(f)
        imgRGBs, imgGRAYs, imgBINARYs, rotateRects = rotateImg.rotateImgs(image, rects, onlyCut=True)
        strRGBs, strGRAYs, strBINARYs, strRects = splitLine.splitLineImgs(imgRGBs, imgGRAYs, imgBINARYs, rotateRects)

        # print(len(strRects), len(strRGBs), "   ")
        for i in range(len(strRGBs)):
            savePath = os.path.join(saveTYPE1Dir, str(i).zfill(5)+".png")
            cv2.imwrite(savePath, strRGBs[i])

        with open(saveTYPE1Dir + ".json", 'w') as f:
            json.dump(strRects, f, cls=NpEncoder)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flooder', default='marge', help='what model want test')
    opt = parser.parse_args()

    PreFlooders = opt.flooder
    onlyCutImage(PreFlooders)