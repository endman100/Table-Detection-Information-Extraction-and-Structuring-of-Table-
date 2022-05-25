import numpy as np
import cv2
import colorsys
import PIL.Image as Image
import os
import json
from shutil import copyfile

def get_dominant_color(image):

    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    max_score = 0.0001
    dominant_color = None
    for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
        # 转为HSV标准
        saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
        y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
        y = (y-16.0)/(235-16)
 
        #忽略高亮色
        # if y > 0.9:
        #     continue
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = (r,g,b)
    return dominant_color
def findAngle(imgTemp, savePath=None):
    img = imgTemp.copy()
    if(savePath):
        imgMinBox = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0
    if(len(contours) > 1):
        h, w = img.shape
        kernel = np.ones((h//10, w//10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = 0
        maxRect = None
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)
            if(area > maxArea):
                maxArea = area
                maxRect = [box, rect[-1]]
        if(maxRect):
            angle = maxRect[1]
            if(angle < -45):
                angle = 90 + angle
            if(savePath):
                cv2.drawContours(img, [maxRect[0]], -1, (0, 0, 255), 3)
    if(savePath):
        cv2.imwrite(savePath, img)
    return angle
def rotateAngle(img, angle, savePath=None, scale=1.0):
    # print(len(img.shape))
    dominant_color = (0,0,0)
    dominant_color = get_dominant_color(img)
    dominant_color = [dominant_color[2], dominant_color[1], dominant_color[0]]
    # print(img.shape)
    h, w = img.shape[0], img.shape[1]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=dominant_color)
    if(savePath):
        cv2.imwrite(savePath, rotated)        
    return rotated
def grayToBinary(grayImg):
    ret, binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    whiteCount = np.sum(binaryImg[:,:] > 128)
    blackCount = np.sum(binaryImg[:,:] < 128)
    if(blackCount < whiteCount):
        ret, binaryImg = cv2.threshold(binaryImg, 128, 255, cv2.THRESH_BINARY_INV)
    return binaryImg
def cutImg(img, rects):
    returnImgs = []
    for rect in rects:
        imgCut = img[rect[1] : rect[3], rect[0] : rect[2]]
        returnImgs.append(imgCut)
    return returnImgs
def rotateImgs(image, rects, savePath=None, onlyCut=False):
    imgRGBs = []
    imgGRAYs = []
    imgBINARYs = []
    imgsCut = cutImg(image, rects)
    if(savePath):
        if not os.path.exists(savePath):
            os.makedirs(savePath)
    returnRects = []
    for i, img in enumerate(imgsCut):
        h, w, c = img.shape
        if(h == 0 or w == 0):
            continue
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBinary = grayToBinary(imgGray)
        angle = 0
        returnRects.append(rects[i]+[angle])
        if(onlyCut):
            imgRGBs.append(img)
            imgGRAYs.append(imgGray)
            imgBINARYs.append(imgBinary)
            continue
        if(savePath):
            saveImgPath = os.path.join(savePath, str(i)+"binary.jpg")
            cv2.imwrite(saveImgPath, imgBinary)
            saveImgPath = os.path.join(savePath, str(i)+"ori.jpg")
            cv2.imwrite(saveImgPath, img)
            angle = findAngle(imgBinary, savePath=os.path.join(savePath, str(i)+"MinBox.jpg"))
            imgRGBs.append(rotateAngle(img, angle, savePath=os.path.join(savePath, str(i)+"imgRotateRGB.jpg")))
            # imgGRAYs.append(rotateAngle(imgGray, angle, savePath=os.path.join(savePath, str(i)+"imgRotateGRAY.jpg")))
            # imgBINARYs.append(rotateAngle(imgBinary, angle, savePath=os.path.join(savePath, str(i)+"imgRotateBINARY.jpg")))
            imgGRAYs.append(rotateAngle(imgGray, angle))
            imgBINARYs.append(rotateAngle(imgBinary, angle))
        else:
            angle = findAngle(imgBinary)
            imgRGBs.append(rotateAngle(img, angle))
            imgGRAYs.append(rotateAngle(imgGray, angle))
            imgBINARYs.append(rotateAngle(imgBinary, angle))
        # print(rects[i]+[angle])
        
        
    return imgRGBs, imgGRAYs, imgBINARYs, returnRects
if __name__ == '__main__':
    imgsDir = "../groundtruth/marge/"
    preDir = "./unet-Result-Rulebase/"
    saveDir = "./result/"
    for flooder in os.listdir(imgsDir):

        imgPath = os.path.join(imgsDir, flooder, flooder+".jpg")
        image = cv2.imread(imgPath)
        print(imgPath)

        prePath = os.path.join(preDir, flooder+".json")
        with open(prePath, 'r') as f:
            rects = json.load(f)

        saveFlooder = os.path.join(saveDir, flooder)
        if not os.path.exists(saveFlooder):
            os.makedirs(saveFlooder)

        imgRGBs, imgGRAYs, imgBINARYs = rotateImgs(image, rects)
        for i in range(len(imgRGBs)):
            cv2.imwrite(os.path.join(saveFlooder, str(i)+"imgRGBs.jpg"), imgRGBs[i])
            cv2.imwrite(os.path.join(saveFlooder, str(i)+"imgGRAYs.jpg"), imgGRAYs[i])
            cv2.imwrite(os.path.join(saveFlooder, str(i)+"imgBINARYs.jpg"), imgBINARYs[i])


        


            
            


        

