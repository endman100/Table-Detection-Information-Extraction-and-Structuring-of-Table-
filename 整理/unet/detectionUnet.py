import torch
from torchvision.utils import save_image
import numpy as np
import cv2
import json
import os
from model.Unet import UNet
def saveTorchImg(img, fileName):
    saveImg = img.numpy().astype(np.uint8).transpose((1, 2, 0))
    saveImg = cv2.cvtColor(saveImg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fileName, saveImg)
class UnetModel():
    def __init__(self, modelStore):
        self.model = UNet(3, 1).cuda()
        self.model.load_state_dict(torch.load(modelStore))
        self.model.eval()
    def predict_cutresize(self, img, re_size, side_size):
        with torch.no_grad():
            h, w, c = img.shape
            imgResize = self.img_resize(img, re_size)
            imgTorch = torch.from_numpy(imgResize.transpose((2, 0, 1)))

            imgTorchs = self.cutImage(imgTorch, side_size)
            outputImgs = []
            for i in imgTorchs:
                output = self.model(i.cuda().float().div(255).unsqueeze(0))
                output = output.mul(255).byte().squeeze(0).cpu().numpy().transpose((1, 2, 0))
                outputImgs.append(output)
            predictImg = self.margeImage(outputImgs, [imgTorch.shape[1], imgTorch.shape[2]], side_size)
            predictImg = cv2.resize(predictImg, (w, h))
            imgs, rects = self.findContour(predictImg, img)

            rectsTemp = []
            for rect in rects:
                scoreAll = np.sum(predictImg[rect[0]:rect[2], rect[1]:rect[3]]) / 255.
                temp = (rect[2]-rect[0]) * (rect[3]-rect[1])
                score = scoreAll/temp
                rectsTemp.append([rect[1].item(), rect[0].item(),
                                  rect[3].item(), rect[2].item(), score])
        del imgTorch, imgTorchs, rects
        return imgs, rectsTemp, predictImg
    def findContour(self, predictImg, img):
        _, predictImg = cv2.threshold(predictImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imgs = []
        rects = []
        contours, hierarchy = cv2.findContours(predictImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for k in contours:
            temp = np.array(k)
            x = temp[:,:,0]
            y = temp[:,:,1]
            xMax = x.max()
            xMin = x.min()
            yMax = y.max()
            yMin = y.min()
            length = xMax-xMin
            width = yMax-yMin
            area = length*width
            imgs.append(img[yMin:yMax, xMin:xMax])
            rects.append([yMin, xMin, yMax, xMax])
        return imgs, rects
    def cutImage(self, image, sliceSize=1000):
        # print(image.shape)
        c, h, w = image.shape

        imgs = []
        for i in range(h//sliceSize*2+1):
            for j in range(w//sliceSize*2+1):
                img = torch.zeros((3, sliceSize, sliceSize))
                img[:] = 255
                temp = image[:,sliceSize//2*i: sliceSize//2*(i+2) if sliceSize//2*(i+2)<h else h
                              ,sliceSize//2*j: sliceSize//2*(j+2) if sliceSize//2*(j+2)<w else w]
                tc, th, tw = temp.shape
                # print(temp.shape,  img[:, :th, :tw].shape)
                img[:, :th, :tw] = temp
                # saveTorchImg(img, str(i)+"-"+str(j)+'.png')

                imgs.append(img)
        del img
        return imgs
    def margeImage(self, imgs, shape, sliceSize=1000):
        h = shape[0]
        w = shape[1]
        returnImg = np.zeros((h, w, 1))
        countImg = np.zeros((h, w, 1))

        maxi, maxj = h//sliceSize*2+1, w//sliceSize*2+1
        for i in range(maxi):
            for j in range(maxj):
                starth, startw = sliceSize//2*i, sliceSize//2*j
                endh = sliceSize//2*(i+2) if sliceSize//2*(i+2)<h else h
                endw = sliceSize//2*(j+2) if sliceSize//2*(j+2)<w else w
                th, tw = endh-starth, endw-startw
                # cv2.imwrite(str(i)+"-"+str(j)+'mask.png', imgs[maxi*i+j])
                returnImg[starth:endh,startw:endw] += imgs[maxj*i+j][0:th, 0:tw]
                countImg[starth:endh,startw:endw] += 1
        # returnImg = returnImg/countImg
        returnImg[returnImg>255] = 255
        return returnImg.astype(np.uint8)
    def img_resize(self, image, size):
        height, width = image.shape[0], image.shape[1]
        # 判断图片的长宽比率
        scale = width / height
        if width > height:
            img_new = cv2.resize(image, (size, int(size*scale)))
        else:
            img_new = cv2.resize(image, (int(size*scale), size))
        return img_new
def NMS(dets, thresh=0.5): 
    x1, y1, x2, y2, scores = [], [], [], [], []
    for i in dets:
        x1.append(i[0])
        y1.append(i[1])
        x2.append(i[2])
        y2.append(i[3])
        scores.append(i[4])
    x1, y1, x2, y2, scores = np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0: 
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return [dets[i] for i in keep]
def delLowScores(rects, thresh=0.4):
    returnRects = []
    for rect in rects:
        if(rect[4] > thresh):
            returnRects.append(rect)
    return returnRects
def margeRect(rects, IOUThreshold=0.1):
    rectSum = []
    for i in range(len(rects)):
        if(rects[i][0] == -1):
            continue
        for j in range(i + 1, len(rects)):
            # print(rects[i], rects[j], j)
            if(rects[j][0] == -1):
                continue
            IOU, intersect, aArea, bArea = countIOU(rects[i], rects[j])
            aIOU = intersect / aArea
            bIOU = intersect / bArea
            if(aIOU > IOUThreshold or bIOU > IOUThreshold):
                rects[j][0] = min(rects[i][0], rects[j][0])
                rects[j][1] = min(rects[i][1], rects[j][1])
                rects[j][2] = max(rects[i][2], rects[j][2])
                rects[j][3] = max(rects[i][3], rects[j][3])
                rects[j][4] = max(rects[i][4], rects[j][4])
                j = -1
                break
        if(j != -1):
            rectSum.append(rects[i])
    return rectSum
def delInclude(rects, includeThreshold=0.9):
    rectSum = []

    for i in range(len(rects)):
        if(rects[i][0] == -1):
            continue
        for j in range(i + 1, len(rects)):
            # print(rects[i], rects[j], j)
            if(rects[j][0] == -1):
                continue
            IOU, intersect, aArea, bArea = countIOU(rects[i], rects[j])
            if(aArea <= intersect and aArea / intersect > includeThreshold):
                j = -1
                break
            elif(bArea <= intersect and bArea / intersect > includeThreshold):
                rects[j][0] = -1
        if(j != -1):
            rectSum.append(rects[i])
    return rectSum
def countIOU(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    if left_line >= right_line or top_line >= bottom_line:
        return 0, -1, S_rec1, S_rec2
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0, intersect, S_rec1, S_rec2
def delSmallerThan8X8(rects):
    returnRects = []
    for rect in rects:
        h = (rect[2]-rect[0])
        w = (rect[3]-rect[1])
        if(h >= 8 and w >= 8):
            returnRects.append(rect)
    return returnRects
def delUnreasonableRects(rects, ratio=4):
    # print(type(rects))
    hList = []
    wList = []
    for rect in rects:
        h = (rect[2]-rect[0])//2
        w = (rect[3]-rect[1])//2

        hList.append(h)
        wList.append(w)

    # hList = np.array(hList)
    # wList = np.array(wList)
    hStd = np.std(hList, ddof=0)
    wStd = np.std(wList, ddof=0)
    hMean = np.mean(hList)
    wMean = np.mean(wList)
    # print("hStd: {:.4f} wStd: {:.4f}".format(hStd, wStd))
    # print("hMean:{:.4f} wMean:{:.4f}".format(hMean, wMean))
    returnRects = []
    for i, (h, w) in enumerate(zip(hList, wList)):
        if(h < hMean + hStd*ratio and h > hMean - hStd*ratio and w < wMean + wStd*ratio and w > wMean - wStd*ratio):
            returnRects.append(rects[i])
    if(len(returnRects) < len(rects)):
        return delUnreasonableRects(returnRects)
    return returnRects

def testResize():
    model = UnetModel(modelStore="./checkpoint/epoch-0_step-32000_Acc 0.99, TP 22527028, TN 236084788, FP 1405728, FN 1713583, Recall 0.93, Precision 0.94, F1 0.94,.pkl")
    path = r"E:\mainproject\科技部計畫OCR 20200722\groundtruth\image"
    saveDirs = r"../unet-Result/"
    
    for re_size in range(500, 5001, 100):
        for side_size in range(200, 2001, 100): #2401 max size
            print(str(re_size)+"-"+str(side_size))
            saveDir = os.path.join(saveDirs, str(re_size)+"-"+str(side_size))
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            else:
                continue
            for fileName in os.listdir(path):
                name, ext = os.path.splitext(fileName)
                filePath = os.path.join(path, fileName)
                savePath = os.path.join(saveDir, name+'.json')
                print(filePath)
                img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grayOut = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                imgsUnet, rectUnet, predictImg  = model.predict_cutresize(img, re_size, side_size)

                with open(savePath, 'w') as f:
                    json.dump(rectUnet, f)

                for rect in rectUnet:
                   cv2.rectangle(grayOut, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                saveImgPath = os.path.join(saveDir, fileName)
                saveImgPathMask = os.path.join(saveDir, fileName+"mask.png")
                cv2.imwrite(saveImgPath, grayOut)
                cv2.imwrite(saveImgPathMask, predictImg)
def testbest():
    model = UnetModel(modelStore="./checkpoint/epoch-0_step-32000_Acc 0.99, TP 22527028, TN 236084788, FP 1405728, FN 1713583, Recall 0.93, Precision 0.94, F1 0.94,.pkl")
    path = r"C:\Users\endman100\Desktop\科技部計畫OCR 20200722\groundtruth\image"
    saveDir = r"../unet-Result-best/"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for fileName in os.listdir(path):
        name, ext = os.path.splitext(fileName)
        filePath = os.path.join(path, fileName)
        savePath = os.path.join(saveDir, name+'.json')
        print(filePath)
        img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayOut = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        imgsUnet, rectUnet, predictImg  = model.predict_cutresize(img, 2300, 1500)

        with open(savePath, 'w') as f:
            json.dump(rectUnet, f)

        for rect in rectUnet:
           cv2.rectangle(grayOut, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        saveImgPath = os.path.join(saveDir, fileName)
        saveImgPathMask = os.path.join(saveDir, fileName+"mask.png")
        cv2.imwrite(saveImgPath, grayOut)
        cv2.imwrite(saveImgPathMask, predictImg)
def testbest():
    model = UnetModel(modelStore="./checkpoint/epoch-0_step-32000_Acc 0.99, TP 22527028, TN 236084788, FP 1405728, FN 1713583, Recall 0.93, Precision 0.94, F1 0.94,.pkl")
    path = r"C:\Users\endman100\Desktop\科技部計畫OCR 20200722\groundtruth\image"
    saveDir = r"../unet-Result-best/"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for fileName in os.listdir(path):
        name, ext = os.path.splitext(fileName)
        filePath = os.path.join(path, fileName)
        savePath = os.path.join(saveDir, name+'.json')
        print(filePath)
        img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # grayOut = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        imgsUnet, rectUnet, predictImg  = model.predict_cutresize(img, 2300, 1500)

        with open(savePath, 'w') as f:
            json.dump(rectUnet, f)

        for rect in rectUnet:
           cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        saveImgPath = os.path.join(saveDir, fileName)
        saveImgPathMask = os.path.join(saveDir, fileName+"mask.png")
        cv2.imwrite(saveImgPath, img)
        cv2.imwrite(saveImgPathMask, predictImg)
def testbestRulebase():
    model = UnetModel(modelStore="./checkpoint/epoch-0_step-32000_Acc 0.99, TP 22527028, TN 236084788, FP 1405728, FN 1713583, Recall 0.93, Precision 0.94, F1 0.94,.pkl")
    path = r"C:\Users\endman100\Desktop\科技部計畫OCR 20200722\groundtruth\image"
    # saveDir = r"../unet-Result-best-rulebase/"
    saveDir = r"../unet-path/"

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for fileName in os.listdir(path):
        name, ext = os.path.splitext(fileName)
        filePath = os.path.join(path, fileName)
        savePath = os.path.join(saveDir, name+'.json')
        print(filePath)
        img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # grayOut = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        imgsUnet, rectUnet, predictImg  = model.predict_cutresize(img, 2300, 1500)

        
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"00"+".png"))
        rectUnet = delSmallerThan8X8(rectUnet)
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"01"+"delSmallerThan8X8.png"))
        rectUnet = delUnreasonableRects(rectUnet, ratio=6)  
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"02"+"delUnreasonableRects.png"))      
        rectUnet = NMS(rectUnet)
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"03"+"NMS.png"), drawScore=True)
        rectUnet = delLowScores(rectUnet)
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"04"+"delLowScores.png"), drawScore=False)
        rectUnet = delInclude(rectUnet)
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"05"+"delInclude.png"))
        rectUnet = margeRect(rectUnet)
        save_rects(img, rectUnet, os.path.join(saveDir, fileName+"06"+"margeRect.png"))

        # with open(savePath, 'w') as f:
        #     json.dump(rectUnet, f)

        # for rect in rectUnet:
        #    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 10)
        # saveImgPath = os.path.join(saveDir, fileName)
        # saveImgPathMask = os.path.join(saveDir, fileName+"mask.png")
        # cv2.imwrite(saveImgPath, img)
        # cv2.imwrite(saveImgPathMask, predictImg)
def save_rects(image, rects, savePath, drawScore=False):
    imageT = drawCounters(rects, image.copy(), drawScore=drawScore)
    cv2.imwrite(savePath, imageT)
def drawCounters(counters, img, color=(0, 0, 255), drawScore=False):
    for counter in counters:
        cv2.rectangle(img, (counter[0], counter[1]), (counter[2], counter[3]), color, 4)
        if(drawScore):
            # print(counter[-1])
            cv2.putText(img, "{:.2f}".format(counter[-1]), (counter[0], counter[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
    return img
if __name__ == '__main__':
    testbestRulebase()



    