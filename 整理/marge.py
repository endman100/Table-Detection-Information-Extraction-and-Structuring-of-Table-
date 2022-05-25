import os
import cv2
import numpy as np
import json
from PIL import Image
from tools import findGrid
from tools import drawer

def drawCounters(counters, img, color=(0, 0, 255)):
	for counter in counters:
		cv2.rectangle(img, (counter[0], counter[1]), (counter[2], counter[3]), color, 4)
	return img
def drawFillCounters(counters, img):
	for counter in counters:
		cv2.rectangle(img, (counter[0], counter[1]), (counter[2], counter[3]), (255, 255, 255), -1)
	return img
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
def delNoInCenterLine(rects, xCenters, yCenters):
	rectSum = []
	for rect in rects:
		flag = False
		for i, xCenter in enumerate(xCenters):
			if(rect[0]<xCenter and rect[2]>xCenter):
				flag=True
				break
		for i, yCenter in enumerate(yCenters):
			if(rect[1]<yCenter and rect[3]>yCenter):
				flag=True
				break

		if(flag):
			rectSum.append(rect[:])
	return rectSum
def delNoInCenter(rects, xCenters, yCenters, xKey=0, yKey=0):
	rectSum = []
	for rect in rects:
		flag = False
		for i, xCenter in enumerate(xCenters):
			if(rect[0]<xCenter and rect[2]>xCenter):
				flag+=1
				break
		for j, yCenter in enumerate(yCenters):
			if(rect[1]<yCenter and rect[3]>yCenter):
				flag+=1
				break

		if(flag == 2 or (i==xKey) or (j==yKey)):
			rectSum.append(rect[:])
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
def delUnreasonableRects(rects, ratio=5.1):
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
def readTxt(path):
	with open(path,'r') as f:  #讀取真實結果
		fStrs = f.readlines()
		fStrs = [fStr.replace("\n", "") for fStr in fStrs]
		polys = [[float(i) for i in fStr.split(" ")] for fStr in fStrs]
	rects = polysTorects(polys)
	return rects
def polysTorects(polys):
	# print(polys)
	return [[poly[0], poly[1], poly[4], poly[5]] for poly in polys]
def gridRule(rects):
	xCenter, yCenter = findGrid.findGrid(rects, image.shape)
	rectsTemp = delNoInCenter(rects, xCenter, yCenter)
	print("gridRule", len(rects), len(rectsTemp))
	if(len(rects) == len(rectsTemp)):
		return rectsTemp, xCenter, yCenter
	else:
		rectsTemp, xCenter, yCenter = gridRule(rectsTemp)
		return rectsTemp, xCenter, yCenter

targets = ["./unet-Result-best-rulebase/", "./yolo-Result-best-rulebase/"]
GTFlooders   = "../groundtruth/margeBlock/"
rects = []

for GTFlooderName in os.listdir(GTFlooders):
	print(GTFlooderName, end="\r")
	GTPath = os.path.join(GTFlooders, GTFlooderName)
	ImgPath = os.path.join(GTPath, GTFlooderName + ".jpg")
	GTTxtPath = os.path.join(GTPath, GTFlooderName + ".txt")
	classNamePath = os.path.join(GTPath, "classes.txt")
	image = cv2.imread(ImgPath)
	image2 = image.copy()
	image3 = image.copy()
	image4 = image.copy()

	rects = []
	flag = 1
	for target in targets:	
		predictPath = os.path.join(target, GTFlooderName + ".json")
		savePath = os.path.join(target, GTFlooderName + ".png")

		with open(predictPath, 'r') as f:
			predictRects = json.load(f)
		# print(len(predictRects))

		rects += predictRects
		if flag:
			print(flag)
			flag = 0

			for rect in predictRects:
				cv2.rectangle(image2, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 3)
		else:
			print(flag)
			for rect in predictRects:
				cv2.rectangle(image3, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 3)

	for rect in rects:
		cv2.rectangle(image4, (rect[0], rect[1]), (rect[2], rect[3]), (0, 125, 125), 3)

	rects = delUnreasonableRects(rects)
	rects = delInclude(rects)
	rects = margeRect(rects)

	for rect in rects:
		cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)
	# rects, xCenter, yCenter = gridRule(rects)

	# image = drawer.drawGrid(xCenter, yCenter, image, color=(255, 0, 0))
	# for rect in rects:
	# 	cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 3)


	SavePath = "./marge/"
	with open(os.path.join(SavePath, GTFlooderName + ".json"), 'w') as f:
		json.dump(rects, f)

	# for rect in rects:
	# 	cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
	cv2.imwrite(os.path.join(SavePath, GTFlooderName + "0.png"), image)
	cv2.imwrite(os.path.join(SavePath, GTFlooderName + "2.png"), image2)
	cv2.imwrite(os.path.join(SavePath, GTFlooderName + "3.png"), image3)
	cv2.imwrite(os.path.join(SavePath, GTFlooderName + "4.png"), image4)