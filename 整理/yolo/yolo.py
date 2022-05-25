import os
import cv2
import numpy as np
import json
from PIL import Image

from my import detectionYolo
from my import darknet
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
def delLowScores(rects, thresh=0.4):
	returnRects = []
	for rect in rects:
		if(rect[4] > thresh):
			returnRects.append(rect)
	return returnRects
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
def margeRect(rects, IOUThreshold=0):
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
def cutsizeTest():
	for i in range(600, 5000, 100):
		if not os.path.exists(os.path.join(saveDir, str(i).zfill(5))):
			os.makedirs(os.path.join(saveDir, str(i).zfill(5)))
		print(i)
		for fileName in os.listdir(imageFlooder):
			name, ext = os.path.splitext(fileName)
			print(fileName)
			if(ext != ".jpg"):
				continue
			filePath = os.path.join(imageFlooder, fileName)
			image = cv2.imread(filePath)
			img, detect = yolov4.image_detection_cutresize(filePath, size=i)

			rectsTemp = []
			print(len(detect))
			for label, confidence, bbox in detect:
				left, top, right, bottom = darknet.bbox2points(bbox)
				rectsTemp.append([left, top, right, bottom, confidence])

			savePath = os.path.join(saveDir, str(i).zfill(5), name+".json")
			
			with open(savePath, 'w') as f:
				json.dump(rectsTemp, f)
			img_save_path = os.path.join(saveDir, str(i).zfill(5), name+".png")
			image = drawCounters(rectsTemp, image)
			cv2.imwrite(img_save_path, image)
def test_sideWindow():
	if not os.path.exists(os.path.join(saveDir, "sidewindow")):
		os.makedirs(os.path.join(saveDir, "sidewindow"))
	for fileName in os.listdir(imageFlooder):
		name, ext = os.path.splitext(fileName)
		print(fileName)
		if(ext != ".jpg"):
			continue
		filePath = os.path.join(imageFlooder, fileName)
		image = cv2.imread(filePath)
		img, detect = yolov4.image_detection_cut(filePath)

		rectsTemp = []
		print(len(detect))
		for label, confidence, bbox in detect:
			left, top, right, bottom = darknet.bbox2points(bbox)
			rectsTemp.append([left, top, right, bottom, confidence])
		
		
		# rectsTemp = delSmallerThan8X8(rectsTemp)
		# rectsTemp = delUnreasonableRects(rectsTemp, ratio=6)
		# rectsTemp = NMS(rectsTemp)
		# rectsTemp = delLowScores(rectsTemp)
		# rectsTemp = margeRect(rectsTemp)

		savePath = os.path.join(saveDir, "sidewindow", name+".json")
		with open(savePath, 'w') as f:
			json.dump(rectsTemp, f)
		img_save_path = os.path.join(saveDir, "sidewindow", name+".png")
		image = drawCounters(rectsTemp, image)
		cv2.imwrite(img_save_path, image)
def test():
	if not os.path.exists(os.path.join(saveDir, "ori")):
		os.makedirs(os.path.join(saveDir, "ori"))
	for fileName in os.listdir(imageFlooder):
		name, ext = os.path.splitext(fileName)
		print(fileName)
		if(ext != ".jpg"):
			continue
		filePath = os.path.join(imageFlooder, fileName)
		image = cv2.imread(filePath)
		img, detect = yolov4.image_detection(filePath)

		rectsTemp = []
		print(len(detect))
		for label, confidence, bbox in detect:
			left, top, right, bottom = darknet.bbox2points(bbox)
			rectsTemp.append([left, top, right, bottom, confidence])
		
		
		# rectsTemp = delSmallerThan8X8(rectsTemp)
		# rectsTemp = delUnreasonableRects(rectsTemp, ratio=6)
		# rectsTemp = NMS(rectsTemp)
		# rectsTemp = delLowScores(rectsTemp)
		# rectsTemp = margeRect(rectsTemp)

		savePath = os.path.join(saveDir, "ori", name+".json")
		with open(savePath, 'w') as f:
			json.dump(rectsTemp, f)
		img_save_path = os.path.join(saveDir, "ori", name+".png")
		image = drawCounters(rectsTemp, image)
		cv2.imwrite(img_save_path, image)
def test_best():
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	for fileName in os.listdir(imageFlooder):
		name, ext = os.path.splitext(fileName)
		print(fileName)
		if(ext != ".jpg"):
			continue
		filePath = os.path.join(imageFlooder, fileName)
		image = cv2.imread(filePath)
		img, detect = yolov4.image_detection_cutresize(filePath, size=1208)

		rectsTemp = []
		print(len(detect))
		for label, confidence, bbox in detect:
			left, top, right, bottom = darknet.bbox2points(bbox)
			rectsTemp.append([left, top, right, bottom, confidence])
		
		save_rects(image, rectsTemp, os.path.join(saveDir, name+"00.png"))
		rectsTemp = delSmallerThan8X8(rectsTemp)
		save_rects(image, rectsTemp, os.path.join(saveDir, name+"01delSmallerThan8X8.png"))
		rectsTemp = delUnreasonableRects(rectsTemp, ratio=6)
		save_rects(image, rectsTemp, os.path.join(saveDir, name+"02delUnreasonableRects.png"))
		# rectsTemp = NMS(rectsTemp)
		# rectsTemp = delLowScores(rectsTemp)
		rectsTemp = margeRect(rectsTemp)
		save_rects(image, rectsTemp, os.path.join(saveDir, name+"03margeRect.png"))

		# savePath = os.path.join(saveDir, name+".json")
		# with open(savePath, 'w') as f:
		# 	json.dump(rectsTemp, f)
		img_save_path = os.path.join(saveDir, name+"04.png")
		image = drawCounters(rectsTemp, image)
		cv2.imwrite(img_save_path, image)
def save_rects(image, rects, savePath):
	imageT = drawCounters(rects, image.copy())
	cv2.imwrite(savePath, imageT)
imageFlooder = r"../../groundtruth/image"
# saveDir = "../yolo-path/"
saveDir = "../yolo-Result-best-rulebase/"
yolov4 = detectionYolo.YoloPredict(weights="./my/weights/20210122yolov4/yolov4_final.weights", config_file="./my/yolov4.cfg", data_file="./obj.data")
# test_sideWindow()
# test()
# cutsizeTest()
test_best()