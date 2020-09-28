import json
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import csv

import torch

def drawCounters(counters, img, color=(0, 0, 255)):
	for counter in counters:
		cv2.rectangle(img, (counter[1], counter[0]), (counter[3], counter[2]), color, 4)
	return img
def drawFillCounters(counters, img):
	for counter in counters:
		cv2.rectangle(img, (counter[1], counter[0]), (counter[3], counter[2]), (255, 255, 255), -1)
	return img
def yoloBoxToCv2(detection, shape):
	height = shape[0]
	width = shape[1]
	center_x = int(detection[1] * width)
	center_y = int(detection[2] * height)
	w = int(detection[3] * width)
	h = int(detection[4] * height)
	x = int(center_x - w / 2)
	y = int(center_y - h / 2)
	return [y, x, y+h, x+w]
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
def testMask(preImage, gtImage):
	preImage = preImage.reshape((-1))//255
	gtImage = gtImage.reshape((-1))//255
	# print(preImage.shape, gtImage.shape)

	preImage_torch = torch.from_numpy(preImage).cuda()
	gtImage_torch  = torch.from_numpy(gtImage).cuda()
	
	TP = torch.sum((gtImage_torch == 1) & (preImage_torch == 1)).item() #gpu version
	FN = torch.sum((gtImage_torch == 0) & (preImage_torch == 1)).item()
	FP = torch.sum((gtImage_torch == 1) & (preImage_torch == 0)).item()
	TN = torch.sum((gtImage_torch == 0) & (preImage_torch == 0)).item()

	# accuracy = confusion_matrix(preImage, gtImage, labels=[1, 0]) #cpu version
	# TP = accuracy[0,0]
	# FN = accuracy[0,1]
	# FP = accuracy[1,0]
	# TN = accuracy[1,1]
	# print(accuracy)
	P = TP/(TP+FP)
	R = TP/(TP+FN)
	F1 = 2*P*R/(P+R)

	Acc = (TP+TN)/(TP+TN+FP+FN)
	print("Accuracy =", Acc)
	print("Precision=", P)
	print("Recall   =", R)
	print("F1       =", F1)

	Acc = "{:.4f}".format(Acc*100)
	P = "{:.4f}".format(P*100)
	R = "{:.4f}".format(R*100)
	F1 = "{:.4f}".format(F1*100)
	writer.writerow([Acc, P, R, F1])

jsonFlooder = "./result/marge/all/"
GTFlooders   = "./groundtruth/marge"
storeFlooder = "./storeImg/"
with open('result.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)

	for GTFlooderName in os.listdir(GTFlooders):
		print(GTFlooderName)
		GTPath = os.path.join(GTFlooders, GTFlooderName)

		ImgPath = os.path.join(GTPath, GTFlooderName + ".jpg")
		GTTxtPath = os.path.join(GTPath, GTFlooderName + ".txt")
		classNamePath = os.path.join(GTPath, "classes.txt")

		jsonPath = os.path.join(jsonFlooder, GTFlooderName + ".json")

		image = cv2.imread(ImgPath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		imagePreMask = np.zeros(image.shape)
		with open(jsonPath) as f:  #讀取預測結果
			prediectData = json.load(f)
		# print(jsonData.shape)
		image = drawCounters(prediectData, image, color=(255, 0, 0))
		imagePreMask = drawFillCounters(prediectData, imagePreMask)

		with open(GTTxtPath,'r') as f:  #讀取真實結果
			fStrs = f.readlines()
			fStrs = [fStr.replace("\n", "") for fStr in fStrs]
			yoloRects = [[float(i) for i in fStr.split(" ")] for fStr in fStrs]
			cv2Rects = [yoloBoxToCv2(i, image.shape) for i in yoloRects]
		# image = drawCounters(cv2Rects, image, color=(0, 0, 255))

		imageMaskGt = np.zeros(image.shape)
		imageMaskGt = drawFillCounters(cv2Rects, imageMaskGt)
		testMask(imagePreMask, imageMaskGt)

		# pairRectsA = []
		# pairRectsB = []
		# TP = 0
		# for i, rect in enumerate(prediectData):
		# 	for j, cv2Rect in enumerate(cv2Rects):
		# 		iou = countIOU(rect, cv2Rect)[0]
		# 		if(iou > 0.5):
		# 			# print(iou, i, j)
		# 			pairRectsA.append(rect)
		# 			pairRectsB.append(cv2Rect)
		# 			TP+=1

		# image = drawCounters(pairRectsA, image, color=(0, 255, 0))
		cv2.imwrite(os.path.join(storeFlooder, GTFlooderName + ".jpg"), image)
		# # cv2.imshow('My Image', cv2.resize(image, (800, 800)))
		# # cv2.waitKey(0)
		# # cv2.destroyAllWindows()

		# FP = len(cv2Rects) - TP
		# print(TP/(TP+FP))