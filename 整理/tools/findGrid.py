import numpy as np 
import cv2
import os
import json
import collections
from tqdm import tqdm
def delCenter(center, imgShape, minStr=8):
	centerF = np.flip(np.copy(center))[:]
	tempF = centerF[1:]-centerF[0:-1] < -minStr
	tempF = np.append(tempF, [True])

	tempNum = 0
	for i in range(len(tempF)-1, -1, -1):
		if(tempF[i] == True):
			tempNum = centerF[i]
		else:
			centerF[i] = tempNum

	centerN = np.copy(center)
	temp = centerN[1:]-centerN[0:-1] > minStr
	temp = np.append(temp, [True])
	tempNum = 0
	for i in range(len(temp)-1, -1, -1):
		if(temp[i] == True):
			tempNum = centerN[i]
		else:
			centerN[i] = tempNum
	centerM = (centerN*2//3 + np.flip(centerF)//3)


	return centerM
def getCenterList(rects, imgShape):
	centerX = []
	centerY = []

	sumX, sumY = 0, 0
	for rect in rects:
		centerX.append((rect[2]+rect[0])//2)
		centerY.append((rect[3]+rect[1])//2)

		sumX += rect[2]-rect[0]
		sumY += rect[3]-rect[1]
	averageX = sumX/len(centerX)
	averageY = sumY/len(centerY)
	# print(centerX, centerY)
	centerX = np.array(centerX)
	centerY = np.array(centerY)
	centerX.sort()
	centerY.sort()
	# print(averageX, averageY)
	centerX = delCenter(centerX, imgShape, minStr=averageX/5)
	centerY = delCenter(centerY, imgShape, minStr=averageY/5)
	# print("after", centerX, centerY)

	# temp = np.bincount(centerX)
	# mask = temp>3
	# temp = temp[mask]
	# centerX = np.where(mask == True)[0]
	# mean, std = np.mean(temp), np.std(temp)
	# print(temp, np.mean(temp), np.std(temp))
	# mask = temp > mean/2
	# temp = temp[mask]
	# centerX = centerX[mask]
	# print(temp)
	# print(centerX)
	


	temp = np.bincount(centerX) > 2
	# print(np.bincount(centerX)[temp])
	centerX = np.where(temp == True)[0]

	temp = np.bincount(centerY) > 2
	# print(np.bincount(centerY)[temp])
	centerY = np.where(temp == True)[0]

	# print("cut", centerX, centerY)
	return centerX, centerY
def findGrid(rects, imgShape):
	centerX, centerY = getCenterList(rects, imgShape)
	# print(centerX, centerY)
	return centerX, centerY