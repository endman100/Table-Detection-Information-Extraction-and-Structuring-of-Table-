import os
import cv2
import numpy as np
def correctionImg(cutRGB, cutGRAY, cutBINARY, rect=None, boarder = 5):
	h, w, c = cutRGB.shape
	imgBinary = cutBINARY.copy()
	imgBinary = killLine(imgBinary)
	imgBinary = killboarder(imgBinary)
	sumX = np.sum(imgBinary, axis=0)/255
	sumY = np.sum(imgBinary, axis=1)/255
	startX = (sumX>0).argmax(axis=0) #找到第一個出現字的寬度位置
	startY = (sumY>0).argmax(axis=0) #找到第一個出現字的起始位置

	sumXFlip = sumX[::-1]
	sumYFlip = sumY[::-1]
	endX = len(sumXFlip) - (sumXFlip>1).argmax(axis=0) - 1 #找到最後一個出現字的寬度位置
	endY = len(sumYFlip) - (sumYFlip>1).argmax(axis=0) - 1 #找到最後一個出現字的起始位置
	
	startX = 0 if startX - boarder < 0 else startX - boarder
	startY = 0 if startY - boarder < 0 else startY - boarder
	endX = w if endX + boarder > w else endX + boarder
	endY = h if endY + boarder > h else endY + boarder
	# print(startX, startY, endX, endY)
	RGB, GRAY, BINARY = cutRGB[startY:endY, startX:endX], cutGRAY[startY:endY, startX:endX], cutBINARY[startY:endY, startX:endX]
	if (rect != None):
		return RGB, GRAY, BINARY, [rect[0]+startX, rect[1]+startY, rect[0]+endX, rect[1]+endY, rect[4], rect[5]]
		# return RGB, GRAY, BINARY, [rect[0], rect[1], rect[2], rect[3], rect[4]]
	else:
		return RGB, GRAY, BINARY
def killLine(img, door=240):
	sumX = np.sum(img, axis=0) / img.shape[0]
	sumY = np.sum(img, axis=1) / img.shape[1]
	img[:, np.where(sumX > door)] = 0
	img[np.where(sumY > door), :] = 0
	return img
def killboarder(img):
	img[0,:] = 0
	img[-1,:] = 0
	img[:,0] = 0
	img[:,-1] = 0
	return img
def killVerticalLine(img, door=240):
	sumX = np.sum(img, axis=0) / img.shape[0]
	img[:, np.where(sumX > door)] = 0
	return img
def splitLine(imgRGB, imgGRAY, imgBINARY, savePath=None, rect=None):
	img = imgBINARY.copy()
	h, w = img.shape

	kernel = np.ones((1, w//5),np.uint8)  
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	img = killVerticalLine(img)
	kernel = np.ones((h//10, 1),np.uint8) 
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	kernel = np.ones((1, w),np.uint8) 
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	img = killboarder(img)
	(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

	contours, hierarchy = cv2.findContours(img ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if(savePath):
		imgOut = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		cv2.drawContours(imgOut, contours, -1, (0, 0, 255), 1)
		cv2.imwrite(os.path.join(savePath+"CLOSE.png"), imgOut)
		cv2.imwrite(os.path.join(savePath+"Ori.png"), imgRGB)


	xCenter = []#找行輪廓中心
	for k in range(len(contours)):
		temp = np.array(contours[k])
		x, y = temp[:,:,0], temp[:,:,1]
		xMax, xMin, yMax, yMin = x.max(), x.min(), y.max(), y.min()
		height, width = yMax-yMin, xMax-xMin
		if(height > h / 10):
			xCenter.append((yMax+yMin)//2)
	xCenter.sort()

	xEdge = [0] #找行切割中線
	if(len(xCenter)) : tempEdge = xCenter[0]
	for i in range(1, len(xCenter)):
		xEdge.append((tempEdge + xCenter[i])//2)
		tempEdge = xCenter[i]
	xEdge.append(h)

	returnRGBs, returnGRAYs, returnBINARYs = [], [], []
	if(rect != None): returnRects = []
	if(len(xEdge)) : tempX = xEdge[0]#行切割
	for i in range(1, len(xEdge)):
		cutRGB, cutGRAY, cutBINARY = imgRGB[tempX:xEdge[i],:], imgGRAY[tempX:xEdge[i],:], imgBINARY[tempX:xEdge[i],:]
		if(rect != None): tempRect = [rect[0], rect[1]+tempX, rect[2], rect[1]+xEdge[i], rect[4], rect[5]]
		RGB, GRAY, BINARY, tempRect = correctionImg(cutRGB, cutGRAY, cutBINARY, rect=tempRect)#刪除多於空白
		if(savePath):
			cv2.imwrite(os.path.join(savePath+str(i)+"lingCut.png"), cutBINARY)
			cv2.imwrite(os.path.join(savePath+str(i)+"correction.png"), BINARY)
		returnRGBs.append(RGB) 
		returnGRAYs.append(GRAY) 
		returnBINARYs.append(BINARY) 
		if(rect != None): returnRects.append(tempRect)
		tempX = xEdge[i]
	if(rect != None):
		return returnRGBs, returnGRAYs, returnBINARYs, returnRects
	else:
		return returnRGBs, returnGRAYs, returnBINARYs
def splitLineImgs(imgRGBs, imgGRAYs, imgBINARYs, rects, savePath=None):
	if(savePath):
		print(savePath)
		if not os.path.exists(savePath):
			os.makedirs(savePath)
	returnImgRGBs, returnImgGRAYs, returnImgBINARYs, returnRects = [], [], [], []
	for i, img in enumerate(imgRGBs):
		if(savePath):
			imgRGB, imgGRAY, imgBINARY, cutRects = splitLine(imgRGBs[i], imgGRAYs[i], imgBINARYs[i], rect=rects[i]
									   , savePath=os.path.join(savePath, str(i)+"_"))
		else:
			imgRGB, imgGRAY, imgBINARY, cutRects = splitLine(imgRGBs[i], imgGRAYs[i], imgBINARYs[i], rect=rects[i])
		for j, _ in enumerate(cutRects):
			cutRects[j].append(i)
		returnImgRGBs += imgRGB
		returnImgGRAYs += imgGRAY
		returnImgBINARYs += imgBINARY
		returnRects += cutRects
	return returnImgRGBs, returnImgGRAYs, returnImgBINARYs, returnRects
if __name__ == '__main__':
	imgsFlooders = "angleresult"
	saveFlooders = "result"
	for path, childDir, childFile in os.walk(imgsFlooders):
		if(len(childDir) > 0):
			continue
		saveFlooder = path.replace(imgsFlooders, saveFlooders)
		if not os.path.exists(saveFlooder):
			os.makedirs(saveFlooder)
		for fileName in childFile:
			filePath = os.path.join(path, fileName)
			fileType = fileName.split(".")[0].split("img")[1]
			if(fileType == "BINARYs"):
				print(filePath)
				imageBinary = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
				imgRGB = cv2.imread(filePath.replace("BINARYs", "RGBs"))
				imgGRAY = cv2.imread(filePath.replace("BINARYs", "GRAYs"))

				cv2.imwrite(os.path.join(saveFlooder, fileName), imageBinary)
				imgs = splitLine(imgRGBs, imgGRAYs, imgBINARYs, savePath=os.path.join(saveFlooder, fileName))
				for i, img in enumerate(imgs):
					savePath=os.path.join(saveFlooder, fileName+str(i)+".png")
					cv2.imwrite(savePath, img)
			