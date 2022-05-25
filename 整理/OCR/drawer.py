import cv2
def drawCounters(counters, img, color=(0, 0, 255), width=4, predList=False, savePath=None):
	for i, counter in enumerate(counters):
		# print((counter[2]-counter[0])//100+1)
		cv2.rectangle(img, (counter[0], counter[1]), (counter[2], counter[3]), color, width)
		if(predList):
			cv2.putText(img, counter[7].upper(), (counter[0], counter[1])
				  	, cv2.FONT_HERSHEY_SIMPLEX, (counter[3]-counter[1])//100+1, (255, 0, 0), 2, cv2.LINE_AA)
	if(savePath):
		print("store " + savePath)
		cv2.imwrite(savePath, img)
	return img
def drawGrid(centerY, centerX, img, width=3, color=(0, 0, 255), colorKey=(0, 255, 255)):
	for i in range(len(centerY)):
		cv2.line(img, (centerY[i], 0), (centerY[i], img.shape[0]), color, width)
	for i in range(len(centerX)):
		cv2.line(img, (0, centerX[i]), (img.shape[1], centerX[i]), color, width)
	return img
def drawReferece(img, rects, yGrid, xGrid, color=(0, 255, 0)):
	for rect in rects:
		centerY = (rect[0]+rect[2])//2
		centerX = (rect[1]+rect[3])//2
		cv2.line(img, (centerY, centerX), (yGrid[rect[9]], xGrid[rect[8]]), color, 2)
	return img
