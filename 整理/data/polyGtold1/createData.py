import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import random
import os
import math
import json
strLib = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_','(',')','/','#']
def loadFonts(fontPath = "../fonts/"):
	fonts = []
	count = 0
	count2 = 0
	for i in os.listdir(fontPath):
		count2+=1
		for j in range(10, 60):
			try:

				temp = ImageFont.truetype(fontPath+i, j)
			except Exception as e:
				# print(i, j)
				pass
			else:
				fonts.append(temp)
				count+=1
	print(count, count2)
	return fonts
def drawGrid(gridSize, colorful, backgroundColor):
	#style
	#0 空的
	#1 邊框
	#2 邊框有缺
	#3 中心圓圈
	#4 中心圓圈小的
	#5 上方圓圈小的
	#6 下方圓圈小的
	img = np.zeros((gridSize, gridSize, 3), np.uint8)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	style = random.randint(0, 6)

	if(colorful):
		gridBackgroundColor = (random.randint(1,254),random.randint(1,254),random.randint(1,254),255)  #隨機顏色
		
	else:
		# print("backgroundColor", backgroundColor)
		gridBackgroundColor = backgroundColor[:]
	draw.rectangle([0, 0, gridSize, gridSize], fill = gridBackgroundColor)


	if(style == 1):
		draw.rectangle([0, 0, gridSize,gridSize], fill = None, outline=0, width=2)
	if(style == 2):
		if(random.randint(0, 1)):
			draw.rectangle([0, 0, gridSize, gridSize], fill = None, outline=0, width=2)
	if(style == 3):
		if(random.randint(0, 1)): #是否填色
			draw.ellipse([(0, 0), (gridSize, gridSize)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline ="black", width=2)
		else:
			draw.ellipse([(0, 0), (gridSize, gridSize)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline =None, width=2)
	if(style == 4):
		if(random.randint(0, 1)): #是否填色
			draw.ellipse([(gridSize*0.25, gridSize*0.25), (gridSize*0.75, gridSize*0.75)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline ="black", width=2)
		else:
			draw.ellipse([(gridSize*0.25, gridSize*0.25), (gridSize*0.75, gridSize*0.75)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline =None, width=2)
	if(style == 5):
		if(random.randint(0, 1)): #是否填色
			draw.ellipse([(gridSize*0.25, gridSize*0.5), (gridSize*0.75, gridSize*1)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline ="black", width=2)
		else:
			draw.ellipse([(gridSize*0.25, gridSize*0.5), (gridSize*0.75, gridSize*1)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline =None, width=2)
	if(style == 6):
		if(random.randint(0, 1)): #是否填色
			draw.ellipse([(gridSize*0.25, 0), (gridSize*0.75, gridSize*0.5)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline ="black", width=2)
		else:
			draw.ellipse([(gridSize*0.25, 0), (gridSize*0.75, gridSize*0.5)] , 
						   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
						   outline =None, width=2)

	img = np.array(img_pil)
	# img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	gridBackgroundColor = (gridBackgroundColor[0], gridBackgroundColor[1], gridBackgroundColor[2])
	return img, gridBackgroundColor
def drawText(img, target, yx, fonts, gridSize, gridColor, maxStrLen=13, textOrientation=random.randint(0, 2)):
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	font = fonts[random.randint(0, len(fonts) - 1)]

	#字方位編碼
	#0:靠左 1:靠中 2:靠右
	#textOrientation = (random.randint(0, 3), random.randint(0, 3))
	textOrientation = 1
	# textColor = (random.randint(0,255),random.randint(0,255),random.randint(0,255),255)  #隨機的字顏色
	textColor = (0,0,0,0) if random.randint(0, 1) else (255,255,255,255)
	# hasNoStr = random.randint(0, 10) > 8
	# if(hasNoStr):
	# 	return img, target, None
	

	strlen = random.randint(1, maxStrLen)
	drawUnderLine = random.randint(0, 10) > 8
	drawTopLine = random.randint(0, 10) > 8
	text = ''.join([strLib[x] for x in np.random.randint(len(strLib), size=strlen)])
	baseY =  yx[0]	  * gridSize + 3
	baseX =  yx[1]	  * gridSize + 3
	endY  = (yx[0] + 1) * gridSize - 3
	endX  = (yx[1] + 1) * gridSize - 3

	textwidth, textheight = draw.textsize(text, font=font)
	if textOrientation == 0:
		nextY = baseY
		nextX = baseX
	elif textOrientation == 1:
		nextY = (baseY + endY) // 2 - textheight//2
		nextX =  baseX
	elif textOrientation == 2:
		nextY = endY - textheight * 2
		nextX = baseX
	startX = nextX
	startY = nextY
	maxX = 0

	txtHeap = ""
	for char in text:
		twidth, theight = draw.textsize(char, font=font)
			
		if(nextX + twidth > endX):
			maxX = nextX
			nextX = baseX
			nextY += textheight

			if(nextY + textheight > endY):
				# print("break")
				nextY -= textheight
				break
		draw.text((nextX, nextY), char, font=font, fill=textColor)
		txtHeap += char
		if (drawUnderLine):
			draw.line((nextX, nextY + textheight, nextX + twidth, nextY + textheight), fill=textColor)
		if (drawTopLine):
			draw.line((nextX, nextY, nextX + twidth, nextY), fill=textColor)

		# ttwidth, ttheight = draw.textsize(char, font=font)
		if(nextY + textheight > endY):
			print("break nextY")
			break
		if(nextX + twidth > endX):
			print("break nextX")
			nextX += twidth
			break
		target[nextY : nextY + textheight, nextX : nextX + twidth, :] = 255
		nextX += twidth


	img = np.array(img_pil)

	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

	angle = random.randint(0, 359)

	baseY =  yx[0]	  * gridSize
	baseX =  yx[1]	  * gridSize
	endY  = (yx[0] + 1) * gridSize
	endX  = (yx[1] + 1) * gridSize
	img[baseY:endY, baseX:endX] = rotate(img[baseY:endY ,baseX:endX], angle, gridColor)

	targetTemp = rotate(target[baseY:endY ,baseX:endX], angle, (0,0,0))
	# ret, targetTemp = cv2.threshold(targetTemp,254,255,cv2.THRESH_BINARY)
	target[baseY:endY, baseX:endX] = np.expand_dims(targetTemp, axis = 2)

	# print(targetTemp.shape)
	contours, hierarchy = cv2.findContours(targetTemp,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if(len(contours) > 1):
		print(error)
		exit() 

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(targetTemp, (baseX+x, baseY+y), (baseX+x + w, baseY+y + h), (0, 0, 255), 2)
	maxX = nextX if nextX>maxX else maxX
	returnBox = [startX, startY
				,startX, nextY+textheight
				,maxX, nextY+textheight
				,maxX, startY, txtHeap]
	returnBox[0], returnBox[1] = get_point(returnBox[0], returnBox[1], (baseX+endX)//2, (baseY+endY)//2, angle)
	returnBox[2], returnBox[3] = get_point(returnBox[2], returnBox[3], (baseX+endX)//2, (baseY+endY)//2, angle)
	returnBox[4], returnBox[5] = get_point(returnBox[4], returnBox[5], (baseX+endX)//2, (baseY+endY)//2, angle)
	returnBox[6], returnBox[7] = get_point(returnBox[6], returnBox[7], (baseX+endX)//2, (baseY+endY)//2, angle)

	x1 = min(returnBox[0], returnBox[2], returnBox[4], returnBox[6])
	x2 = max(returnBox[0], returnBox[2], returnBox[4], returnBox[6])
	y1 = min(returnBox[1], returnBox[3], returnBox[5], returnBox[7])
	y2 = max(returnBox[1], returnBox[3], returnBox[5], returnBox[7])

	returnBox = [x1, y1, x1, y2, x2, y2, x2, y1]

	return img, target, returnBox
def drawLine(img, maxCount = 15):
	lineCount = random.randint(0, maxCount)
	imgShape = img.shape
	for i in range(lineCount):
		cv2.line(img, (random.randint(0, imgShape[1]), random.randint(0, imgShape[0])), 
					  (random.randint(0, imgShape[1]), random.randint(0, imgShape[0])), 
					  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
					  (random.randint(1, 3)))

def createImg(fonts):
	imgSizeHeight = 512#random.randint(800, 1500)  #隨機的圖片高度
	imgSizeWidth = 512#random.randint(800, 1500)  #隨機的圖片寬度
	gridSize = random.randint(50, 511)  #網格大小

	backgroundColor = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) #背景顏色
	img = np.zeros((imgSizeHeight, imgSizeWidth, 3), np.uint8)
	target = np.zeros((imgSizeHeight, imgSizeWidth, 1), np.uint8)
	img[:,:] = backgroundColor[:] #圖片背景顏色
	boxs = []

	
	colorful = random.randint(0, 1)  #是否彩色
	for x in range((imgSizeHeight // gridSize)):
		for y in range((imgSizeWidth // gridSize)):
			cv2Grid, gridColor = drawGrid(gridSize, colorful, backgroundColor)

			starty = y*gridSize
			endy = (y+1)*gridSize
			if(endy >= imgSizeWidth):
				endy = imgSizeWidth

			startx = x*gridSize
			endx = (x+1)*gridSize
			if(endx >= imgSizeHeight):
				endx = imgSizeHeight
			img[startx:endx, starty:endy] = cv2Grid[ : endx-startx, : endy-starty]

			img, target, boxsXY = drawText(img, target, (x, y), fonts, gridSize, gridColor)
			if(boxsXY):
				boxs.append(boxsXY)
	drawLine(img)
	return img, target, boxs
def rotate(image, angle, borderValue, center=None, scale=1.0):
	# 获取图像尺寸
	(h, w) = image.shape[:2]
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	# 执行旋转
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h), borderValue=borderValue)
	# 返回旋转后的图像
	return rotated
def get_point(x,y,cX,cY,angle):
	angle=360-angle
	new_x = (x - cX) * math.cos(math.pi / 180.0 * angle) - (y - cY) * math.sin(math.pi / 180.0 * angle) + cX
	new_y = (x - cX) * math.sin(math.pi / 180.0 * angle) + (y - cY) * math.cos(math.pi / 180.0 * angle) + cY
	return round(new_x), round(new_y) #四捨五入取整
if __name__ == '__main__' : 
	pathImgStore = "./test/img/"
	pathTargetStore = "./test/mask/"
	pathTargetStoreTxt = "./test/gt/"
	maxImgCount = 100
	# athImgStore = "./train/img/"
	# pathTargetStore = "./train/mask/"
	# pathTargetStoreTxt = "./train/gt/"
	# maxImgCount = 100000
	fonts = loadFonts()
	for i in range(1, maxImgCount+1):
		print(i, "/", maxImgCount)
		img, target, boxs = createImg(fonts)
		cv2.imwrite(pathImgStore + "img_" + str(i) + ".png", img)
		cv2.imwrite(pathTargetStore + "gt_img_" + str(i)+ ".png", target)
		
		data = []
		for box in boxs:
			data.append(",".join([str(num) for num in box]) + "\n")
		# print(data)
		with open(pathTargetStoreTxt + "gt_img_" + str(i)+ ".txt", "w", encoding='UTF-8') as f:
			f.writelines(data)