import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import random
import os
import json
strLib = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_','(',')','/','#']

def drawGrid(draw, imgSize, gridSize):
	#style
	#0 空的
	#1 邊框
	#2 邊框有缺
	#3 中心圓圈
	#4 中心圓圈小的
	#5 上方圓圈小的
	#6 下方圓圈小的
	colorful = random.randint(0, 1)
	style = random.randint(0, 6)

	for y in range((imgSize[0] // gridSize) + 1):
		for x in range((imgSize[1] // gridSize) + 1):

			if(colorful):
				gridBackgroundColor = (random.randint(0,255),random.randint(0,255),random.randint(0,255),255)  #隨機顏色
				draw.rectangle([x*gridSize, y*gridSize, (x+1)*gridSize, (y+1)*gridSize], fill = gridBackgroundColor)
			if(style == 1):
				draw.rectangle([x*gridSize, y*gridSize, (x+1)*gridSize, (y+1)*gridSize], fill = None, outline=0, width=2)
			if(style == 2):
				if(random.randint(0, 1)):
					draw.rectangle([x*gridSize, y*gridSize, (x+1)*gridSize, (y+1)*gridSize], fill = None, outline=0, width=2)
			if(style == 3):
				if(random.randint(0, 1)): #是否填色
					draw.ellipse([(x*gridSize, y*gridSize), (x*gridSize+gridSize, y*gridSize+gridSize)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline ="black", width=2)
				else:
					draw.ellipse([(x*gridSize, y*gridSize), (x*gridSize+gridSize, y*gridSize+gridSize)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline =None, width=2)
			if(style == 4):
				if(random.randint(0, 1)): #是否填色
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize+gridSize*0.25), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*0.75)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline ="black", width=2)
				else:
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize+gridSize*0.25), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*0.75)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline =None, width=2)
			if(style == 5):
				if(random.randint(0, 1)): #是否填色
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize+gridSize*0.5), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*1)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline ="black", width=2)
				else:
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize+gridSize*0.5), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*1)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline =None, width=2)
			if(style == 6):
				if(random.randint(0, 1)): #是否填色
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*0.5)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline ="black", width=2)
				else:
					draw.ellipse([(x*gridSize+gridSize*0.25, y*gridSize), (x*gridSize+gridSize*0.75, y*gridSize+gridSize*0.5)] , 
								   fill = (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 
								   outline =None, width=2)


	return style
def drawText(draw, font, imgSize, gridSize, maxStrLen=13, textOrientation=random.randint(0, 2)):
	#字方位編碼
	#0:靠左 1:靠中 2:靠右
	#textOrientation = (random.randint(0, 3), random.randint(0, 3))
	textOrientation = 1
	target = np.zeros((imgSize[0], imgSize[1], 3), np.uint8)
	for y in range((imgSize[0] // gridSize)):
		for x in range((imgSize[1] // gridSize)):
			hasNoStr = random.randint(0, 10) > 8
			if(hasNoStr):
				continue
			textColor = (random.randint(0,255),random.randint(0,255),random.randint(0,255),255)  #隨機的字顏色
			strlen = random.randint(1, maxStrLen)
			drawUnderLine = random.randint(0, 10) > 8
			drawTopLine = random.randint(0, 10) > 8
			text = ''.join([strLib[x] for x in np.random.randint(len(strLib), size=strlen)])
			baseY =  y      * gridSize + 3
			baseX =  x      * gridSize + 3
			endY  = (y + 1) * gridSize - 3
			endX  = (x + 1) * gridSize - 3

			textwidth, textheight = draw.textsize(text, font=font)
			if textOrientation == 0:
				nextY = baseY
				nextX = baseX
			elif textOrientation == 1:
				nextY = (baseY + endY) // 2 - textheight//2
				nextX =  baseX
			elif textOrientation == 2:
				nextY = endY - textheight*2
				nextX = baseX



			
			for char in text:
				twidth, theight = draw.textsize(char, font=font)
				flag = False
				if char == '@':
					flag = True
					char = strLib[random.randint(0, 3)]
					nextY += theight//3
					
				if(nextX + twidth > endX):
					nextX = baseX
					nextY += textheight
					if(nextY + textheight >= endY):
						break
				draw.text((nextX, nextY), char, font=font, fill=textColor)
				if (drawUnderLine):
					if flag:
						draw.line((nextX, nextY + textheight - theight/3, nextX + twidth, nextY + textheight - theight/3), fill=textColor)
					else:
						draw.line((nextX, nextY + textheight, nextX + twidth, nextY + textheight), fill=textColor)
				if (drawTopLine):
					if flag:
						draw.line((nextX, nextY  - theight/3, nextX + twidth, nextY  - theight/3), fill=textColor)
					else:
						draw.line((nextX, nextY, nextX + twidth, nextY), fill=textColor)

				ttwidth, ttheight = draw.textsize(char, font=font)
				
				#print(nextY, nextX, textheight, ttwidth)
				# if ((not drawTopLine) & (not drawUnderLine)):
				# 	target[nextY : nextY + textheight, nextX : nextX + ttwidth,2] = 1
				# if (drawUnderLine & (not drawTopLine)):
				# 	target[nextY : nextY + textheight, nextX : nextX + ttwidth,2] = 2
				# if (drawTopLine & (not drawUnderLine)):
				# 	target[nextY : nextY + textheight, nextX : nextX + ttwidth,2] = 3
				# if (drawTopLine & drawUnderLine):
				# 	target[nextY : nextY + textheight, nextX : nextX + ttwidth,2] = 4

				target[nextY : nextY + textheight, nextX : nextX + ttwidth, :] = 255

				if flag:
					nextY -= theight//3
					nextX += ttwidth
				else:
					nextX += twidth
	return target			
def drawLine(draw, imgSize, maxLineCount=50):
	for i in range(random.randint(1, maxLineCount)):
		x1 = random.randint(0, imgSize[1])
		x2 = random.randint(0, imgSize[1])
		y1 = random.randint(0, imgSize[0])
		y2 = random.randint(0, imgSize[0])
		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255),255)  #隨機的線顏色
		draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1,5))
def drawCircle(draw, imgSize, maxCircleCount=5):
	for i in range(random.randint(1, maxCircleCount)):
		x1 = random.randint(0, imgSize[1])
		x2 = random.randint(0, imgSize[1])
		y1 = random.randint(0, imgSize[0])
		y2 = random.randint(0, imgSize[0])
		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))  #隨機的線顏色
		draw.ellipse((x1, y1, x2, y2), fill=None, outline=color, width=random.randint(1,5))
def createImg():
	imgSizeHeight = random.randint(500, 1000)  #隨機的圖片高度
	imgSizeWidth = random.randint(500, 1000)  #隨機的圖片寬度
	gridSize = random.randint(50, 300)  #網格大小

	backgroundColor = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) #背景顏色

	img = np.zeros((imgSizeHeight, imgSizeWidth, 3), np.uint8)
	img[:,:] = backgroundColor[:] #圖片背景顏色
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	font = fonts[random.randint(0, len(fonts) - 1)]

	drawGrid(draw, (imgSizeHeight, imgSizeWidth), gridSize)
	target = drawText(draw, font, (imgSizeHeight, imgSizeWidth), gridSize)
	# drawLine(draw, (imgSizeHeight, imgSizeWidth))
	# drawCircle(draw, (imgSizeHeight, imgSizeWidth))
	
	img = np.array(img_pil)
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	return img, target
def loadFonts(fontPath = "./fonts/"):
	fonts = []
	for i in os.listdir(fontPath):
		for j in range(10, 50):
			try:

				temp = ImageFont.truetype(fontPath+i, j)
			except Exception as e:
				print(i, j)
				pass
			else:
				fonts.append(temp)
	return fonts
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated
if __name__ == '__main__' : 
	pathImgStore = "./test1/storeImg/"
	pathTargetStore = "./test1/storeTarget/"
	maxImgCount = 1000
	fonts = loadFonts()
	for i in range(maxImgCount):
		print(i, "/", maxImgCount)
		img, target = createImg()
		# rotated = random.randint(0, 359)
		# img = rotate(img, rotated)
		# target = rotate(target, rotated)
		# cv2.imwrite(pathImgStore + str(i) + "_" + str(rotated) + ".png", img)
		# cv2.imwrite(pathTargetStore + str(i) + "_" +str(rotated) + ".png", target)

		cv2.imwrite(pathImgStore + str(i) + ".png", img)
		cv2.imwrite(pathTargetStore + str(i)+ ".png", target)