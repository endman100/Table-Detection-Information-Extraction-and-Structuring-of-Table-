import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image, ImageColor 
import random
import os
import json
import math

import dashed
strLibs = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','_','(',')','/','#']
strLib = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

gridBorderStyle = ["line", "dashed", "null", "bold"]
def loadFonts(fontPath = "./fonts/"):
	fonts = []
	for i in os.listdir(fontPath):
		for j in range(10, 80):
			try:

				temp = ImageFont.truetype(fontPath+i, j)
			except Exception as e:
				# print(i, j)
				pass
			else:
				fonts.append(temp)
	return fonts

gridSameple = {}
def getGridSameple():
	for dirPath, dirNames, fileNames in os.walk("sampleBlock"):
		if(len(dirNames) == 0):
			gridSameple[os.path.basename(dirPath)] = []
			for fileName in fileNames:
				filePath = os.path.join(dirPath, fileName)
				gridSameple[os.path.basename(dirPath)].append(cv2.imread(filePath))
getGridSameple()

def copyGridSameple(img, position, opt):
	starth, startw, endh, endw = position
	centerh, centerw = (starth + endh) //2, (startw + endw) //2
	h, w = endh - starth, endw - startw

	copyImg = random.choice(gridSameple[opt["blockType"]])

	ch, cw, cc = copyImg.shape
	if h > w:
		copyImg = cv2.resize(copyImg, (w, ch*w//cw))
	else:
		copyImg = cv2.resize(copyImg, (cw*h//ch, h))

	ch, cw, cc = copyImg.shape
	if(opt["backgroundColor"][0] != 255):
		copyImg[copyImg <= 200] = 0 #這裡未來有問題
		copyImg[copyImg > 200] = opt["backgroundColor"][0] #這裡未來有問題
	startch = max(centerh-ch//2, 0)
	startcw = max(centerw-cw//2, 0)
	endch, endcw = startch+ch, startcw+cw
	outh, outw = 0, 0
	if(endch > img.shape[0]):
		outh = endch - img.shape[0]
		endch = endch
	if(endcw > img.shape[1]):
		outw = endcw - img.shape[1]
		endcw = endcw
	# print(ch, outh, cw, outw)
	img[startch:endch, startcw: endcw] = copyImg[0:ch-outh, 0:cw-outw]


def drawGrid(img, opt):
	for h_key in range(len(opt["gridhw"][0])-1):
		for w_key in range(len(opt["gridhw"][1])-1):
			starth, startw = opt["gridhw"][0][h_key]  , opt["gridhw"][1][w_key]
			endh  , endw   = opt["gridhw"][0][h_key+1], opt["gridhw"][1][w_key+1]
			centerh, centerw = (starth + endh) //2, (startw + endw) //2
			h, w = endh - starth, endw - startw
			radius = h / 2.3 if w > h else w / 2.3
			radius = int(radius)

			if(opt["blockType"] == "bigCircle"):
				cv2.circle(img, (centerw, centerh), radius, random.choice(opt["blockColors"]), -1)
				if(opt["sameBlockBorder"]):
					if(opt["blockborderType"] == "line"):
						cv2.circle(img, (centerw, centerh), radius, (0,0,0), radius//20)
					if(opt["blockborderType"] == "dashed"):
						dashed.drawcircle(img, (centerw, centerh), radius, (0,0,0), radius//20)
				else:
					style = random.choice(gridBorderStyle)
					if(style == "line"):
						cv2.circle(img, (centerw, centerh), radius, (0,0,0), radius//20)
					elif(style == "dashed"):
						dashed.drawcircle(img, (centerw, centerh), radius, (0,0,0), radius//20)
					elif(style == "bold"):
						cv2.circle(img, (centerw, centerh), radius, (0,0,0), radius//5)


			elif(opt["blockType"] == "smallCircle"):
				cv2.circle(img, (centerw, centerh), radius//2, random.choice(opt["blockColors"]), -1)
				if(opt["sameBlockBorder"]):
					if(opt["blockborderType"] == "line"):
						cv2.circle(img, (centerw, centerh), radius//2, (0,0,0), radius//40)
					if(opt["blockborderType"] == "dashed"):
						dashed.drawcircle(img, (centerw, centerh), radius//2, (0,0,0), radius//40)
				else:
					style = random.choice(gridBorderStyle)
					if(style == "line"):
						cv2.circle(img, (centerw, centerh), radius//2, (0,0,0), radius//40)
					elif(style == "dashed"):
						dashed.drawcircle(img, (centerw, centerh), radius//2, (0,0,0), radius//40)
					elif(style == "bold"):
						cv2.circle(img, (centerw, centerh), radius//2, (0,0,0), radius//10)
			elif(opt["blockType"] == "rect"):
				cv2.rectangle(img, (startw, starth), (endw, endh), random.choice(opt["blockColors"]), -1)
				if(opt["sameBlockBorder"]):
					cv2.rectangle(img, (startw, starth), (endw, endh), (0,0,0), radius//20)
				else:
					style = random.choice(gridBorderStyle)
					if(style == "line"):
						cv2.rectangle(img, (startw, starth), (endw, endh), (0,0,0), radius//20)
					elif(style == "dashed"):
						dashed.drawrect(img, (startw, starth), (endw, endh), (0,0,0), radius//20, style="line")
					elif(style == "bold"):
						cv2.rectangle(img, (startw, starth), (endw, endh), (0,0,0), radius//5)

					if(random.randint(0, 100) < 8):
						for i in range(startw, endw, w//20):
							for j in range(starth, endh, h//20):
								cv2.circle(img, (i, j), radius//50+1, (0,0,0), -1)
			else:
				copyGridSameple(img, (starth, startw, endh, endw), opt)
def getTextColor(opt, backgroundColor):
	while(1):
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		if(color[0] != backgroundColor[0] or color[1] != backgroundColor[1] or color[2] != backgroundColor[2]):
			break
	return color
def image_resize(image, width = None, height = None, inter = cv2.INTER_LANCZOS4):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim)
	return resized
def drawText(img, target, opt, fonts):
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	angle = 0
	if(opt["rotate"] and (not opt["mutiLine"])):
		angle = random.randint(-45, 45)


	for h_key in range(len(opt["gridhw"][0])-1):
		for w_key in range(len(opt["gridhw"][1])-1):
			starth, startw = opt["gridhw"][0][h_key], opt["gridhw"][1][w_key]
			endh  , endw   = opt["gridhw"][0][h_key+1], opt["gridhw"][1][w_key+1]
			h, w = endh - starth, endw - startw

			starth += h * opt["blockMargin"]
			startw += w * opt["blockMargin"]
			endh   -= h * opt["blockMargin"]
			endw   -= w * opt["blockMargin"]

			radius = h / 2.3 if w > h else w / 2.3
			radius = int(radius)
			if(opt["textPositionType"] == "under"):
				starth = (endh + starth)//2 + radius//2
				opt["mutiLine"] = False
			endh, starth, endw, startw = int(endh), int(starth), int(endw), int(startw)

			centerh, centerw = (starth + endh) //2, (startw + endw) //2
			h, w = endh - starth, endw - startw

			font = fonts[random.randint(0, len(fonts) - 1)]
			textColor = getTextColor(opt, img_pil.getpixel((centerw, centerh)))
			hasNoStr = random.randint(0, 10) > 9
			if(hasNoStr):
				# print("no str")
				continue

			linesCount = 1
			if(opt["mutiLine"]):
				linesCount = random.randint(1, 5)
			text = ""
			maxWidth = 0
			for i in range(linesCount):
				lineLen = random.randint(1, opt["maxTextLen"])
				if(linesCount == 1):
					temp = ''.join([strLib[x] for x in np.random.randint(len(strLib), size=lineLen)])
				else:
					temp = ''.join([strLibs[x] for x in np.random.randint(len(strLibs), size=lineLen)])
				text += temp+"\n"
				(width, baseline), (offset_x, offset_y) = font.font.getsize(temp)
				if(maxWidth < width):
					maxWidth = width
			
			ascent, descent = font.getmetrics()
			textswidth, textsheight = maxWidth, (ascent+descent)*linesCount

			# inverseTextColor = ((textColor[0]+128)%256, (textColor[1]+128)%256, (textColor[2]+128)%256)
			inverseTextColor = img_pil.getpixel((centerw, centerh))
			inverseTextColor = (inverseTextColor[2], inverseTextColor[1], inverseTextColor[0])
			tempImg = Image.new("RGB", (textswidth, textsheight), color=inverseTextColor)

			draw = ImageDraw.Draw(tempImg)
			draw.multiline_text((0, 0), text, font=font, fill=textColor, align="center", anchor="lt")
			# tempImg.show()
			# print(text+"\n"+"-"*50)
			strImg = np.array(tempImg)
			strImg = cv2.cvtColor(strImg,cv2.COLOR_RGB2BGR)

			if(opt["rotate"]):
				strImg = rotate(strImg, angle, (inverseTextColor[2], inverseTextColor[1], inverseTextColor[0]))
			elif(random.randint(0, 1000) < 5 and (not opt["mutiLine"])):
				angle = random.randint(-90, 90)
				strImg = rotate(strImg, angle, (inverseTextColor[2], inverseTextColor[1], inverseTextColor[0]))

			if(h>w):
				strImg = image_resize(strImg, height=h)
			else:
				strImg = image_resize(strImg, width=w)
			if(strImg.shape[0]>h):
				strImg = image_resize(strImg, height=h)
			if(strImg.shape[1]>w):
				strImg = image_resize(strImg, width=w)

			mask = (strImg[:,:,0] != inverseTextColor[0]) & (strImg[:,:,1] != inverseTextColor[1]) & (strImg[:,:,2] != inverseTextColor[2])
			maskh, maskw = mask.shape
			startMaskh = centerh-maskh//2
			startMaskw = centerw-maskw//2

			# p10, p11 = get_point(startMaskh, startMaskw, centerh, centerw, angle)
			# p20, p21 = get_point(startMaskh+maskh, startMaskw, centerh, centerw, angle)
			# p30, p31 = get_point(startMaskh+maskh, startMaskw+maskw, centerh, centerw, angle)
			# p40, p41 = get_point(startMaskh, startMaskw+maskw, centerh, centerw, angle)
			# points = np.array([[p11, p10], [p21, p20], [p31, p30], [p41, p40]], np.int32)
			# points = points.reshape((-1, 1, 2))
			# cv2.fillPoly(img, [points], (0, 0, 255))
			# cv2.rectangle(img, (startMaskw, startMaskh), (startMaskw+maskw, startMaskh+maskh), (0, 255, 0), 2)

			
			# cv2.putText(img, text, (startMaskw, startMaskh), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 1)
			if(len(strImg[mask]) == 0):
				print("error")
				continue
			img[startMaskh:startMaskh+maskh, startMaskw:startMaskw+maskw][mask] = strImg[mask]
			target[startMaskh:startMaskh+maskh, startMaskw:startMaskw+maskw] = 255
	return
def drawLine(img, maxCount = 15):
	lineCount = random.randint(0, maxCount)
	imgShape = img.shape
	for i in range(lineCount):
		cv2.line(img, (random.randint(0, imgShape[1]), random.randint(0, imgShape[0])), 
					  (random.randint(0, imgShape[1]), random.randint(0, imgShape[0])), 
					  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
					  (random.randint(1, 3)))
def getGrid(opt):
	h, w = opt["imgSizeHW"]
	if(opt["differentBlockSize"]):
		gridH, gridW = [], []

		count = 0
		while(count < h):
			gridH.append(count)
			count += random.randint(100, 400)
		# gridH.append(h)

		count = 0
		while(count < w):
			gridW.append(count)
			count += random.randint(100, 400)
		# gridW.append(w)
		
	else:
		gridSizeH, gridSizeW = random.randint(100, 400), random.randint(100, 400)
		gridSizeW = gridSizeH + random.randint(-gridSizeH//10, gridSizeH//10)
		gridH, gridW = [i for i in range(0, h, gridSizeH)], [i for i in range(0, w, gridSizeW)]
		# gridH.append(h)
		# gridW.append(w)
	opt["imgSizeHW"] = (gridH[-1], gridW[-1])  #隨機的圖片高與寬度
	return (tuple(gridH), tuple(gridW))
def getBlockColors():
	colorList = []
	if random.randint(0, 100) < 56:
		colorList.append((255, 255, 255))
	else:
		for i in range(random.randint(1, 17)):
			colorList.append((random.randint(0, 255),
							  random.randint(0, 255), 
							  random.randint(0, 255)))
	return tuple(colorList)
def createImg(fonts):
	opt = {}
	opt["imgSizeHW"] = (random.randint(800, 1500), random.randint(800, 1500))  #隨機的圖片高與寬度
	opt["backgroundColor"] = (255, 255, 255) if random.randint(0, 100) < 89 else (88, 88, 88) #決定背景色

	temp = random.randint(0, 100) #表格邊框樣式
	if(temp < 23):  
		opt["borderType"] = "round"
	elif(temp < 67):
		opt["borderType"] = "rect"
	else:
		opt["borderType"] = "null"

	opt["blockColors"] = getBlockColors() #決定區塊顏色類型
	opt["sameBlockBorder"] = random.randint(0, 100) < 84  #是否統一區塊邊框
	opt["blockAngle"] = random.randint(0, 100) < 23  #區塊是否有角度
	opt["blockMargin"] = random.randint(5, 20)/100 #內縮比例

	temp = random.randint(0, 100) #區塊樣式
	if(temp < 27):  
		opt["blockType"] = "bigCircle"
	elif(temp < 49):
		opt["blockType"] = "smallCircle"
	elif(temp < 90):
		opt["blockType"] = "rect"
	elif(temp < 95):
		opt["blockType"] = "octagon"
	else:
		opt["blockType"] = "irregular"
	temp = random.randint(0, 100) #區塊邊框樣式
	if(temp < 87):
		opt["blockborderType"] = "line"
	else:
		opt["blockborderType"] = "dashed"

	opt["ellipse"] = random.randint(0, 100) < 5 #是否有橢圓干擾
	opt["line"] =  random.randint(0, 100) < 5 #是否有虛線干擾
	opt["watermark"] = random.randint(0, 100) < 5#是否有浮水印干擾
	opt["differentBlockSize"] = random.randint(0, 100) < 5  #是否表格大小不一
	opt["mutiLine"] = random.randint(0, 100) < 38  #是否有多行字
	opt["differentTextColor"] = random.randint(0, 100) < 33  #是否有多行字
	opt["maxTextLen"] = random.randint(4, 15)
	opt["rotate"] = random.randint(0, 100) < 11 #是否統一角度
	if(opt["rotate"]):
		opt["mutiLine"] = False

	opt["textPositionType"] = "mid"
	if(opt["blockType"] == "smallCircle"):
		opt["textPositionType"] = textPositionType = "under" if random.randint(0, 100) < 22 else "mid"
	
	opt["gridhw"] = getGrid(opt) #取得表格框線

	opt["keyhw"] = (random.randint(0, len(opt["gridhw"][0])), random.randint(0, len(opt["gridhw"][1]))) #決定主軸
	# print(opt["keyhw"])

	img = np.zeros((opt["imgSizeHW"][0], opt["imgSizeHW"][1], 3), np.uint8)
	img[:,:] = opt["backgroundColor"][:] #圖片背景顏色

	target = np.zeros((opt["imgSizeHW"][0], opt["imgSizeHW"][1]), np.uint8)

	drawGrid(img, opt)
	# if(opt["line"]):
	# 	drawLine(img)
	# if(opt["ellipse"]):
	# 	drawEllipse(img)
	# if(opt["watermark"]):
	# 	drawWatermark(img)
	drawText(img, target, opt, fonts)
	
	return img, target
def rotate(image, angle, borderValue, center=None, scale=1.0):
	# 获取图像尺寸
	(h, w) = image.shape[:2]
	# 若未指定旋转中心，则将图像中心设为旋转中心
	if center is None:
		center = (w / 2, h / 2)
	p00, p01 = get_point(h, w, center[1], center[0], angle)
	p10, p11 = get_point(h, 0, center[1], center[0], angle)
	p20, p21 = get_point(0, w, center[1], center[0], angle)
	p30, p31 = get_point(0, 0, center[1], center[0], angle)
	startw = min(p00, p10, p20, p30)
	starth = min(p01, p11, p21, p31)
	endw = max(p00, p10, p20, p30)
	endh = max(p01, p11, p21, p31)
	temph, tempw = endh - starth, endw - startw
	temph = max(temph, h)
	tempw = max(tempw, w)
	centerh, centerw = temph //2, tempw //2

	tempImg = np.zeros((temph, tempw, 3), dtype="uint8")
	tempImg[:,:,0] = borderValue[0]
	tempImg[:,:,1] = borderValue[1]
	tempImg[:,:,2] = borderValue[2]
	# print(centerh-h//2,centerh-h//2+h, centerw-w//2,centerw-w//2+w,temph, tempw, h, w)
	tempImg[centerh-h//2:centerh-h//2+h, centerw-w//2:centerw-w//2+w] = image

	# tempImg = np.zeros((h*2, w*2, 3), dtype="uint8")
	# tempImg[:,:,0] = borderValue[0]
	# tempImg[:,:,1] = borderValue[1]
	# tempImg[:,:,2] = borderValue[2]
	# tempImg[int(h*0.5):int(h*0.5)+h, int(w*0.5):int(w*0.5)+w] = image

	# 执行旋转
	M = cv2.getRotationMatrix2D((centerw, centerh), angle, scale)
	rotated = cv2.warpAffine(tempImg, M, (tempImg.shape[1], tempImg.shape[0]), borderValue=borderValue)
	# 返回旋转后的图像
	return rotated
def get_point(x,y,cX,cY,angle):
	# angle=360-angle
	new_x = (x - cX) * math.cos(math.pi / 180.0 * angle) - (y - cY) * math.sin(math.pi / 180.0 * angle) + cX
	new_y = (x - cX) * math.sin(math.pi / 180.0 * angle) + (y - cY) * math.cos(math.pi / 180.0 * angle) + cY
	return round(new_x), round(new_y) #四捨五入取整
if __name__ == '__main__' : 
	pathImgStore = "./test/img/"
	pathTargetStore = "./test/target/"
	maxImgCount = 100
	pathImgStore = "./train/img/"
	pathTargetStore = "./train/target/"
	# maxImgCount = 100000
	fonts = loadFonts()
	for i in range(maxImgCount):
		print(i, "/", maxImgCount)
		img, target = createImg(fonts)
		cv2.imwrite(pathImgStore + str(i) + ".png", img)
		cv2.imwrite(pathTargetStore + str(i) + ".png", target)
		