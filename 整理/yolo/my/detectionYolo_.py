import sys, os, tempfile
from PIL import Image, ImageFont, ImageDraw
#Consider activate CPU detection if out of VRAM.
#os.environ['FORCE_CPU'] = 'true'
# from darknet import performDetect
import darknet

#Methods that tries to make the console cleaner.
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__


def detectText(image, cfg='./yolov4.cfg', weight='./weights/20210118/yolov4_1000.weights', metadata='./obj.data', windowSizeW=608, windowSizeH=608, strideR=360, strideD=360, produceImage=False, SaveImage=False, DoOCR=True, delUntrustable=True, delRedundant=True, delOverlay=True, confThresh=0.25, consolePrint=False):
	'''
	argument:
		image {Image|str}:     PIL的Image或同等的RGB圖片，也可以傳遞檔案路徑。
		cfg {str}:             模型的cfg設定檔路徑。
		weight {str}:          模型的權重路徑。
		metadata {str}:        模型的附加資料。
		windowSizeW {int}:     滑動窗格的窗格寬度。
			default : 608
		windowSizeH {int}:     滑動窗格的窗格高度。
			default : 608
		strideR {int}:         窗格每次向右偏移的像素量。
			default : 360
		strideD {int}:         窗格每次向下偏移的像素量。
			default : 360
		produceImage {bool}:   決定是否為框選結果或OCR結果繪製圖片。
			default : False
		SaveImage {bool}:      決定是否儲存繪製的圖片。當啟用時，必須也啟用produceImage。
			default : False
		DoOCR {bool}:          決定是否對框選結果做OCR。
			default : True
		delUntrustable {bool}: 框選框有超出滑動窗格的範圍時予以刪除。
			default : True
		delRedundant {bool} :  窗格由左至右、由上至下滑動。如果框選框的左上角的點位於被後續滑動窗格覆蓋到的區域，則刪除。
			default : True
		delOverlay {bool} :    如果框選框重疊，擇信心分數較低的刪除。
			default : True
		confThresh {int}:      設定信心分數的閥值。
			default : 0.25
		consolePrint {bool}:   是否印出偵測信息。
			default : False
	return:
		{
            'info' : [偵測到了幾個框, 淘汰過後剩幾個框, 有多少個框有被OCR偵測到文字],
            'confidence' : [a, b, c, .....],
            'boxes' : [[左上角的x, 左上角的y, 右下角的x, 右下角的y], [], [], .......],
            'strings' : [texta, textb, textc, .....], #如果DoOCR=False就不會有
            'PIL_RGB_BOX' : imgBox,    #如果produceImage=False就不會有
            'PIL_RGB_OCR' : imgOCR     #如果produceImage=False或DoOCR=False就不會有
        }
	'''
	#File input.
	if isinstance(image, str):
		img = Image.open(image).convert(mode = 'RGB')
		
		filenameBox = os.path.basename(image).split('.')[0] + '_BOX.png'
		filenameOCR = os.path.basename(image).split('.')[0] + '_OCR.png'
	else:
		img = image.copy().convert(mode = 'RGB')
		filenameBox = 'result_BOX.png'
		filenameOCR = 'result_OCR.png'

	if SaveImage:
		produceImage = True

	#Initial value for every cycle.
	scale = 1
	scaled = False
	wordsDetected = 0
	wordsSelected = 0
	boxes = []
	confidence = []

	#Resize if any side of img smaller than windowSize.
	if img.width < windowSizeW or img.height < windowSizeH:
		scaleW, scaleH = windowSizeW / img.width, windowSizeH / img.height
		scale = scaleW if scaleW > scaleH else scaleH
		scaled = True
		original_img = img.copy()
		img = img.resize((round(img.width * scale), round(img.height * scale)))

	#Determine the position of every sliding window.
	lastStrideW = img.width - windowSizeW
	lastStrideH = img.height - windowSizeH

	stepsRow = [x for x in range(0, img.height - windowSizeH, strideD)]
	stepsRow.append(lastStrideH)
	stepsRow.append(img.height)
	stepsColumn = [x for x in range(0, img.width - windowSizeW, strideR)]
	stepsColumn.append(lastStrideW)
	stepsColumn.append(img.width)

	#Sliding window.
	#blockPrint()
	for row, y in enumerate(stepsRow):
		if y == stepsRow[-1]:
			break
		for column, x in enumerate(stepsColumn):
			if x == stepsColumn[-1]:
				break
			trash = []
			tmp = img.crop((x, y, x + windowSizeW, y + windowSizeH))
			tmp.save(tempfile.gettempdir() + '\\tmp.png')
			result = performDetect(imagePath = tempfile.gettempdir() + '\\tmp.png',
				                   thresh = confThresh,
				                   configPath = cfg,
				                   weightPath = weight,
				                   metaPath = metadata,
				                   showImage = False,
				                   makeImageOnly = True)
			#print(result)
			box = [list(x[2]) for x in result]
			
			conf = [x[1] for x in result]
			wordsDetected += len(box)
			for ind in range(len(box)):
				box[ind][0] = box[ind][0] + x - box[ind][2] / 2
				box[ind][1] = box[ind][1] + y - box[ind][3] / 2
				box[ind][2] = box[ind][0] + box[ind][2]
				box[ind][3] = box[ind][1] + box[ind][3]
				#1
				if delUntrustable and\
				   (box[ind][0] < x or\
				    box[ind][1] < y or\
				    box[ind][2] > x + windowSizeW or\
				    box[ind][3] > y + windowSizeH):
					trash.append(ind)
					continue
				#2
				if delRedundant and\
				   (box[ind][0] >= stepsColumn[column + 1] or\
				    box[ind][1] >= stepsRow[row + 1]):
					trash.append(ind)
					continue
				box[ind] = [int(x / scale) for x in box[ind]]
			for ind in range(len(trash)-1, -1, -1):
				del box[trash[ind]], conf[trash[ind]]
			boxes += box
			confidence += conf
			wordsSelected += len(box)
	#enablePrint()
	#3
	if delOverlay:
		trash.clear()
		for ind1, box1 in enumerate(boxes[:-1]):
			for ind2, box2 in enumerate(boxes[ind1 + 1:], start = ind1 + 1):
				if (box1[2] > box2[0]) and\
				   (box2[2] > box1[0]) and\
				   (box1[3] > box2[1]) and\
				   (box2[3] > box1[1]):
					ind = ind1 if confidence[ind1] < confidence[ind2] else ind2
					if ind not in trash:
						trash.append(ind)
		trash.sort()
		for ind in range(len(trash)-1, -1, -1):
			del boxes[trash[ind]], confidence[trash[ind]]
		wordsSelected -= len(trash)

	info = [wordsDetected, wordsSelected]

	if consolePrint:
		print(wordsDetected, 'strings Detected.')
		print(wordsSelected, 'strings Selected.')

	if scaled:
		img = original_img

	#Open new file and draw result.
	if produceImage and DoOCR:
		resultImg = Image.new('RGB', img.size, (255, 255, 255))
		draw = ImageDraw.Draw(resultImg)
		font = ImageFont.truetype('./font/calibri.ttf', 20)
	#os.environ['path'] = r'tessreact.exe path' + os.environ['path']
	OCRNumber = 0
	strings = []
	if DoOCR:
		for rect in boxes:
			tmp = img.crop(tuple(rect)).convert(mode = 'L')
			text = pytesseract.image_to_string(tmp, lang = 'eng')
			strings.append(text)
			if produceImage:
				draw.text((rect[0], rect[1]), text, align='center', font = font, fill = (0, 0, 0, 255))
			OCRNumber += 1 if len(text) > 0 else 0
		if consolePrint:
			print(OCRNumber, "strings identified.")
		info.append(OCRNumber)
	if produceImage:
		draw = ImageDraw.Draw(img)
		for rect in boxes:
			draw.rectangle(tuple(rect), fill = None, outline = (0, 0, 0), width = 2)
	if SaveImage:
		resultImg.save(filenameOCR)
		if consolePrint:
			print(filenameOCR, 'saved.')
		img.save(filenameBox)
		if consolePrint:
			print(filenameBox, 'saved.')

	returnDict = {'info' : info, 'confidence' : confidence, 'boxes' : boxes, 'strings' : strings}
	if produceImage:
		returnDict['PIL_RGB_BOX'] = img
		if DoOCR:
			returnDict['PIL_RGB_OCR'] = resultImg
	return returnDict

if __name__ == "__main__":
	#while True:
	#filename = input('>>> ').replace('\"', '')
	filename =b"../../data/yoloGtold1/train/1.png"
	# print(filename)
	# result = detectText(image=filename, produceImage=True, SaveImage=False, consolePrint=False, DoOCR=False)
	# print(result)
	# result['PIL_RGB_BOX'].save("result2.jpg")
	net = darknet.load_net(b'./yolov4.cfg', b'./weights/20210118/yolov4_1000.weights', 0)
	meta = darknet.load_meta(b'./obj.data')
	r = darknet.detect(net, meta, filename)
	print(r)