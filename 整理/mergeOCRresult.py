import os
import cv2
import json

def drawRects(image, Rects, color=(0, 0, 255)):
    for rect in Rects:
        # print(rect)
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), color, 3)
        cv2.putText(image, rect[-1], (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

def rectFind(rect, rects):
	for key, value in enumerate(rects):
		if(value[0] == rect[0] and value[1] == rect[1] and value[2] == rect[2] and value[3] == rect[3]):
			return key
	return None
def vote(results, saveDir):
	if not os.path.exists(saveDir):
		os.makedirs(saveDir)
	imageRects = [[] for i in range(18)]
	for result in results:
		for filename in os.listdir(result):
			name, ext = os.path.splitext(filename)
			num = int(name.split("_")[1])-1
			if(ext != ".json"):
				continue
			filepath = os.path.join(result, filename)
			with open(filepath, 'r') as f:
				rects = json.load(f)
				imageRects[num].append(rects)
	
	for imageNumber, rects18 in enumerate(imageRects):
		margeRects = []
		for i in range(len(results)):
			for rect in rects18[i]:
				margeRects.append(rect[:])
				margeRects[-1][-1] = [margeRects[-1][-1].upper()]
				for j in range(i+1, len(results)):
					# print(len(rects18), j,)
					key = rectFind(rect, rects18[j])
					if(key != None):
						margeRects[-1][-1].append(rects18[j][key][-1].upper())
						del rects18[j][key]
					else:
						# margeRects[-1][-1].append("")
						# print(key)
						del margeRects[-1]
						break
					
		for rect in margeRects:
			strVote = {}
			for strA in rect[-1]:
				if(strA not in strVote):
					strVote[strA] = 0
				strVote[strA] += 1
			rect[-1] = max(strVote, key=strVote.get)

		imageName = "FPK_"+str(imageNumber+1).zfill(2)
		image = cv2.imread(os.path.join("../groundtruth/image/", imageName+".jpg"))
		imgSavePath = os.path.join(saveDir, imageName+".jpg")
		drawRects(image, margeRects, color=(255, 0, 0))
		print(imageName)
		cv2.imwrite(imgSavePath, image)
		with open(os.path.join(saveDir, imageName+".json"), 'w') as f:
			json.dump(margeRects, f)
# results = [
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.RARE_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.CRNN_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.STARNet_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.Rosetta_RGB",
# ]
# results = vote(results, "./CrossTest-Result/yolo-Result-best-rulebase_vote_RGB/")

# results = [
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.RARE_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.CRNN_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.STARNet_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.Rosetta_RGB",
# ]
# results = vote(results, "./CrossTest-Result/unet-Result-best-rulebase_vote_RGB/")

# results = [
# "./CrossTest-Result/marge_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/marge_dataColor.RARE_RGB",
# "./CrossTest-Result/marge_dataColor.CRNN_RGB",
# "./CrossTest-Result/marge_dataColor.STARNet_RGB",
# "./CrossTest-Result/marge_dataColor.Rosetta_RGB",
# ]
# results = vote(results, "./CrossTest-Result/marge_vote_RGB/")
# results = [
# "./CrossTest-Result/marge_CRNN_BINARY",
# "./CrossTest-Result/marge_CRNN_RGB",
# "./CrossTest-Result/marge_CRNN_TYPE1",
# "./CrossTest-Result/marge_dataColor.CRNN_RGB",
# "./CrossTest-Result/marge_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/marge_dataColor.RARE_RGB",
# "./CrossTest-Result/marge_dataColor.Rosetta_RGB",
# "./CrossTest-Result/marge_dataColor.STARNet_RGB",
# "./CrossTest-Result/marge_RARE_BINARY",
# "./CrossTest-Result/marge_RARE_RGB",
# "./CrossTest-Result/marge_RARE_TYPE1",
# "./CrossTest-Result/marge_Rosetta_BINARY",
# "./CrossTest-Result/marge_Rosetta_RGB",
# "./CrossTest-Result/marge_Rosetta_TYPE1",
# "./CrossTest-Result/marge_STARNet_BINARY",
# "./CrossTest-Result/marge_STARNet_RGB",
# "./CrossTest-Result/marge_STARNet_TYPE1",
# ]
# vote(results, "./CrossTest-Result/marge_vote2_RGB/")

# results = [
# "./CrossTest-Result/unet-Result-best-rulebase_CRNN_BINARY",
# "./CrossTest-Result/unet-Result-best-rulebase_CRNN_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_CRNN_TYPE1",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.CRNN_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.RARE_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.Rosetta_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_dataColor.STARNet_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_RARE_BINARY",
# "./CrossTest-Result/unet-Result-best-rulebase_RARE_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_RARE_TYPE1",
# "./CrossTest-Result/unet-Result-best-rulebase_Rosetta_BINARY",
# "./CrossTest-Result/unet-Result-best-rulebase_Rosetta_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_Rosetta_TYPE1",
# "./CrossTest-Result/unet-Result-best-rulebase_STARNet_BINARY",
# "./CrossTest-Result/unet-Result-best-rulebase_STARNet_RGB",
# "./CrossTest-Result/unet-Result-best-rulebase_STARNet_TYPE1",
# ]
# vote(results, "./CrossTest-Result/unet-Result-best-rulebase_vote2_RGB/")

# results = [
# "./CrossTest-Result/yolo-Result-best-rulebase_CRNN_BINARY",
# "./CrossTest-Result/yolo-Result-best-rulebase_CRNN_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_CRNN_TYPE1",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.CRNN_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.NoneResNetBiLSTMCTC_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.RARE_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.Rosetta_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_dataColor.STARNet_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_RARE_BINARY",
# "./CrossTest-Result/yolo-Result-best-rulebase_RARE_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_RARE_TYPE1",
# "./CrossTest-Result/yolo-Result-best-rulebase_Rosetta_BINARY",
# "./CrossTest-Result/yolo-Result-best-rulebase_Rosetta_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_Rosetta_TYPE1",
# "./CrossTest-Result/yolo-Result-best-rulebase_STARNet_BINARY",
# "./CrossTest-Result/yolo-Result-best-rulebase_STARNet_RGB",
# "./CrossTest-Result/yolo-Result-best-rulebase_STARNet_TYPE1",
# ]
# vote(results, "./CrossTest-Result/yolo-Result-best-rulebase_vote2_RGB/")