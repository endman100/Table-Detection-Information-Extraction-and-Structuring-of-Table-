from CRNNText.recognitionVersion2 import recognitionModel2
import json
import os
import csv
import cv2
import numpy as np
from PIL import Image
def grayToBinary(grayImg):
    ret, binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    whiteCount = np.sum(binaryImg[:,:] > 128)
    blackCount = np.sum(binaryImg[:,:] < 128)
    if(blackCount < whiteCount):
        ret, binaryImg = cv2.threshold(binaryImg, 128, 255, cv2.THRESH_BINARY_INV)
    return binaryImg
def minDistance(word1, word2):
    if not word1:
        return len(word2 or '') or 0

    if not word2:
        return len(word1 or '') or 0

    size1 = len(word1)
    size2 = len(word2)

    last = 0
    tmp = list(range(size2 + 1))
    value = None

    for i in range(size1):
        tmp[0] = i + 1
        last = i
        # print word1[i], last, tmp
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
                # print(last, tmp[j], tmp[j + 1], value)
            last = tmp[j+1]
            tmp[j+1] = value
        # print tmp
    return value

recognition = recognitionModel2.recognitionCRNNModel("./CRNNText/recognitionVersion2/checkpoint/cnn/netCRNN_8_3000.pth")
imgFlooder  = "./groundtruth/recognitionRotate/"

with open('result.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for path, childFlooderList, fileNames in os.walk(imgFlooder):
		# print(path)
		if(len(fileNames) > 0):
			jsonPath = os.path.join(path, "recognition.json")
			with open(jsonPath , 'r') as f:
				txtName = json.loads(f.read())
		count = 0
		correct2 = 0
		countDistance = 0
		for fileName in fileNames:
			root, ext = os.path.splitext(fileName)
			filePath = os.path.join(path, fileName)
			if(ext == ".png"):
				image = cv2.imread(filePath)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image = grayToBinary(image)
				cv2.imwrite(filePath.replace("./groundtruth/recognitionRotate/", "./groundtruth/recognitionRotateBinary/"), image)

				pil_image = Image.fromarray(image)
				txt = recognition.predict(pil_image)

				LD = minDistance(txtName[int(root)], txt)
				# print(filePath, txtName[int(root)], txt)
				countDistance+=LD
				strLen = txtName[int(root)]
				if(LD <= len(strLen)):
					correct2 += 1-LD/len(strLen)

				if(txtName[int(root)] == txt):
					count+=1
				# print(int(root))
				# print(txtName[int(root)], txt)
		if(len(fileNames) > 0):
			
			Acc = "{:.4f}".format(count/len(fileNames)*100)
			MLD = "{:.4f}".format(countDistance/len(fileNames))
			Acc2= "{:.4f}".format(correct2/len(fileNames)*100)
			print("{} ACC:{} ;ACC2:{} ;mean Levenshtein distance: {}".format(path, Acc, Acc2, MLD))
			writer.writerow([path, Acc, Acc2, MLD])
