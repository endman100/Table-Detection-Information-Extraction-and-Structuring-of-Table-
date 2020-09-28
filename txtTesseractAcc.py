import pytesseract
from PIL import Image, ImageFont, ImageDraw
import os
import json
import os
import csv

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
		countDistance = 0
		for fileName in fileNames:

			root, ext = os.path.splitext(fileName)
			filePath = os.path.join(path, fileName)
			if(ext == ".png"):
				image = Image.open(filePath)
				txt = pytesseract.image_to_string(image, lang='eng')
				# print(txt)
				
				countDistance += minDistance(txtName[int(root)], txt)
				if(txtName[int(root)] == txt):
					count+=1
				else:
					print(filePath, txtName[int(root)], txt)
				# print(int(root))
				# print(txtName[int(root)], txt)
		if(len(fileNames) > 0):
			
			Acc = "{:.4f}".format(count/len(fileNames)*100)
			MLD = "{:.4f}".format(countDistance/len(fileNames))
			print("{} ACC:{} ;mean Levenshtein distance: {}".format(path, Acc, MLD))
			writer.writerow([path, Acc, MLD])
