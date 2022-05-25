import numpy as np
from openpyxl import Workbook
def toXls(xGrid, yGrid, rect, pred, storeName):
	ansList = []
	for j in range(len(yGrid)):
		ansList.append([])
		for k in range(len(xGrid)):
			ansList[-1].append("-")
	for j in range(len(rect)):
		imgCenterX = (rect[j][0] + rect[j][2])//2
		imgCenterY = (rect[j][1] + rect[j][3])//2
		targetX = np.argmin(np.absolute(xGrid-imgCenterY), axis=0)
		targetY = np.argmin(np.absolute(yGrid-imgCenterX), axis=0)
		if(ansList[targetY][targetX] != "-"):
			ansList[targetY][targetX] += "-"+pred[j].upper()
		else:
			ansList[targetY][targetX] = pred[j].upper()
	#print(ansList)
	ansListTemp = []
	for j in range(len(ansList)):
		flag = 0
		for k in range(len(ansList[j])):
			if(ansList[j][k] != "-"):
				flag += 1
		if(flag >= 2):
			ansListTemp.append(ansList[j][:])
	ansList = ansListTemp[:]
	j = 0
	#print(ansList)
	while(j < len(ansList[0])):
		flag = 0
		for k in range(len(ansList)):
			if(ansList[k][j] != "-"):
				flag += 1
		if(flag < 2):
			for k in range(len(ansList)):
				del ansList[k][j]
		else:
			j+=1
	#print(ansList)

	xKey = -1
	xKeyMin = 1000
	for j in range(len(ansList)):
		counter = 0
		flag = 0
		for k in range(len(ansList[j])):
			counter += len(ansList[j][k])
			if(len(ansList[j][k]) > 5):
				flag = 1
				break
		if(counter < xKeyMin and flag == 0):
			xKeyMin = counter
			xKey = j
	yKey = -1
	yKeyMin = 1000
	for j in range(len(ansList[0])):
		counter = 0
		flag = 0
		for k in range(len(ansList)):
			counter += len(ansList[k][j])
			if(len(ansList[k][j]) > 5):
				flag = 1
				break
		if(counter < yKeyMin and flag == 0):
			yKeyMin = counter
			yKey = j

	#print(xKey, yKey)
	wb = Workbook()
	ws = wb.active
	ws.append(["NUMBER", "CONTENT"])
	for j in range(len(ansList)):
		if(j == xKey):
			continue
		for k in range(len(ansList[j])):
			if(k == yKey):
				continue
			ws.append([ansList[j][yKey]+ansList[xKey][k], ansList[j][k]])
	wb.save(storeName+'.xlsx')