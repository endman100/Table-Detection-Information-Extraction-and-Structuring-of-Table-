import numpy as np
def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[0]) 
    return sub_li 
def toArray(rects):#[[y1, x1, y2, x2, score, angle, block, txt, xtarget, ytarget]]
	maxx, maxy = 0, 0
	for rect in rects:
		if(rect[8] > maxx): maxx = rect[8]
		if(rect[9] > maxy): maxy = rect[9]
	maxx += 1
	maxy += 1
	returnArray = [] #初始化
	for i in range(maxx):
		returnArray.append([])
		for j in range(maxy):
			returnArray[i].append([])

	for rect in rects:
		returnArray[rect[8]][rect[9]].append([rect[1], rect[7]])

	returnStrs = [] #初始化
	for i in range(maxx):
		returnStrs.append([])
		for j in range(maxy):
			returnStrs[i].append(None)

	for xkey, xvalue in enumerate(returnArray):
		for ykey, yvalue in enumerate(xvalue):
			for i in Sort(yvalue):#根據高度做排序
				if(returnStrs[xkey][ykey] == None):
					returnStrs[xkey][ykey] = i[1]
				else:
					returnStrs[xkey][ykey] += "\n"+i[1]
	# print(returnStrs)
	return returnStrs