import cv2
import os
imgPath = "./train/storeImg/"
storePath = "./train/draw/"
for i in os.listdir(imgPath):
	aname = i.split(".")[0]
	bname = i.split(".")[-1]
	# if(bname == "txt"):
	# 	print(i)
	# 	os.rename(imgPath+i,imgPath+aname+"."+bname)
	if(bname == "png"):	
		print(i)
		with open(imgPath + aname + ".txt",'r') as fp:
			all_lines = fp.readlines()
			# print(all_lines)
		img = cv2.imread(imgPath + i)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		grayOut = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		boxsText = ""
		for j in all_lines:
			values = j.split(" ")

			centerx = float(values[1])
			centery = float(values[2])
			w = float(values[3])
			h = float(values[4])
			# centerx = " " + str(centerx)
			# centery = " " + str(centery)
			# w       = " " + str(w)
			# h       = " " + str(h)
			# boxsText += "0" + centerx + centery + w + h + "\n" 

			startx = int((centerx - w/2) * img.shape[1])
			starty = int((centery - h/2) * img.shape[0])
			endx   = int((centerx + w/2) * img.shape[1])
			endy   = int((centery + h/2) * img.shape[0])

			cv2.rectangle(grayOut, (startx, starty), (endx, endy), (0, 0, 255), 2)

		cv2.imwrite(storePath + i, grayOut)
		# fp = open(imgPath + str(i) + ".txt", "w")
		# fp.write(boxsText)
		# fp.close()
