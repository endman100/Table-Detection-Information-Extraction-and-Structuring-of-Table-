import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw

path = './train/'
for img_filename in os.listdir(path):	
	name, ext = os.path.splitext(img_filename)
	print(img_filename, ext)
	if(ext != ".png"):
		continue
	img_path = os.path.join(path, img_filename)
	txt_path = os.path.join(path, name+".txt")


	img = cv2.imread(img_path)

	with open(txt_path, 'r', encoding='UTF-8') as f:
		lines = f.readlines()
		lines = [line.split(" ") for line in lines]

	h, w, c = img.shape
	for line in lines:
		_, centerw, centerh, boxw, boxh = line
		centerh, centerw, boxh, boxw = float(centerh)*h, float(centerw)*w, float(boxh)*h, float(boxw)*w
		centerh, centerw, boxh, boxw = int(centerh), int(centerw), int(boxh), int(boxw)

		starth, startw = centerh - boxh, centerw - boxw
		boxh, boxw = centerh + boxh, centerw + boxw

		cv2.rectangle(img, (startw, starth), (boxw, boxh), (0, 0, 255), 2)
	cv2.imwrite("./show/" + name + ".png", img)


