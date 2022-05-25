import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
def extract_vertices(lines):
	'''extract vertices info from txt lines
	Input:
		lines   : list of string info
	Output:
		vertices: vertices of text regions <numpy.ndarray, (n,8)>
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
	'''
	labels = []
	vertices = []
	for line in lines:
		vertices.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
		label = 0 if '###' in line else 1
		labels.append(label)
	return np.array(vertices), np.array(labels)
def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
		# print(box)
	return img

img_flooder_path = './block/train_img/'
txt_flooder_path = './block/train_gt/'
test_flooder_path = './block/train_test/'
for img_filename in os.listdir(img_flooder_path):
	print(img_filename)
	name, ext = os.path.splitext(img_filename)
	img_path = os.path.join(img_flooder_path, img_filename)
	txt_path = os.path.join(txt_flooder_path, "gt_"+name+".txt")
	test_path = os.path.join(test_flooder_path, img_filename)

	img = Image.open(img_path)

	with open(txt_path, 'r', encoding='UTF-8') as f:
		lines = f.readlines()
	boxes = extract_vertices(lines)[0]
	plot_img = plot_boxes(img, boxes)
	plot_img.save(test_path)