import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import json

import detect
import lanms
from model import EAST
def plot_result_boxes(boxes, imgShape, img=None):
	if(type(img) == type(None)):
		img = Image.new(mode = "RGB", size = [imgShape[1], imgShape[0]])
	else:
		img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	if(type(img) == type(None)):
		for box in boxes:
			draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], fill ="#ffffff", outline=(255,255,255))
	else:	
		for box in boxes:
			# draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(255,0,0))
			draw.line([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], box[0], box[1]], fill="red", width=5)
	return img
def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score	   : score map from model <numpy.ndarray, (1,row,col)>
		geo		 : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes	   : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = detect.restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	# boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes
def cutImage(img, sliceSize=1000):
	y = img.shape[0]
	x = img.shape[1]
	if(x<sliceSize or y<sliceSize):
		imgs = [[img[:], 0, 0]]
	else:
		imgs = [[img[sliceSize*i: sliceSize*(i+1) if sliceSize*(i+2)<y else y,
					 sliceSize*j: sliceSize*(j+1) if sliceSize*(j+2)<x else x,:], sliceSize*i, sliceSize*j]
					for i in range(y//sliceSize)
					for j in range(x//sliceSize)]
	return imgs
def detectMy(img, model, device, score_thresh=0.9):
	img, ratio_h, ratio_w = detect.resize_img(img)
	with torch.no_grad():
		score, geo = model(detect.load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy(), score_thresh=score_thresh)
	return detect.adjust_ratio(boxes, ratio_w, ratio_h)
def predict(img, nms_thresh=0.32, score_thresh=0.9):
	imgs = cutImage(img)
	boxesAll = []
	for im in imgs:
		# print(im[0].shape, im[1:])
		boxes = detectMy(Image.fromarray(cv2.cvtColor(im[0], cv2.COLOR_BGR2RGB)), model, device, score_thresh=score_thresh)
		for box in boxes:
			box[:8:2] += im[2]
			box[1::2] += im[1]
			boxesAll.append(np.expand_dims(box, axis=0))
	boxesAll = np.concatenate(boxesAll, axis=0)
	boxesAll = lanms.merge_quadrangle_n9(boxesAll.astype('float32'), nms_thresh)
	return boxesAll
def testModel(img_flooder_path, save_flooder_path, model_path, score_thresh=0.9):
	model.load_state_dict(torch.load(model_path))
	model.eval()

	for img_filename in os.listdir(img_flooder_path):
		print(img_filename)
		name, ext = os.path.splitext(img_filename)
		img_path = os.path.join(img_flooder_path, img_filename)
		json_save_path = os.path.join(save_flooder_path, name+".json")
		img_save_path = os.path.join(save_flooder_path, name+".jpg")
		if not os.path.exists(save_flooder_path):
			os.makedirs(save_flooder_path) 
		img = cv2.imread(img_path)

		boxes = predict(img, nms_thresh=0.32, score_thresh=score_thresh)
		rects = []
		for box in boxes:
			# print(box[-1])
			temp   = box[:-1]
			scores = box[-1]
			x = temp[::2]
			y = temp[1::2]
			x1, x2, y1, y2 = min(x).item(), max(x).item(), min(y).item(), max(y).item()
			rects.append([x1, y1, x2, y2, scores.item()])
		# print(rects)

		# rects = delSmallerThan8X8(rects)
		# rects = delUnreasonableRects(rects, ratio=6)
		# rects = NMS(rects)
		# rects = delLowScores(rects)
		# rects = delInclude(rects)


		rectsTemp = []
		for box in rects:
			# rectsTemp.append(box.tolist())
			rectsTemp.append(box)
		with open(json_save_path, 'w') as f:
			json.dump(rectsTemp, f)
		# with open(txt_save_path, 'w') as out_file:
		# for box in boxes:
		# 	print(box[:-1].tolist())
		# 	out_file.write(" ".join([str(num) for num in box[:-1]])+"\n")

		# print(boxes.tolist())
		plot_img = plot_result_boxes(boxes, img.shape, img=img)	
		plot_img.save(img_save_path)
def testModelPoly(img_flooder_path, save_flooder_path, model_path, score_thresh=0.9):
	model.load_state_dict(torch.load(model_path))
	model.eval()

	for img_filename in os.listdir(img_flooder_path):
		print(img_filename)
		name, ext = os.path.splitext(img_filename)
		img_path = os.path.join(img_flooder_path, img_filename)
		json_save_path = os.path.join(save_flooder_path, name+".json")
		img = cv2.imread(img_path)

		boxes = predict(img, nms_thresh=0.32, score_thresh=score_thresh)

		# print(boxes.tolist())
		img_save_path = os.path.join(save_flooder_path, img_filename+".png")
		plot_img = plot_result_boxes(boxes, img.shape, img=img)	
		plot_img.save(img_save_path)

def testModels(img_flooder_path, save_flooder_path, model_flooder_path):
	for model_filename in os.listdir(model_flooder_path):
		print(model_filename)
		model_path = os.path.join(model_flooder_path, model_filename)
		model.load_state_dict(torch.load(model_path))
		model.eval()

		save_path = os.path.join(save_flooder_path, model_filename)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		for img_filename in os.listdir(img_flooder_path):
			name, ext = os.path.splitext(img_filename)
			img_path = os.path.join(img_flooder_path, img_filename)
			img_save_path = os.path.join(save_path, name+".txt")
			img = cv2.imread(img_path)

			boxes = predict(img)

			print(boxes)
			# plot_img = plot_result_boxes(boxes, img.shape)	
			# plot_img.save(img_save_path)

def drawCounters(counters, img, color=(0, 0, 255)):
	for counter in counters:
		cv2.rectangle(img, (counter[1], counter[0]), (counter[3], counter[2]), color, 4)
	return img
def drawFillCounters(counters, img):
	for counter in counters:
		cv2.rectangle(img, (counter[1], counter[0]), (counter[3], counter[2]), (255, 255, 255), -1)
	return img
def yoloBoxToCv2(detection, shape):
	height = shape[0]
	width = shape[1]
	center_x = int(detection[1] * width)
	center_y = int(detection[2] * height)
	w = int(detection[3] * width)
	h = int(detection[4] * height)
	x = int(center_x - w / 2)
	y = int(center_y - h / 2)
	return [y, x, y+h, x+w]
def testMask(preImage, gtImage):
	preImage = preImage[:,:,0].reshape((-1))//255
	gtImage = gtImage[:,:,0].reshape((-1))//255

	preImage_torch = torch.from_numpy(preImage).cuda()
	gtImage_torch  = torch.from_numpy(gtImage).cuda()
	
	TP = torch.sum((gtImage_torch == 1) & (preImage_torch == 1)).item() #gpu version
	FN = torch.sum((gtImage_torch == 0) & (preImage_torch == 1)).item()
	FP = torch.sum((gtImage_torch == 1) & (preImage_torch == 0)).item()
	TN = torch.sum((gtImage_torch == 0) & (preImage_torch == 0)).item()

	# accuracy = confusion_matrix(preImage, gtImage, labels=[1, 0]) #cpu version
	# TP = accuracy[0,0]
	# FN = accuracy[0,1]
	# FP = accuracy[1,0]
	# TN = accuracy[1,1]
	# print(accuracy)
	P = TP/(TP+FP)
	R = TP/(TP+FN)
	F1 = 2*P*R/(P+R)

	Acc = (TP+TN)/(TP+TN+FP+FN)
	print("Accuracy =", Acc)
	print("Precision=", P)
	print("Recall   =", R)
	print("F1		=", F1)

	Acc = "{:.4f}".format(Acc*100)
	P = "{:.4f}".format(P*100)
	R = "{:.4f}".format(R*100)
	F1 = "{:.4f}".format(F1*100)
	writer.writerow([Acc, P, R, F1])
def countAcc(EASTImgFlooders, GTFlooders):
	for GTFlooderName in os.listdir(GTFlooders):
		print(GTFlooderName)
		GTPath = os.path.join(GTFlooders, GTFlooderName)

		ImgPath = os.path.join(GTPath, GTFlooderName + ".jpg")
		GTTxtPath = os.path.join(GTPath, GTFlooderName + ".txt")
		classNamePath = os.path.join(GTPath, "classes.txt")

		UnetImgPath = os.path.join(EASTImgFlooders, GTFlooderName + ".png")

		
		image = cv2.imread(ImgPath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		print(UnetImgPath)
		imagePreMask = cv2.imread(UnetImgPath)#讀取預測結果
		# print(imagePreMask)

		with open(GTTxtPath,'r') as f:  #讀取真實結果
			fStrs = f.readlines()
			fStrs = [fStr.replace("\n", "") for fStr in fStrs]
			yoloRects = [[float(i) for i in fStr.split(" ")] for fStr in fStrs]
			cv2Rects = [yoloBoxToCv2(i, image.shape) for i in yoloRects]
		# image = drawCounters(cv2Rects, image, color=(0, 0, 255))
		imageMaskGt = np.zeros(image.shape)
		imageMaskGt = drawFillCounters(cv2Rects, imageMaskGt)
		
		testMask(imagePreMask, imageMaskGt)
def NMS(dets, thresh=0.5): 
	x1, y1, x2, y2, scores = [], [], [], [], []
	for i in dets:
		x1.append(i[0])
		y1.append(i[1])
		x2.append(i[2])
		y2.append(i[3])
		scores.append(i[4])
	x1, y1, x2, y2, scores = np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores)
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

	#打分从大到小排列，取index  
	order = scores.argsort()[::-1]  
	#keep为最后保留的边框  
	keep = []  
	while order.size > 0: 
		i = order[0]  
		keep.append(i)  
		#计算窗口i与其他所有窗口的交叠部分的面积
		xx1 = np.maximum(x1[i], x1[order[1:]])  
		yy1 = np.maximum(y1[i], y1[order[1:]])  
		xx2 = np.minimum(x2[i], x2[order[1:]])  
		yy2 = np.minimum(y2[i], y2[order[1:]])  
  
		w = np.maximum(0.0, xx2 - xx1 + 1)  
		h = np.maximum(0.0, yy2 - yy1 + 1)  
		inter = w * h  
		#交/并得到iou值  
		ovr = inter / (areas[i] + areas[order[1:]] - inter)  
		#inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
		inds = np.where(ovr <= thresh)[0]  
		#order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
		order = order[inds + 1]

	return [dets[i] for i in keep]
def delLowScores(rects, thresh=0.4):
	returnRects = []
	for rect in rects:
		if(rect[4] > thresh):
			returnRects.append(rect)
	return returnRects
def delInclude(rects, includeThreshold=0.9):
	rectSum = []

	for i in range(len(rects)):
		if(rects[i][0] == -1):
			continue
		for j in range(i + 1, len(rects)):
			# print(rects[i], rects[j], j)
			if(rects[j][0] == -1):
				continue
			IOU, intersect, aArea, bArea = countIOU(rects[i], rects[j])
			if(aArea <= intersect and aArea / intersect > includeThreshold):
				j = -1
				break
			elif(bArea <= intersect and bArea / intersect > includeThreshold):
				rects[j][0] = -1
		if(j != -1):
			rectSum.append(rects[i])
	return rectSum
def countIOU(rec1, rec2):
	S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
	S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
	sum_area = S_rec1 + S_rec2

	left_line = max(rec1[1], rec2[1])
	right_line = min(rec1[3], rec2[3])
	top_line = max(rec1[0], rec2[0])
	bottom_line = min(rec1[2], rec2[2])
	if left_line >= right_line or top_line >= bottom_line:
		return 0, -1, S_rec1, S_rec2
	else:
		intersect = (right_line - left_line) * (bottom_line - top_line)
		return (intersect / (sum_area - intersect))*1.0, intersect, S_rec1, S_rec2
def delSmallerThan8X8(rects):
	returnRects = []
	for rect in rects:
		h = (rect[2]-rect[0])
		w = (rect[3]-rect[1])
		if(h >= 8 and w >= 8):
			returnRects.append(rect)
	return returnRects
def delUnreasonableRects(rects, ratio=4):
	# print(type(rects))
	hList = []
	wList = []
	for rect in rects:
		h = (rect[2]-rect[0])//2
		w = (rect[3]-rect[1])//2

		hList.append(h)
		wList.append(w)

	# hList = np.array(hList)
	# wList = np.array(wList)
	hStd = np.std(hList, ddof=0)
	wStd = np.std(wList, ddof=0)
	hMean = np.mean(hList)
	wMean = np.mean(wList)
	# print("hStd: {:.4f} wStd: {:.4f}".format(hStd, wStd))
	# print("hMean:{:.4f} wMean:{:.4f}".format(hMean, wMean))
	returnRects = []
	for i, (h, w) in enumerate(zip(hList, wList)):
		if(h < hMean + hStd*ratio and h > hMean - hStd*ratio and w < wMean + wStd*ratio and w > wMean - wStd*ratio):
			returnRects.append(rects[i])
	if(len(returnRects) < len(rects)):
		return delUnreasonableRects(returnRects)
	return returnRects


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EAST().to(device)
if __name__ == '__main__':
	# img_flooder_path = '../../groundtruth/image/'
	# save_flooder_path = '../EAST-Result-pretrain/'
	# model_path  = './pths/east_vgg16.pth'
	# testModel(img_flooder_path, save_flooder_path, model_path)

	# img_flooder_path = '../../groundtruth/image/'
	# save_flooder_path = '../EAST-Result_retrain/'
	# model_flooder_path  = './pths/Create/'
	# testModels(img_flooder_path, save_flooder_path, model_flooder_path)

	img_flooder_path = '../../groundtruth/image/'
	save_flooder_path = '../EAST-Result/'
	model_path  = './pths/CreateBlock/model_epoch_8.pth'
	testModel(img_flooder_path, save_flooder_path, model_path)


	# EASTImgFlooders = '../EAST-Result/'
	# GTFlooders   = "../../groundtruth/marge"
	# with open('result.csv', 'w', newline='') as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	countAcc(EASTImgFlooders, GTFlooders)

		
	

	


