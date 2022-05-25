import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
from . import darknet
# import darknet

class YoloPredict(object):
	"""docstring for YoloPredict"""
	def __init__(self, batch_size=1, weights="./weights/20210121yolov4/yolov4_3000.weights", config_file="./yolov4.cfg", data_file="./obj.data", thresh=0.25):
		super(YoloPredict, self).__init__()
		self.input = input
		self.batch_size = batch_size
		self.weights = weights
		self.config_file = config_file
		self.data_file = data_file
		self.thresh = thresh
		self.check_arguments_errors()
		self.network, self.class_names, self.class_colors = darknet.load_network(
			self.config_file,
			self.data_file,
			self.weights,
			batch_size=self.batch_size
		)
	def check_arguments_errors(self):
		assert 0 < self.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
		if not os.path.exists(self.config_file):
			raise(ValueError("Invalid config path {}".format(os.path.abspath(self.config_file))))
		if not os.path.exists(self.weights):
			raise(ValueError("Invalid weight path {}".format(os.path.abspath(self.weights))))
		if not os.path.exists(self.data_file):
			raise(ValueError("Invalid data file path {}".format(os.path.abspath(self.data_file))))
	def load_images(self, images_path):
		"""
		If image path is given, return it directly
		For txt file, read it and return each line as image path
		In other case, it's a folder, return a list with names of each
		jpg, jpeg and png file
		"""
		input_path_extension = images_path.split('.')[-1]
		if input_path_extension in ['jpg', 'jpeg', 'png']:
			return [images_path]
		elif input_path_extension == "txt":
			with open(images_path, "r") as f:
				return f.read().splitlines()
		else:
			return glob.glob(
				os.path.join(images_path, "*.jpg")) + \
				glob.glob(os.path.join(images_path, "*.png")) + \
				glob.glob(os.path.join(images_path, "*.jpeg"))
	def image_detection_cut(self, image_path):
		# Darknet doesn't accept numpy images.
		# Create one with image we reuse for each detect
		width = darknet.network_width(self.network)
		height = darknet.network_height(self.network)
		darknet_image = darknet.make_image(width, height, 3)

		image = cv2.imread(image_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images_rgb = self.cutImage(image_rgb, sliceSize=width)

		detections = []
		for i, image_rgb in enumerate(images_rgb):
			image_resized = cv2.resize(image_rgb[0], (width, height),
									interpolation=cv2.INTER_LINEAR)
			darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
			detectionsT = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
			detectionsT = [(detection[0], float(detection[1]), (detection[2][0], detection[2][1], detection[2][2], detection[2][3])) for detection in detectionsT]
			# imageT = self.draw_boxes(detectionsT, image_resized, (255, 0, 0))
			# cv2.imwrite(str(i)+'output.jpg', image_resized)

			detectionsT = [(detection[0], detection[1], (detection[2][0]+image_rgb[2], detection[2][1]+image_rgb[1], detection[2][2], detection[2][3])) for detection in detectionsT]
			detections+=detectionsT
		detections = self.NMS(detections)
		image = self.draw_boxes(detections, image, (255, 0, 0))
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
	def image_detection(self, image_path):
		width = darknet.network_width(self.network)
		height = darknet.network_height(self.network)
		darknet_image = darknet.make_image(width, height, 3)

		image = cv2.imread(image_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image_resized = cv2.resize(image_rgb, (width, height),
								interpolation=cv2.INTER_LINEAR)
		darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
		detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)

		ori_image_height, ori_image_width, _ = image.shape
		scale_height, scale_width = ori_image_height/height, ori_image_width/width
		detections = [(detection[0], detection[1]
					, (detection[2][0]*scale_width, detection[2][1]*scale_height
					 , detection[2][2]*scale_width, detection[2][3]*scale_height)) for detection in detections]
		image = self.draw_boxes(detections, image, (255, 0, 0))
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
	def image_detection_cutresize(self, image_path, size=608):
		# Darknet doesn't accept numpy images.
		# Create one with image we reuse for each detect
		width = darknet.network_width(self.network)
		height = darknet.network_height(self.network)
		darknet_image = darknet.make_image(width, height, 3)

		image = cv2.imread(image_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_resized = self.img_resize(image_rgb, size)
		# cv2.imwrite('output_resize.jpg', image_resized)
		images_rgb = self.cutImage(image_resized, sliceSize=width)

		detections = []
		for i, image_rgb in enumerate(images_rgb):
			darknet.copy_image_from_bytes(darknet_image, image_rgb[0].tobytes())
			detectionsT = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
			detectionsT = [(detection[0], float(detection[1]), (detection[2][0], detection[2][1], detection[2][2], detection[2][3])) for detection in detectionsT]
			# imageT = self.draw_boxes(detectionsT, image_rgb[0], (255, 0, 0))
			# cv2.imwrite(str(i)+'output.jpg', imageT)

			detectionsT = [(detection[0], detection[1], (detection[2][0]+image_rgb[2], detection[2][1]+image_rgb[1], detection[2][2], detection[2][3])) for detection in detectionsT]
			detections += detectionsT

		ori_image_height, ori_image_width, _ = image.shape
		resize_height, resize_width, _ = image_resized.shape
		scale_height, scale_width = ori_image_height/resize_height, ori_image_width/resize_width
		detections = [(detection[0], detection[1]
					 ,(detection[2][0]*scale_width, detection[2][1]*scale_height
					 , detection[2][2]*scale_width, detection[2][3]*scale_height)) for detection in detections]
		detections = self.NMS(detections)
		image = self.draw_boxes(detections, image, (255, 0, 0))
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
	def cutImage(self, image, sliceSize=800):
		h, w, c = image.shape

		imgs = []
		for i in range(h//sliceSize*2+1):
			for j in range(w//sliceSize*2+1):
				img = np.zeros((sliceSize, sliceSize, 3), dtype=np.uint8)
				img[:] = 255
				temp = image[sliceSize//2*i: sliceSize//2*(i+2) if sliceSize//2*(i+2)<h else h
							,sliceSize//2*j: sliceSize//2*(j+2) if sliceSize//2*(j+2)<w else w]
				th, tw, tc = temp.shape
				img[:th, :tw] = temp
				imgs.append([img, i*sliceSize//2, j*sliceSize//2])
		return imgs
	def draw_boxes(self, detections, image, color):
		import cv2
		for label, confidence, bbox in detections:
			left, top, right, bottom = darknet.bbox2points(bbox)
			cv2.rectangle(image, (left, top), (right, bottom), color, 1)
			cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
						(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						color, 2)
		return image
	def img_resize(self, image, size):
		height, width = image.shape[0], image.shape[1]
		# 判断图片的长宽比率
		scale = width / height
		if width > height:
			img_new = cv2.resize(image, (size, int(size*scale)))
		else:
			img_new = cv2.resize(image, (int(size*scale), size))
		return img_new
	def NMS(self, dets, thresh=0.45): 
		x1, y1, x2, y2, scores = [], [], [], [], []
		for i in dets:
			cw, ch, w, h = i[2]
			x1.append(cw-w//2)
			y1.append(ch-h//2)
			x2.append(cw+w//2)
			y2.append(ch+h//2)
			scores.append(i[1])
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
		
if __name__ == "__main__":
	filename ="../../../groundtruth/image/FPK_02.jpg"
	yolov4 = YoloPredict(weights="./weights/20210122yolov4/yolov4_final.weights")

	# img, detect = yolov4.image_detection_cutresize(filename)
	# img, detect = yolov4.image_detection_cut(filename)
	img, detect = yolov4.image_detection_cutresize(filename, size=1208)
	# print(detect)
	cv2.imwrite('output.jpg', img)