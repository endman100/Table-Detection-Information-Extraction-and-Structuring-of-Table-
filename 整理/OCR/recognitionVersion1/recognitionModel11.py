import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
cudnn.benchmark = True
cudnn.deterministic = True
class optClass():
	def __init__(self):
		self.image_folder = "demo_image/"
		self.workers = 4
		self.batch_size = 192
		self.saved_model = "./recognitionVersion1/checkpoint/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
		self.batch_max_length = 25
		self.imgH = 32
		self.imgW = 100
		self.rgb = False
		self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
		self.sensitive = True
		self.PAD = False
		self.Transformation="TPS"
		self.FeatureExtraction="VGG"
		self.SequenceModeling="BiLSTM"
		self.Prediction="Attn"
		self.num_fiducial=20
		self.input_channel=1
		self.output_channel=512
		self.hidden_size=256


class recognitionModel1():
	def __init__(self, opt=None):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if(opt):
			self.opt = opt
		else:
			self.opt = optClass()
		self.setParser()
		print(self.opt.saved_model)
		self.converter, self.model = self.init2(self.opt)
		self.initDataLoader()
	def setParser(self):
		if self.opt.sensitive:
			self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

		cudnn.benchmark = True
		cudnn.deterministic = True
		self.opt.num_gpu = torch.cuda.device_count()
	def init2(self, opt):
		""" model configuration """
		if 'CTC' in opt.Prediction:
			converter = CTCLabelConverter(opt.character)
		else:
			converter = AttnLabelConverter(opt.character)
		opt.num_class = len(converter.character)

		if opt.rgb:
			opt.input_channel = 3
		model = Model(opt).cuda()
		model = torch.nn.DataParallel(model).to(self.device)
		model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))

		
		# predict
		model.eval()
		return converter, model
	def initDataLoader(self, image_folder=None):
		if(image_folder):
			self.opt.image_folder = image_folder
		AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
		demo_data = RawDataset(root=self.opt.image_folder, opt=self.opt)  # use RawDataset
		demo_loader = torch.utils.data.DataLoader(
			demo_data, batch_size=self.opt.batch_size,
			shuffle=False,
			num_workers=int(self.opt.workers),
			collate_fn=AlignCollate_demo , pin_memory=True)
		self.demo_loader = demo_loader
	def predict(self):
		self.model.eval()
		returnText = []
		with torch.no_grad():
			for image_tensors, image_path_list in self.demo_loader:
				# cv2.imshow('My Image', (((image_tensors[0]*0.5+0.5)*255).permute(1, 2, 0).numpy().astype('uint8')))
				# cv2.waitKey(0)
				batch_size = image_tensors.size(0)
				image = image_tensors.to(self.device)
				# For max length prediction
				length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
				text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

				if 'CTC' in self.opt.Prediction:
					preds = self.model(image, text_for_pred)

					# Select max probabilty (greedy decoding) then decode index to character
					preds_size = torch.IntTensor([preds.size(1)] * batch_size)
					_, preds_index = preds.max(2)
					preds_index = preds_index.view(-1)
					# print(preds_index, preds_size)
					preds_str = self.converter.decode(preds_index, preds_size)

				else:
					preds = self.model(image, text_for_pred, is_train=False)

					# select max probabilty (greedy decoding) then decode index to character
					_, preds_index = preds.max(2)
					preds_str = self.converter.decode(preds_index, length_for_pred)


				preds_prob = F.softmax(preds, dim=2)
				preds_max_prob, _ = preds_prob.max(dim=2)
				# print("preds_max_prob", preds_max_prob)
				for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
					if 'Attn' in self.opt.Prediction:
						pred_EOS = pred.find('[s]')
						pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
						pred_max_prob = pred_max_prob[:pred_EOS]

					# calculate confidence score (= multiply of pred_max_prob)
					if(len(pred_max_prob) == 0):
						confidence_score = 0
					else:
						confidence_score = pred_max_prob.cumprod(dim=0)[-1]
					returnText.append(pred)
		return returnText
if __name__ == '__main__':
	pass
	# recognition = recognitionModel()
	# recognition.ChangeDataLoaderPath("../store/FPK_01/")
	# img_name, pred, confidence_score = recognition.predict()
	# for i in range(len(img_name)):
	# 	print(f'{img_name[i]:25s}\t{pred[i]:25s}\t{confidence_score[i]:0.4f}')

