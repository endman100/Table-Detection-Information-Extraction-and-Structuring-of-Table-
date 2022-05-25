import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import cv2

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
class optClass():
	def __init__(self):
		self.image_folder = "demo_image/"
		self.workers = 4
		self.batch_size = 192
		self.saved_model = "recognitionVersion1/checkpoint/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
		self.batch_max_length = 25
		self.imgH = 32
		self.imgW = 100
		self.rgb = False
		self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
		self.sensitive = True
		self.PAD = False
		self.Transformation="TPS"
		self.FeatureExtraction="ResNet"
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
			self.opt = self.setParser()
		print(self.opt.saved_model)
		self.converter, self.model = self.init2()
		self.demo_loader = None		
	def setParser(self):
		# parser = argparse.ArgumentParser()
		# parser.add_argument('--image_folder', default="demo_image/",  help='path to image_folder which contains text images')
		# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
		# parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
		# parser.add_argument('--saved_model',  default="recognitionVersion1/checkpoint/TPS-ResNet-BiLSTM-Attn.pth", help="path to saved_model to evaluation")
		# """ Data processing """
		# parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
		# parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
		# parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
		# parser.add_argument('--rgb', action='store_true', help='use rgb input')
		# parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
		# parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
		# parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
		# """ Model Architecture """
		# parser.add_argument('--Transformation', type=str,  default="TPS", help='Transformation stage. None|TPS')
		# parser.add_argument('--FeatureExtraction', type=str,  default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
		# parser.add_argument('--SequenceModeling', type=str,  default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
		# parser.add_argument('--Prediction', type=str,  default="Attn", help='Prediction stage. CTC|Attn')
		# parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
		# parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
		# parser.add_argument('--output_channel', type=int, default=512,
		# 					help='the number of output channel of Feature extractor')
		# parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
		# opt = parser.parse_args()

		""" vocab / character number configuration """
		opt = optClass()
		if opt.sensitive:
			opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

		cudnn.benchmark = True
		cudnn.deterministic = True
		opt.num_gpu = torch.cuda.device_count()
		return opt
	def init2(self):
		""" model configuration """
		if 'CTC' in self.opt.Prediction:
			converter = CTCLabelConverter(self.opt.character)
		else:
			converter = AttnLabelConverter(self.opt.character)
		self.opt.num_class = len(converter.character)

		if self.opt.rgb:
			self.opt.input_channel = 3
		model = Model(self.opt).cuda()
		model = torch.nn.DataParallel(model).to(self.device)
		model.load_state_dict(torch.load(self.opt.saved_model, map_location=self.device))


		# predict
		model.eval()
		return converter, model

	def predict(self, image=None):
		with torch.no_grad():
			image = torch.from_numpy(image).float().div(255).unsqueeze(0).unsqueeze(0)
			# print(image.shape, "image.shape")
		
			predList = []
			confidence_scoreList = []
			image = image.cuda()
			
			#print(image.shape)
			# For max length prediction
			length_for_pred = torch.IntTensor([self.opt.batch_max_length] * 1).to(self.device)
			text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0).to(self.device)

			if 'CTC' in self.opt.Prediction:

				preds = self.model(image, text_for_pred).log_softmax(2)

				# Select max probabilty (greedy decoding) then decode index to character
				preds_size = torch.IntTensor([preds.size(1)] * 1)
				_, preds_index = preds.max(2)
				preds_index = preds_index.view(-1)
				preds_str = self.converter.decode(preds_index.data, preds_size.data)

			else:
				# print(image.shape, "image.shape")
				preds = self.model(image, text_for_pred, is_train=False)

				# select max probabilty (greedy decoding) then decode index to character
				_, preds_index = preds.max(2)

				preds_str = self.converter.decode(preds_index, length_for_pred)

			preds_prob = F.softmax(preds, dim=2)
			preds_max_prob, _ = preds_prob.max(dim=2)
			# print(preds_str, preds_max_prob)
			for pred, pred_max_prob in zip(preds_str, preds_max_prob):
				# print(pred, pred_max_prob)
				if 'Attn' in self.opt.Prediction:
					pred_EOS = pred.find('[s]')
					pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
					pred_max_prob = pred_max_prob[:pred_EOS]
				if(len(pred_max_prob) == 0):
					confidence_score = 0
					predList = pred
					confidence_scoreList = 0
				else:
					confidence_score = pred_max_prob.cumprod(dim=0)[-1]
					predList = pred
					confidence_scoreList = confidence_score.item()
		return predList, confidence_scoreList
if __name__ == '__main__':
	recognition = recognitionModel()
	recognition.ChangeDataLoaderPath("../store/FPK_01/")
	img_name, pred, confidence_score = recognition.predict()
	for i in range(len(img_name)):
		print(f'{img_name[i]:25s}\t{pred[i]:25s}\t{confidence_score[i]:0.4f}')

