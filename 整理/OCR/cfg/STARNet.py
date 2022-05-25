class optClass():
	def __init__(self):
		self.image_folder = "../groundtruth/recognitionRotateBinary/all/"
		self.workers = 4
		self.batch_size = 192
		self.saved_model = "./recognitionVersion1/checkpoint/TPS-ResNet-BiLSTM-CTC.pth"
		self.batch_max_length = 25
		self.imgH = 32
		self.imgW = 100
		self.rgb = False
		self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
		self.sensitive = False
		self.PAD = False
		self.Transformation="TPS"
		self.FeatureExtraction="ResNet"
		self.SequenceModeling="BiLSTM"
		self.Prediction="CTC"
		self.num_fiducial=20
		self.input_channel=1
		self.output_channel=512
		self.hidden_size=256