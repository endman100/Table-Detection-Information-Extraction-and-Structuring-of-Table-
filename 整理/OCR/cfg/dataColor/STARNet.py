class optClass():
	def __init__(self):
		self.image_folder = "../groundtruth/recognitionRotateBinary/all/"
		self.workers = 4
		self.batch_size = 192
		self.saved_model = "./saved_models/dataColor/TPSResNetBiLSTMCTC/best_accuracy.pth"
		self.batch_max_length = 25
		self.imgH = 32
		self.imgW = 100
		self.rgb = True
		self.character = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_()/#-+'
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