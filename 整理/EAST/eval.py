import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil
import zipfile


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)

	with zipfile.ZipFile('submit.zip', 'w') as zf:
		for filename in os.listdir(submit_path):
			zf.write(os.path.join(submit_path, filename), filename)
	
	
	res = subprocess.getoutput('python ./evaluate/script.py –g=./dataCreate/test_gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	# print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = './pths/Create/model_epoch_8.pth'
	test_img_path = os.path.abspath('./dataCreate/test_img/')
	submit_path = './submit'
	eval_model(model_name, test_img_path, submit_path)

