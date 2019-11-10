import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import Parsing
import Loading
import Model

if __name__=='__main__':
	gpu = torch.cuda.is_available()
	
	# parsing the arguments
	args = Parsing.Args()
	modlname = args.m
	outputfile = args.o

	test = Loading.LoadData(1)
	numb = len(test)

	test = Loading.DataSet(test)
	test = DataLoader(test, batch_size=32, shuffle=False)
	print ("[Done] Loading all data (testing)!")

	# load done-training model
	model = Model.HandWrittenDigit()
	check = torch.load(modlname)
	model.load_state_dict(check)
	model = model.cuda()
	print ("[Done] Initializing all model!")

	# set to evaluation mode
	model.eval()

	truth = torch.LongTensor().cuda()
	predt = torch.LongTensor().cuda()
	for ind, (img, lab) in enumerate(test):
		img = img.cuda()
		lab = lab.cuda()
		out = model(img)
		
		# compute the accuracy value
		pred = torch.max(out, 1)[1]
		predt = torch.cat((predt, pred))
		truth = torch.cat((truth, lab))

	scre = np.mean((truth == predt).cpu().numpy())
	print ("[Done] Computing accuracy: {:.4f}".format(scre))

	# write the results to file
	index = np.arange(numb)
	index = index.astype("int")

	predict = predt.type(torch.FloatTensor).cpu().numpy().squeeze()
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)
