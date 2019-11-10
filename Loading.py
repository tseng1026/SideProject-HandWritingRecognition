import os
import mnist
import numpy as np
import pandas as pd
from PIL import Image
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, datasets

def LoadData(train_or_test):
	import random
	
	if train_or_test == 0:
		img = mnist.train_images()
		lab = mnist.train_labels()
	
	if train_or_test == 1:
		img = mnist.test_images()
		lab = mnist.test_labels()

	data = list(zip(img, lab))
	random.shuffle(data)

	return data

class DataSet(Dataset):
	def __init__(self, data):
		self.data      = data
		self.transform = transforms.Compose([
						 transforms.ToTensor(),
						 transforms.Normalize((0.1307,), (0.3081,))
						 ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = self.data[ind][0]
		lab = self.data[ind][1]
		img = self.transform(img)
		lab = lab.astype("long")
		return img, lab