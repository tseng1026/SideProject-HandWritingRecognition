import torch.nn as nn
import torchvision
import torchvision.models as models

class HandWrittenDigit(nn.Module):
	def __init__(self):
		super(HandWrittenDigit, self).__init__()

		self.conv1 = nn.Sequential(
					 nn.Conv2d(1, 20, kernel_size=5, padding=1),
					 nn.LeakyReLU(),
					 nn.MaxPool2d(2),
					 )

		self.conv2 = nn.Sequential(
					 nn.Conv2d(20, 50, kernel_size=5, padding=1),
					 nn.LeakyReLU(),
					 nn.MaxPool2d(2),
					 )

		self.last = nn.Sequential(
					nn.Linear(50*5*5, 500),
					nn.ReLU(),
					nn.Linear(500, 10),
					)

	def forward(self, img):
		tmp = self.conv1(img)			# basic convolution layer 1
		tmp = self.conv2(tmp)			# basic convolution layer 2
		tmp = tmp.view(-1, 50*5*5)		# resize to 1 dimension
		mod = self.last(tmp)			# correspond to resultant labels
		return mod
