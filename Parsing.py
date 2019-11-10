import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", default="./model_best.pth.tar")
	parser.add_argument("-o", default="./prediction.csv")

	args = parser.parse_args()
	return args
