import pickle as pkl
import numpy as np
import sys

def parse(filename):
	with open(filename, 'rb') as f:
		data = pkl.load(f)
	times = [x[1]['top1']['time(s)'] for x in data]
	topeak = [x[1]['top5']['%peak'] for x in data]
	geomeans = [x[1]['geomean']['%peak'] for x in data]
	stds = [x[1]['std'] for x in data]
	return times, topeak, np.mean(times), np.std(times), geomeans, stds

if __name__ == "__main__":
	assert len(sys.argv) >= 2
	times, topeak, ave, std, geomeans, stds = parse(sys.argv[1])
	print("times: ")
	for t in times:
		print (t)
	print("topeak: ")
	for t in topeak:
		print (t)
	print ("geomeans:")
	for t in geomeans:
		print(t)
	print ("stds:")
	for std in stds:
		print (std)
	print(f"ave: {ave}, std: {std}")

	
	
	
