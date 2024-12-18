import numpy as np
from scipy.io import loadmat
import pickle, time, os, random
from collections import OrderedDict as odict
import torch

def weighted_pp_sample(w,k, T):
	x = np.random.rand(w.shape[0], w.shape[1])
	w_norm = w / np.tile(w.max(axis=0).reshape(1, -1),[w.shape[0], 1])
	s = (1 - T) * w_norm + T * x
	order_idx = np.sort(s, axis=0)
	npts = order_idx[-k]
	f = s >= npts
	return f



def dist(X, Y):
	""" Computes the distance between two vectors. """
	return np.linalg.norm(np.float32(X) - np.float32(Y), ord=2)  # same as scipy euclidean but faster!

def sample_dist(X, Y):
	# return torch.norm(Y-X.reshape(1,-1),dim=1,p=2)
	# X, Y = np.float32(X), np.float32(Y)
	# f = np.linalg.norm(Y-X.reshape(1,-1),axis=1,ord=2)
	# return f
	return torch.norm(Y-X.reshape(1,-1),dim=1,p=2)
	# return np.sum((np.float32(X) - np.float32(Y)) ** 2, axis=1)

def tesht_map_dist(D, H):
	import heapq
	""" Computes mean average precision (MAP) and distortion between true nearest-neighbors  
		in input space and approximate nearest-neighbors in hash space. 
	"""
	H = torch.Tensor(H)
	N = D.shape[0]
	NUM_NNS = max(10,int(0.02*N))
	queries = random.sample(range(N), min(500,N))

	MAP = []  # [list of MAP values for each query]

	for i in queries:
  # # list of (dist hash space ,odor) from i.
		dij_orig = sample_dist(D[i, :], D)
		nn_orig = dij_orig.argsort(axis=0)
		true_nns =  nn_orig[(nn_orig != i)][:NUM_NNS]
  		# dij_hash = torch.norm(H-H[i, :].reshape(1,-1),dim=1,p=2)
		dij_hash = sample_dist(H[i, :], H)
		pred_nns = dij_hash.argsort(axis=0)
		pred_nns = pred_nns[(pred_nns != i)][:NUM_NNS]
		assert len(true_nns) == len(pred_nns)
		# Compute MAP: https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
		# E.g.  if the top NUM_NNS results are:   1, 0, 0,   1,   1,   1
		#       then the MAP is:            avg(1/1, 0, 0, 2/4, 3/5, 4/6)
		num_correct_thus_far = 0
		map_temp = []
		for idx, nghbr in enumerate(pred_nns):
			if nghbr in true_nns:
				num_correct_thus_far += 1
				map_temp.append((num_correct_thus_far) / (idx + 1))

		map_temp = np.mean(map_temp) if len(map_temp) > 0 else 0
		assert 0.0 <= map_temp <= 1.0
		MAP.append(map_temp)
	x_map = np.mean(MAP)
	# print('Trial', np.mean(x_map), np.std(x_map))
	return x_map


def setup_seed(seed):
	np.random.seed(seed)
	random.seed(seed)


