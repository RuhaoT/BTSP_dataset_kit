from .lsh import *

class flylsh(LSH):
	def __init__(self, data, hash_length, sampling_ratio, embedding_size, binary_mode = False, device = 'cpu'):
		"""
		data: Nxd matrix
		hash_length: scalar
		sampling_ratio: fraction of input dims to sample from when producing a hash
		embedding_size: dimensionality of projection space, m; also the dimensions of weights
		Note that in Flylsh, the hash length and embedding_size are NOT the same
		whereas in usual LSH they are
		"""
		self.hash_length = hash_length
		self.embedding_size = embedding_size
		K = embedding_size // hash_length
		# data normalization
		self.data = data
		self.device = device

		num_projections = int(sampling_ratio * data.shape[1])
		weights = torch.rand((data.shape[1], embedding_size), device=device)
		yindices = np.arange(weights.shape[1])[None, :]
		xindices = weights.argsort(axis=0)[-num_projections:, :] # maintain the top-k values and set to one
		self.weights = torch.zeros_like(weights, device=device)
		self.weights[xindices, yindices] = 1  # sparse projection vectors

		#step2: projection and maintain the salient hash_length vectors.
		all_activations = (self.data @ self.weights)
		self.mem  = all_activations

		# # Type 1: binary representations
		# xindices = np.arange(data.shape[0])[:, None]
		# yindices = all_activations.argsort(axis=1)[:, -hash_length:]  # representation dimension
		# self.hashes = np.zeros_like(all_activations, dtype=np.bool)
		# self.hashes[xindices, yindices] = True  # choo

		# self.mem = self.data @ self.weights
		self.hashes = torch.zeros_like(all_activations, device=device)
		# Type 2: Mem representations
		xindices = np.arange(data.shape[0])[:, None]
		yindices = self.mem.argsort(axis=1)[:, -hash_length:]  # representation dimension
		self.hashes[xindices, yindices] = self.mem[xindices, yindices]  # choo
		if not binary_mode:
			self.hashes[xindices, yindices] = self.mem[xindices, yindices]  # masking the neurons whose mem are less than the threshold nps.
		else:
			self.hashes[xindices, yindices] = 1
		# npts = np.percentile(self.mem, q = 100 - self.hash_length / self.embedding_size * 100, axis=1)
		# if not binary_mode:
		# 	self.hashes = self.mem * (self.mem >= np.tile(npts.reshape(-1, 1), [1,embedding_size]))  # masking the neurons whose mem are less than the threshold nps.
		# else:
		# 	self.hashes = np.float32(self.mem >= npts.reshape(-1, 1))


	def compute_query_mAP(self, n_points, search_radius=1, order=False, qtype='lowd', nnn=None):
		sample_indices = np.random.choice(self.hashes.shape[0], n_points)
		average_precisions = []
		elapsed = []
		numpredicted = []
		ms = lambda l: (np.mean(l), np.std(l))
		for qidx in sample_indices:
			start = time.time()
			if qtype == 'lowd':
				predicted = self.query_lowd_bins(qidx, search_radius, order)
			elif qtype == 'highd':
				predicted = self.query_highd_bins(qidx, order)
			assert len(predicted) < self.hashes.shape[0], 'All point being queried'

			if nnn is None:
				elapsed.append(time.time() - start)
			else:
				if len(predicted) < nnn:
					# raise ValueError('Not a good search radius')
					continue
				elapsed.append(time.time() - start)
				numpredicted.append(len(predicted))

				predicted = predicted[:nnn]

			truenns = self.true_nns(qidx, nnn=len(predicted))
			average_precisions.append(self.AP(predictions=predicted, truth=truenns))
		if nnn is not None:
			if len(average_precisions) < 0.8 * n_points:
				raise ValueError('Not a good search radius')

		return [*ms(average_precisions), *ms(elapsed), *ms(numpredicted)]

	def rank_and_findmAP(self, n_points, nnn):
		ms = lambda l: (np.mean(l), np.std(l))
		average_precisions = []
		elapsed = []
		for idx in range(n_points):
			start = time.time()
			average_precisions.append(self.findmAP(nnn, 1))
			elapsed.append(time.time() - start)
		return [*ms(average_precisions), *ms(elapsed)]

	def update_hash(self, data, binary_mode = False):
		activations = data @ self.weights
		xindices = np.arange(data.shape[0])[:, None]
		yindices = activations.argsort(axis=1)[:, -self.hash_length:]  # representation dimension
		hashes = np.zeros((data.shape[0], self.embedding_size))
		if not binary_mode:
			hashes[xindices, yindices] = activations[xindices, yindices]  # masking the neurons whose mem are less than the threshold nps.
		else:
			hashes[xindices, yindices] = 1
		return hashes
