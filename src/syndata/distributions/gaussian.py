from ..core import DataDist
import numpy as np

class GaussianData(DataDist):
	"""
	Generate multivariate Gaussian data for ClusterData.
	"""

	def __init__(self):
		"""
		Create a GaussianData object.
		"""

	def sample(self, clusterdata):
		n_clusters = clusterdata.n_clusters
		n_samples = clusterdata.n_samples
		n_dim = clusterdata.n_dim
		class_sizes = clusterdata.class_sizes

		X = np.full(shape=(n_samples, 1+n_dim), fill_value=np.nan)
		start = 0
		for i in range(n_clusters):
			end = start + class_sizes[i]
			# Set class label
			X[start:end, -1] = i
			# Sample data
			X[start:end,:-1] = np.random.multivariate_normal(mean=clusterdata.centers[i], 
									cov=clusterdata.cov[i], size=class_sizes[i])
			start = end

		return X
