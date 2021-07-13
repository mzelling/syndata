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
		pass

	def sample_cluster(self,cluster_size, mean,axes,sd):
		"""
		"""
		cov = np.transpose(axes) @ np.diag(sd**2) @ axes
		X = np.random.multivariate_normal(mean=mean, 
									cov=cov, size=cluster_size)
		return X
