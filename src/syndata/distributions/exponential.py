from ..core import DataDist
import numpy as np

class ExpData(DataDist):
	"""
	Generates multivariate exponential data for ClusterData.
	"""

	def __init__(self):
		"""
		Create a ExpData object.
		"""
		pass

	def sample_cluster(self,cluster_size, mean,axes,sd):
		"""
		Sample an exponentially distributed cluster.

		cluster_size : int
			The number of data points in this cluster.
		mean : ndarray
			Cluster of this cluster.
		axes : list of ndarray
			Principle axes of this cluster.
		sd : list of float
			Standard deviations of this cluster's principal axes.
		"""

		# each row of axes is an axis
		n_axes = axes.shape[0]

		# sample on the unit sphere, then dilate/compress along principal axes
		X = np.random.multivariate_normal(mean=np.zeros(n_axes), cov=np.eye(n_axes), 
										  size=cluster_size) * \
			np.random.exponential(scale=1,size=cluster_size)[:,np.newaxis]

		X = X @ np.diag(sd) @ axes

		# mean-shift X
		X = X + mean[np.newaxis,:]

		return X