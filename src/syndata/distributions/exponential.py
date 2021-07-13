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
		cluster_size : int
		mean : ndarray
		axes : list of ndarray
		sd : list of float
		"""

		# each row of axes is an axis
		n_axes = axes.shape[0]

		X = np.zeros(shape=(cluster_size,n_axes));

		# sample along each principal axis independently
		for i, stdev in enumerate(sd):
			# use exponential with desired standard deviation and random sign
			signs = np.random.choice(a=[-1,+1],size=cluster_size,replace=True)
			X[:,i] = np.random.exponential(scale=stdev, size=cluster_size) * signs

		# transform X from principal axis coordinates to standard coordinates
		X = X @ axes

		# mean-shift X
		X = X + mean[np.newaxis,:]

		return X