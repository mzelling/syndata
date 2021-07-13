from ..core import DataDist
import numpy as np

class tData(DataDist):
	"""
	Generates multivariate t-distributed data for ClusterData.
	"""

	def __init__(self, df=1):
		"""
		Create a tData object.
		"""
		self.df = df

	def sample_cluster(self,cluster_size, mean,axes,sd):
		"""
		cluster_size : int
		mean : ndarray
		axes : list of ndarray
		sd : list of float
		"""

		# compute median for absolute value of t distribution
		n = 1000000
		abs_t_median = np.median(np.random.standard_t(df=self.df,size=n))

		# each row of axes is an axis
		n_axes = axes.shape[0]
		X = np.zeros(shape=(cluster_size,n_axes));

		# sample along each principal axis independently
		for i, stdev in enumerate(sd):
			# scale t distribution so that its median equals stdev
			scaling = stdev/(2*abs_t_median)
			X[:,i] = scaling * np.random.standard_t(df=self.df, size=cluster_size)

		# transform X from principal axis coordinates to standard coordinates
		X = X @ axes

		# mean-shift X
		X = X + mean[np.newaxis,:]

		return X