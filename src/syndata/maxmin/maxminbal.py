import numpy as np
from ..core import ClassBal
from .maxmin_sampler import maxmin_sampler

class MaxMinBal(ClassBal):
	"""
	Specify class imbalance by setting the ratio between the highest and lowest class sizes.
	"""

	def __init__(self, imbal_ratio):
		"""
		Creates a MaxMinBal object.
		"""

		self.imbal_ratio = imbal_ratio

	def float_to_int(self, float_class_sz, n_samples):
		class_sz = 1 + np.sort(np.round(float_class_sz))
		to_shrink = len(class_sz) - 1
		while (np.sum(class_sz) > n_samples):
			if (class_sz[to_shrink] > 1):
				class_sz[to_shrink] -= 1
				to_shrink -= 1
			else:
				to_shrink = len(class_sz) - 1
		return class_sz.astype(int)

	def make_class_sizes(self, clusterdata):
		n_samples = clusterdata.n_samples
		n_clusters = clusterdata.n_clusters

		# Set average class size as the reference size.
		ref_class_sz = n_samples/n_clusters

		# Determine minimum class size by requiring the average of the minimum and maximum
		# class sizes to be the reference size.
		min_class_sz = 2*ref_class_sz/(1 + self.imbal_ratio)

		# Set a pairwise sampling constraint to ensure that the sample sizes add to n_samples.
		f = lambda s: (2*ref_class_sz - s)

		# compute float class size estimates
		float_class_sz = maxmin_sampler(n_clusters, ref_class_sz, min_class_sz, self.imbal_ratio, f)

		# transform float class size estimates into integer class sizes
		class_sz = self.float_to_int(float_class_sz, n_samples)

		return class_sz





