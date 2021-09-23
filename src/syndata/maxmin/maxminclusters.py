from .maxmincov import MaxMinCov
from .maxminbal import MaxMinBal
from ..core import ClusterData
from ..centers.boundedsepcenters import BoundedSepCenters
from ..distributions.gaussian import GaussianData
from ..distributions.exponential import ExpData
from ..distributions.standard_t import tData

class MaxMinClusters(ClusterData):
	"""
	Default implementation for ClusterData, uses MaxMin sampling approach.

	"""

	def __init__(self, n_clusters, n_dim, n_samples, imbal_maxmin,
				 aspect_maxmin, radius_maxmin, min_sep=1, max_sep=1.5, 
				 aspect_ref=1.5, scale=1.0, packing=0.1,dist='gaussian',df=1):

		cov_geom = MaxMinCov(ref_aspect=aspect_ref, aspect_maxmin=aspect_maxmin, 
							 radius_maxmin=radius_maxmin)
		center_geom = BoundedSepCenters(min_sep=min_sep,max_sep=max_sep, 
										packing=packing)
		class_bal = MaxMinBal(imbal_ratio=imbal_maxmin)

		if dist=='t':
			data_dist = tData(df=df)
		elif dist=='exp':
			data_dist = ExpData()
		elif dist=='gaussian':
			data_dist = GaussianData()
		else:
			raise ValueError("Distribution not found. Use dist='gaussian' " +
							 "for Gaussian data, dist='t' for t-distributed data," + 
							 " or dist='exp' for exponentially distributed data.")

		# in line below, used to be super().__init__
		ClusterData.__init__(n_clusters,n_dim,n_samples,class_bal,cov_geom,
						 	 center_geom,data_dist,scale)

