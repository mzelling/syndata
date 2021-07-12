import numpy as np
import scipy.stats as stats
from .maxmin_sampler import maxmin_sampler
from ..core import CovGeom

class MaxMinCov(CovGeom):
	"""
	Constructs cluster covariance structures by setting ratios between maximum
	and minimum values of geometric parameters.


	Parameters
	----------
	ref_aspect : float			
		Reference aspect ratio for each cluster.
	aspect_maxmin : float
		Desired ratio between maximum and minimum aspect ratios among clusters.
	radius_max_min : float
		Desired ratio between maximum and minimum cluster radius.

	Attributes
	----------
	ref_aspect : float
	aspect_maxmin : float
	radius_maxmin : float
	"""
	
	def __init__(self, ref_aspect, aspect_maxmin, radius_maxmin):
		"""
		Constructs a MaxMinCov object.

		"""
		self.ref_aspect = ref_aspect
		self.aspect_maxmin = aspect_maxmin
		self.radius_maxmin = radius_maxmin
	

	def make_cluster_aspects(self, n_clusters):
		"""
		Generates aspect ratios (ratio between standard deviations along longest and shortest
		axes) for all clusters.

		Parameters
		----------
		n_clusters : int
			The number of clusters.

		Returns
		-------
		out : ndarray
			The aspect ratios for each cluster.

		"""

		min_aspect = 1 + (self.ref_aspect-1)/np.sqrt(self.aspect_maxmin)
		f = lambda a: ((self.ref_aspect-1)**2)/a
		return 1+maxmin_sampler(n_clusters, self.ref_aspect-1, min_aspect-1, self.aspect_maxmin, f)

		
	def make_cluster_radii(self, n_clusters, ref_radius, n_dim):
		""" 
		Computes cluster radii through constrained random sampling.

		The radius of a cluster is the geometric mean of the standard deviations along 
		the principal axes. Cluster radii are sampled such that the arithmetic mean of
		cluster volumes (cluster radius to the n_dim power) equals the reference volume
		(ref_radius to the n_dim power). The minimum and maximum radii are chosen so that
		the arithmetic average of the corresponding volumes equals the reference volume.

		Parameters
		----------
		n_clusters : int
			The number of clusters.
		ref_radius : float
			The reference radius for all clusters.
		n_dim : int
			The dimensionality of the clusters.

		Returns
		-------
		out : ndarray
			The cluster radii.
		"""
		min_radius = (2*(ref_radius**n_dim)/(1 + self.radius_maxmin**n_dim))**(1/n_dim)
		f = lambda r: (2*(ref_radius**n_dim) - (r**n_dim))**(1/n_dim)
		return maxmin_sampler(n_clusters, ref_radius, min_radius, self.radius_maxmin, f)
	

	def make_axis_sd(self, n_axes, sd, aspect):
		"""
		Generates standard deviations for the principal axes of a single cluster.

		Parameters
		----------
		n_axes : int
			Number of principal axes of this cluster
		sd : float
			Overall standard deviation for this cluster
		aspect : float
			Desired ratio between maximum and minimum axis standard deviations

		Returns
		-------

		"""
		min_sd = sd/np.sqrt(aspect)
		f = lambda s: (sd**2)/s
		return maxmin_sampler(n_axes, sd, min_sd, aspect, f)
		

	def make_cov(self, clusterdata, output_inv=True):
		"""
		Compute covariance structure for each cluster.

		Parameters
		----------
		clusterdata : a ClusterData object
			Specifies the number of clusters and other parameters
		output_inv : bool, optional
			If false, return only covariance matrices

		Returns
		-------
		out : a tuple (axis, sd), where cov and cov_inv are lists of ndarrays
			The lists cov and cov_inv contain the covariance and inverse covariance
			matrices for each cluster. Corresponding list indices refer to the same
			cluster. Alternatively, if output_inv=False then only the list cov is
			returned.
		"""

		axis = list()
		sd = list()
		cov = list()
		cov_inv = list()

		n_clusters = clusterdata.n_clusters
		n_dim = clusterdata.n_dim
		scale = clusterdata.scale
		
		cluster_radii = self.make_cluster_radii(n_clusters, scale, n_dim)
		cluster_aspects = self.make_cluster_aspects(n_clusters)
		
		for clust in range(n_clusters):
			# compute principal axes for cluster
			axes = self.make_orthonormal_axes(n_dim, n_dim)
			axis_sd = self.make_axis_sd(n_dim, cluster_radii[clust], cluster_aspects[clust])

			axis.append(axes)
			sd.append(axis_sd)
			cov.append(np.transpose(axes) @ np.diag(axis_sd**2) @ axes)
			cov_inv.append(np.transpose(axes) @ np.diag(1/axis_sd**2) @ axes)

		out = (axis, sd, cov, cov_inv)
			
		return out