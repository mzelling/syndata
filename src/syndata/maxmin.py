"""
This module provides implementations based on max-min sampling for various
aspects of cluster generation.

CLASSES AND METHODS
	
	MaxMinClusters : data generator based on max-min sampling
		__init__(self, n_clusters, n_dim, n_samples, ...)

	MaxMinCov : sample cluster shapes (covariance structures)
		__init__(self, ref_aspect, aspect_maxmin, radius_maxmin)
		make_cluster_aspects(self, n_clusters)
		make_cluster_radii(self, n_clusters, ref_radius, n_dim)
		make_axis_sd(self, n_axes, sd, aspect)
		make_cov(self, clusterdata)

	MaxMinBal : sample number of data points per cluster
		__init__(self, imbal_ratio)
		float_to_int(self, float_class_sz, n_samples)
		make_class_sizes(self, clusterdata)

	maxmin_sampler(n_samples, ref, min_val, maxmin_ratio, f_constrain)

"""

from .core import ClusterData, CovGeom, ClassBal
from .centers import BoundedSepCenters
from .distributions import GaussianData, ExpData, tData
import numpy as np
import scipy.stats as stats

class MaxMinClusters(ClusterData):
	"""
	Data generator based on max-min sampling. Generate data sets that share
	geometric properties based on max-min ratios. This is the default 
	implementation of ClusterData. 

	You can specify the sampling mechanism for each of the following geometric
	parameters: cluster radius, cluster aspect ratio and class size.


	Geometric attributes
	--------------------
	Cluster radius : float, >0
		Geometric mean of the standard deviations along a cluster's principal
		axes (eigenvectors of covariance matrix)
	Cluster aspect ratio : float, >=1
		Ratio between maximum and minimum standard deviations across a cluster's
		principal axes (eigenvectors of covariance matrix).
	Class size : int
		Number of data points in a cluster


	Configuring geometric attributes
	--------------------------------
	radius_maxmin : float, >=1
		Ratio between maximum and minimum radii across all clusters
	aspect_maxmin : float, >=1
		Ratio between maximum and minimum aspect ratios across all clusters
	aspect_ref : float, >=1
		Reference aspect ratio (aspect ratio a "typical" cluster should have)
	imbal_maxmin : float, >=1
		Ratio between highest and lowest class sizes across clusters


	Attributes
	----------
	All attributes are inherited from the class ClusterData. However, a
	MaxMinClusters object uses specific implementations of the following 
	attributes:

	center_geom : centers.BoundedSepCenters
		Sample cluster centers that satisfy separation constraints
	cov_geom : MaxMinCov
		Sample cluster shapes (covariance structures) with max-min sampling
	class_bal : MaxMinBal
		Sample class sizes (data points in each cluster) with max-min sampling
	data_dist : GaussianData, ExpData, or tData
		Data distribution is chosen at initialization (the dist argument)


	Methods
	-------
	__init__(self, n_clusters, n_dim, n_samples, ...)

	"""

	def __init__(self, n_clusters=6, n_dim=2, n_samples=500, imbal_maxmin=2,
				 aspect_maxmin=2, radius_maxmin=3, alpha_max=0.05, alpha_min=1e-3, 
				 aspect_ref=1.5, packing=0.1,dist='gaussian',df=1, scale=1.0):
		"""
		Create a MaxMinClusters object. 

		See the documentation of class MaxMinClusters for more information about
		the following geometric parameters: cluster radius, cluster aspect ratio,
		and class size.

		Parameters
		----------
		self : MaxMinClusters
			This instance of MaxMinClusters
		n_clusters : int
			Number of clusters to generate.
		n_dim : int
			Dimensionality of the data set to generate.
		n_samples : int
			Total number of samples to generate.
		radius_maxmin : float, >=1
			Ratio between the maximum and minimum cluster radii
		aspect_maxmin : float, >=1
			Ratio between maximum and minimum cluster aspect ratios
		aspect_ref : float, >=1
			Reference (typical) aspect ratio for clusters
		imbal_maxmin : float, >=1
			Ratio between highest and lowest class sizes across clusters
		min_sep : float, >0
			Minimum separation between clusters. For Gaussian clusters, a 
			value of 1.0 indicates that two clusters are just touching (see 
			centers.BoundedSepCenters for more information)
		max_sep : float, >0
			Maximum separation between clusters. For Gaussian clusters, a 
			value of 1.0 indicates that two clusters are just touching (see 
			centers.BoundedSepCenters for more information)
		packing : float between 0 and 1
			Ratio between total volume of clusters to sampling volume (see
			centers.BoundedSepCenters for more information)
		dist : str
			String indicating which probability distribution to use for drawing
			data. There are currently three possibilities: 'gaussian' for 
			Gaussian data, 't' for standard t-distributed data, and 'exp' for 
			double exponentially distributed data.
		df : int
			Degrees of freedom in Student's t distribution to use when
			sampling t-distributed clusters (applicable when dist='t')
		scale : float
			Reference length scale for generated data

		Returns
		-------
		out : MaxMinClusters
			Data generator for sampling synthetic datasets with desired 
			geometric characteristics

		"""

		cov_geom = MaxMinCov(ref_aspect=aspect_ref, aspect_maxmin=aspect_maxmin, 
							 radius_maxmin=radius_maxmin)
		center_geom = BoundedSepCenters(alpha_max=alpha_max,alpha_min=alpha_min, 
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
		ClusterData.__init__(self, n_clusters,n_dim,n_samples,class_bal,cov_geom,
							 center_geom,data_dist,scale)


class MaxMinCov(CovGeom):
	"""
	Defines cluster shapes by setting ratios between maximum and minimum values
	of geometric parameters. The algorithms use pairwise max-min sampling.

	See documentation of class MaxMinClusters for more information.

	Attributes
	----------
	ref_aspect : float, >= 1            
		Reference aspect ratio for each cluster.
	aspect_maxmin : float, >= 1
		Desired ratio between maximum and minimum aspect ratios among clusters.
	radius_max_min : float, >= 1
		Desired ratio between maximum and minimum cluster radius.

	Methods
	-------
	__init__(self, ref_aspect, aspect_maxmin, radius_maxmin)
	make_cluster_aspects(self, n_clusters)
	make_cluster_radii(self, n_clusters, ref_radius, n_dim)
	make_axis_sd(self, n_axes, sd, aspect)
	make_cov(self, clusterdata, ...)

	"""
	
	def __init__(self, ref_aspect, aspect_maxmin, radius_maxmin):
		"""
		Constructs a MaxMinCov object.

		Parameters
		----------
		self : MaxMinCov
			This instance of MaxMinCov
		ref_aspect : float, >= 1            
			Reference aspect ratio for each cluster.
		aspect_maxmin : float, >= 1
			Desired ratio between maximum and minimum aspect ratios among 
			clusters.
		radius_max_min : float, >= 1
			Desired ratio between maximum and minimum cluster radius.

		"""
		self.ref_aspect = ref_aspect
		self.aspect_maxmin = aspect_maxmin
		self.radius_maxmin = radius_maxmin
	

	def make_cluster_aspects(self, n_clusters,seed=None):
		"""
		Generates aspect ratios (ratio between standard deviations along longest
		and shortest axes) for all clusters.

		Parameters
		----------
		self : MaxMinCov
			This instance of MaxMinCov
		n_clusters : int
			The number of clusters.

		Returns
		-------
		out : ndarray
			The aspect ratios for each cluster.
		"""

		min_aspect = 1 + (self.ref_aspect-1)/np.sqrt(self.aspect_maxmin)
		f = lambda a: ((self.ref_aspect-1)**2)/a
		return 1+maxmin_sampler(n_clusters, self.ref_aspect-1, min_aspect-1, self.aspect_maxmin, f,
								seed=seed)

		
	def make_cluster_radii(self, n_clusters, ref_radius, n_dim, seed=None):
		""" 
		Sample cluster radii with pairwise max-min sampling.

		The radius of a cluster is the geometric mean of the standard deviations along 
		the principal axes. Cluster radii are sampled such that the arithmetic mean of
		cluster volumes (cluster radius to the n_dim power) equals the reference volume
		(ref_radius to the n_dim power). The minimum and maximum radii are chosen so that
		the arithmetic average of the corresponding volumes equals the reference volume.

		Parameters
		----------
		self : MaxMinCov
			This instance of MaxMinCov
		n_clusters : int
			Number of clusters
		ref_radius : float
			Reference radius for all clusters
		n_dim : int
			Dimensionality of the data

		Returns
		-------
		out : ndarray
			Cluster radii
		"""
		log_min_radius = np.log(ref_radius) - (1/2)*np.log(self.radius_maxmin)
		f = lambda log_r: 2*np.log(ref_radius) - log_r

		#min_radius = (2*(ref_radius**n_dim)/(1 + self.radius_maxmin**n_dim))**(1/n_dim)
		#f = lambda r: (2*(ref_radius**n_dim) - (r**n_dim))**(1/n_dim)

		log_max_radius = np.log(ref_radius) + (1/2)*np.log(self.radius_maxmin)
		maxmin_log_ratio = log_max_radius/log_min_radius

		return np.exp(maxmin_sampler(n_clusters, np.log(ref_radius), log_min_radius, maxmin_log_ratio, f,
							  seed=seed))
	

	def make_axis_sd(self, n_axes, sd, aspect, seed=None):
		"""
		Sample standard deviations for the principal axes of a single cluster.

		Parameters
		----------
		self : MaxMinCov
			This instance of MaxMinCov
		n_axes : int
			Number of principal axes of this cluster, same as dimensionality
		sd : float
			Overall standard deviation of this cluster (geometric mean of 
			standard deviations)
		aspect : float
			Desired ratio between maximum and minimum standard deviations
			across all principal axes

		Returns
		-------
		out : ndarray
			Standard deviations along principal axes of this cluster

		"""
		min_sd = sd/np.sqrt(aspect)
		f = lambda s: (sd**2)/s
		return maxmin_sampler(n_axes, sd, min_sd, aspect, f, seed=seed)
		

	def make_cov(self, clusterdata, seed=None):
		"""
		Compute covariance structure (cluster shape) for each cluster.

		Parameters
		----------
		self : MaxMinCov
			This instance of MaxMinCov
		clusterdata : ClusterData
			Specifies the number of clusters and other parameters

		Returns
		-------
		out : tuple (axis, sd, cov, cov_inv), where

			axis : list of ndarray
				The i-th element stores principal axes of the i-th cluster
			sd : list of ndarray
				The i-th element stores standard deviations along the principal
				axes of the i-th cluster
			cov : list of ndarray
				The i-th element stores covariance matrix of the i-th cluster
			cov_inv : list of ndarray
				The i-th element stores inverse covariance matrix of the i-th
				cluster

			Matching indices of the output lists refer to the same cluster.

		"""
		if seed:
			np.random.seed(seed)

		axis = list()
		sd = list()
		cov = list()
		cov_inv = list()

		n_clusters = clusterdata.n_clusters
		n_dim = clusterdata.n_dim
		scale = clusterdata.scale
		
		cluster_radii = self.make_cluster_radii(n_clusters, scale, n_dim, seed=seed+2*n_clusters if seed else seed)
		cluster_aspects = self.make_cluster_aspects(n_clusters,seed=seed+3*n_clusters if seed else seed)
		
		for clust in range(n_clusters):
			# compute principal axes for cluster
			axes = self.make_orthonormal_axes(n_dim, n_dim, seed=seed+clust if seed else seed)
			axis_sd = self.make_axis_sd(n_dim, cluster_radii[clust], cluster_aspects[clust],
										seed=seed+clust if seed else seed)

			axis.append(axes)
			sd.append(axis_sd)

			# consider not constructing cov, cov_inv here; instead do it only on request
			cov.append(np.transpose(axes) @ np.diag(axis_sd**2) @ axes)
			cov_inv.append(np.transpose(axes) @ np.diag(1/axis_sd**2) @ axes)

		out = (axis, sd, cov, cov_inv)
			
		return out


class MaxMinBal(ClassBal):
	"""
	Generate class sizes (number of data points in each cluster) with
	pairwise max-min sampling.

	Pairwise max-min sampling uses the average class size as the reference
	value. The sampled class sizes sum to the desired total number of samples
	as specified by the relevant ClusterData object (an argument in the method
	MaxMinBal.make_class_sizes).

	Attributes
	----------
	imbal_ratio : float, >=1
		Desired ratio between largest and smallest class size

	Methods
	-------
	__init__(self, imbal_ratio)
	make_class_sizes(self, clusterdata)

	"""

	def __init__(self, imbal_ratio):
		"""
		Instantiate a MaxMinBal object.

		See the documentation of class MaxMinBal for more information.

		Parameters
		----------
		self : MaxMinBal
			This instance of MaxMinBal
		imbal_ratio : float, >=1
			Desired ratio between largest and smallest class size 

		Returns
		-------
		out : MaxMinBal
			MaxMinBal object for sampling class sizes
		"""

		self.imbal_ratio = imbal_ratio

	def float_to_int(self, float_class_sz, n_samples):
		"""
		Convert float class sizes to integer class sizes 
		while ensuring 1) that each class size is at least
		1 and 2) that the sum of class sizes is n_samples.

		Parameters
		----------
		self : MaxMinBal
			This instance of MaxMinBal
		float_class_sz : ndarray, dtype=float
			Approximate class sizes, not necessarily integer
		n_samples : int
			Desired total number of samples (sum of class sizes)

		Returns
		-------
		out : ndarray, dtype=int
			Class sizes (number of data points in each cluster)
		"""
		# round float class sizes and add 1, then sort
		class_sz = 1 + np.sort(np.round(float_class_sz))
		# start by shrinking the highest class sizes
		class2shrink_idx = len(class_sz) - 1
		while (np.sum(class_sz) > n_samples):
			if (class_sz[class2shrink_idx] > 1):
				class_sz[class2shrink_idx] -= 1
				class2shrink_idx -= 1
			else:
				class2shrink_idx = len(class_sz) - 1
		return class_sz.astype(int)

	def make_class_sizes(self, clusterdata, seed=None):
		"""
		Sample class size (number of data points) for each cluster with
		pairwise max-min sampling. 

		See the documentation of class MaxMinBal for more information.

		Parameters
		----------
		self : MaxMinBal
			This instance of MaxMinBal
		clusterdata : ClusterData
			The underlying data generator

		Returns
		-------
		out : ndarray
			Class sizes (number of data points in each cluster)

		"""
		n_samples = clusterdata.n_samples
		n_clusters = clusterdata.n_clusters

		# Set average class size as the reference size.
		ref_class_sz = n_samples/n_clusters

		# Determine minimum class size by requiring average of minimum and maximum
		# class sizes to be the reference size.
		min_class_sz = 2*ref_class_sz/(1 + self.imbal_ratio)

		# Set pairwise sampling constraint to ensure sample sizes add to n_samples.
		f = lambda s: (2*ref_class_sz - s)

		# compute float class size estimates
		float_class_sz = maxmin_sampler(n_clusters, ref_class_sz, 
										min_class_sz, self.imbal_ratio, f, 
										seed=seed)

		# transform float class size estimates into integer class sizes
		class_sz = self.float_to_int(float_class_sz, n_samples)

		return class_sz


def maxmin_sampler(n_samples, ref, min_val, maxmin_ratio, f_constrain, seed=None):
	"""
	Generates samples around a reference value, with a fixed ratio between the 
	maximum and minimum sample. Sample pairwise to enforce a further constraint 
	on the samples. For example, the geometric mean of the samples can be 
	specified.

	Parameters
	----------
	n_samples : int
	ref : float
	min_val : float
	maxmin_ratio : float
	f_constrain : function

	Returns
	-------
	out : ndarray

	"""
	np.random.seed(seed)

	if (maxmin_ratio == 1) or (min_val == 0):
		out = np.full(n_samples, fill_value=ref)
		return out

	max_val = min_val * maxmin_ratio
	
	if (n_samples > 2):
		# Besides min_val and max_val, only need n-2 samples
		n_gotta_sample = n_samples-2 
		samples = np.full(n_gotta_sample, fill_value=float(ref))
		# Sample according to triangular distribution with endpoints given by min_val
		# and max_val, and mode given by ref. Sample pairwise. The first sample in each
		# pair is generated randomly, and the second sample is calculated from the first.
		while (n_gotta_sample >= 2):
			samples[n_gotta_sample-1] = np.random.triangular(left=min_val, mode=ref, 
																right=max_val)
			samples[n_gotta_sample-2] = f_constrain(samples[n_gotta_sample-1])
			n_gotta_sample -= 2
		out = np.concatenate([[min_val], np.sort(samples), [max_val]])
	elif (n_samples == 2):
		out = np.array([min_val, max_val])
	elif (n_samples == 1):
		out = np.array([ref])

	permuted_out = out[np.random.permutation(out.size)]
	return permuted_out