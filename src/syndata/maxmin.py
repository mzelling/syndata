from .core import ClusterData, CovGeom, ClassBal
from .centers import BoundedSepCenters
from .distributions import GaussianData, ExpData, tData
import numpy as np
import scipy.stats as stats

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
		ClusterData.__init__(self, n_clusters,n_dim,n_samples,class_bal,cov_geom,
						 	 center_geom,data_dist,scale)


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
		out : a tuple (axis, sd, cov, cov_inv), where cov and cov_inv are lists of ndarrays
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

			# can potentially not construct cov, cov_inv here; and instead do it only upon request
			cov.append(np.transpose(axes) @ np.diag(axis_sd**2) @ axes)
			cov_inv.append(np.transpose(axes) @ np.diag(1/axis_sd**2) @ axes)

		out = (axis, sd, cov, cov_inv)
			
		return out


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


def maxmin_sampler(n_samples, ref, min_val, maxmin_ratio, f_constrain):
	"""
	Generates samples around a reference value, with a fixed ratio between the maximum
	and minimum sample. Samples pairwise to enforce a further constraint on the samples.
	For example, the geometric mean of the samples can be specified.

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
	return out