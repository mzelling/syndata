"""
This module provides the core object-oriented framework for generating 
synthetic data sets with clusters. The classes contained here are mainly 
abstract superclasses. They require subclasses to concretely implement 
much of the specified functionality.

CLASSES AND METHODS

	ClusterData: top-level object for generating data
		__init__(self, n_clusters, n_dim, n_samples, class_bal, cov_geom, 
				 center_geom, data_dist, ...)
		to_csv(self, filename)
		generate_data(self, ...)
		add_noise(self, data, snr, p_over_n, ...)

	CovGeom: sample cluster shapes
		__init__(self)
		make_cov(self, clusterdata)
		make_orthonormal_axes(self, n_dim, n_axes)

	CenterGeom: place cluster centers
		__init__(self)
		make_centers(self, clusterdata)

	ClassBal: sample number of points for each cluster
		__init__(self)
		make_class_sizes(self, clusterdata)

	DataDist: draw synthetic data for each cluster
		__init__(self)
		sample(self, clusterdata)
		sample_cluster(self, cluster_size, mean, axes, sd)
"""


import numpy as np
import scipy.stats as stats

class ClusterData:
	"""
	Generates data sets with clusters according to user specifications.

	Parameters
	----------
	n_clusters : int
		Number of clusters
	n_dim : int
		Dimensionality of data points
	n_samples : int
		Total number of data points 
	class_bal : ClassBal object
		Defines relative class sizes (samples sizes for each cluster)
	cov_geom : CovGeom object
		Defines cluster covariance structures
	center_geom : CenterGeom object
		Defines placement of cluster centers
	data_dist : DataDist object
		Defines probability distribution for data
	scale : float, optional
		Sets reference length scale for simulated data

	Attributes
	----------
	centers : 
	class_sizes : 
	cluster_axes :
	cluster_sd :
	cov : 
	cov_inv :
	data :
	labels :
	scale :

	"""
		
	def __init__(self, n_clusters, n_dim, n_samples, class_bal, cov_geom, 
		center_geom, data_dist, scale=1.0):
		"""
		Constructs a ClusterData object.
		"""
		self.n_clusters = n_clusters
		self.n_dim = n_dim
		self.n_samples = n_samples
		self.class_bal = class_bal
		self.cov_geom = cov_geom
		self.center_geom = center_geom
		self.data_dist = data_dist
		self.data = None
		self.labels = None
		self.centers = None
		self.class_sizes = None
		self.cov = None
		self.cov_inv = None
		self.cluster_axes = None
		self.cluster_sd = None
		self.scale = scale

	def to_csv(self, filename):
		"""
		Writes data to a csv file.
		"""
		if (self.data is None):
			raise Exception('No data has been generated. Use ClusterData.generate_data to generate data.')
		else:
			np.savetxt(filename, self.data, delimiter=',')
		
		
	def generate_data(self,add_noise_vars=False, snr=None, p_over_n=None):
		"""
		Generates a dataset with clusters according to a ClusterData object.
		
		Parameters
		----------
		self : ClusterData
			The underlying data generator
		add_noise_vars : bool
			If true, add noise features to the data. Need to set snr and/or
			p_over_n to control the amount of noise added. If both snr and
			p_over_n are given, add the least number of noise features that
			meets or exceeds both thresholds (snr and the p_over_n).
		snr : float
			Set the ratio between the number of meaningful features and the
			number of noise features. Only relevant when add_noise_vars=True.
		p_over_n : float
			Set the ratio between the total number of features (p) and the
			number of samples (n). Only relevant when add_noise_vars=True.

		Returns
		-------
		X : ndarray
			Data matrix whose rows are the data points.
		y : ndarray
			Vector of cluster labels.
		"""
		print('Compute class sizes...')
		self.class_sizes = self.class_bal.make_class_sizes(self)

		print('Compute covariance structures...')
		axes, sd, cov, cov_inv = self.cov_geom.make_cov(self)
		self.cluster_axes = axes
		self.cluster_sd = sd
		self.cov = cov
		self.cov_inv = cov_inv

		print('Place cluster centers...')
		self.centers = self.center_geom.place_centers(self)

		print('Sample data...')
		self.data, self.labels = self.data_dist.sample(self)

		print('Success!')

		if add_noise_vars:
			self.data = self.add_noise(self.data,snr=snr,p_over_n=p_over_n)

		return (self.data, self.labels)


	def add_noise(self, data, snr, p_over_n, model='unif', margin=0.1, 
		print_warning=True):
		"""
		Add noise features to given data.

		Choosing snr determines how "sparse" the resulting data is, whereas 
		p_over_n determines how high-dimensional the data is. If only one of snr
		and p_over_n is given, add the exact number of noise features required 
		to meet the threshold. If both snr and p_over_n are given, add the least 
		number of noise features that meets or exceeds both thresholds. Thus, the 
		resulting data is as noisy or noisier than required by either threshold.

		Parameters
		----------
		self : ClusterData
			The underlying data generator
		data :
			The data to which noise features are added.
		snr : float
			Set the ratio between the number of meaningful features and the
			number of noise features.
		p_over_n : float
			Set the ratio between the total number of features (p) and the
			number of samples (n).
		model : str
			Determine how noise features are calculated.

		Returns
		-------
		out : ndarray
			Input data with added noise features.

		"""

		n_dim = data.shape[1] 
		n_samples = data.shape[0]

		# get lower and upper feature limits
		low = np.min(data,axis=0)
		high = np.max(data,axis=0)

		if (not snr) and (not p_over_n):
			raise Exception('Must provide either snr or p_over_n to determine ' +
							'number of noise features to add.')
		else:
			if snr and not p_over_n:
				# add enough noise features to meet snr threshold
				n_noise_vars = int(np.ceil(n_dim / snr))
				noise_vars = np.random.uniform((1-margin)*low,(1+margin)*high,
											  size=(n_samples,n_noise_vars))
			if p_over_n and not snr:
				# add enough noise features to meet p_over_n threshold
				if n_dim/n_samples <= p_over_n:
					n_noise_vars = int(np.ceil(p_over_n * n_samples)) - n_dim
					noise_vars = np.random.uniform((1-margin)*low,(1+margin)*high,
											  		 size=(n_samples,n_noise_vars))
				else:
					noise_vars = []
					if print_warning:
						print('Warning: data already sufficiently high-dimensional,' +
						  	  ' no noise features added.')
			if snr and p_over_n:
				# add enough noise features to meet both snr and p_over_n thresholds
				n_noise_vars = np.max([int(np.ceil(n_dim/snr)), 
									   int(np.ceil(n_samples*p_over_n))-n_dim])
				noise_vars = np.random.uniform((1-margin)*low,(1+margin)*high,
											   size=(n_samples,n_noise_vars))

			# return data with noise features added
			return np.concatenate([data,noise_vars],axis=1)


class CovGeom:
	"""
	Specifies the covariance structure for ClusterData.
	"""
	
	def __init__(self):
		raise NotImplementedError('Cannot instantiate abstract CovGeom.'+
								  ' Choose a provided implementation,'+
								  ' e.g. maxmin.MaxMinCov, or code your'+
								  ' own algorithm for defining cluster'+
								  ' covariance structure.')

	def make_cov(self, clusterdata):
		raise NotImplementedError('Cannot sample covariance structure from'+
								  ' abstract CovGeom. Choose a provided' 
								  ' implementation, e.g. maxmin.MaxMinCov' +
								  ', or code your own algorithm for defining'
								  ' cluster covariance structure.')

	def make_orthonormal_axes(self, n_dim, n_axes):
		"""
		Samples orthonormal axes for cluster generation.
		Creates n_axes orthonormal axes in n_dim-dimensional space.

		Parameters
		----------
		n_dim : int
			Dimensionality of the data space
		n_axes : int
			Number of axes to generate

		Returns
		-------
		out : (n_axes, n_dim) ndarray
			Each row is an axis.
		"""
		ortho_matrix = stats.ortho_group.rvs(n_dim)
		return ortho_matrix[:n_axes, :]



class CenterGeom:
	"""
	Defines placement of cluster centers for ClusterData.
	"""

	def __init__(self):
		raise NotImplementedError('Cannot instantiate abstract CenterGeom.'+
								  ' Choose a provided implementation,'+
								  ' e.g. centers.BoundedSepCenters, or code your'+
								  ' own algorithm for placing cluster centers.')

	def make_centers(self, clusterdata):
		raise NotImplementedError('Cannot sample cluster centers from abstract'+
								  ' CenterGeom. Choose a provided implementation,'+
								  ' e.g. centers.BoundedSepCenters, or code your'+
								  ' own algorithm for placing cluster centers.')



class ClassBal:
	"""
	Specifies the relative cluster sizes for ClusterData.
	"""

	def __init__(self):
		raise NotImplementedError('Cannot instantiate abstract ClassBal.' + 
								  ' Choose a provided implementation, e.g.'+
								  'maxmin.MaxMinBal, or code your own '+
								  'algorithm for sampling class sizes.')

	def make_class_sizes(self, clusterdata):
		raise NotImplementedError('Cannot sample class sizes from abstract'+
								  ' ClassBal. Choose a provided implementation,'+
								  ' e.g. maxmin.MaxMinBal, or code your own '+
								  'algorithm for sampling class sizes.')


class DataDist:
	"""
	Defines data probability distribution for ClusterData.
	"""

	def __init__(self):
		raise NotImplementedError('Cannot instantiate abstract DataDist. Choose '+
								  ' a provided implementation, e.g. GaussianDist,' +
								  ' or code your own data distribution.')

	def sample(self, clusterdata):
		n_clusters = clusterdata.n_clusters
		n_samples = clusterdata.n_samples
		n_dim = clusterdata.n_dim
		class_sizes = clusterdata.class_sizes
		centers = clusterdata.centers

		axes = clusterdata.cluster_axes
		sd = clusterdata.cluster_sd

		X = np.full(shape=(n_samples, n_dim), fill_value=np.nan)
		y = np.full(n_samples, fill_value=np.nan).astype(int)

		start = 0
		for i in range(n_clusters):
			end = start + class_sizes[i]
			# Set class label
			y[start:end] = i
			# Sample data
			X[start:end,:] = self.sample_cluster(cluster_size=class_sizes[i], 
										  mean=centers[i], axes=axes[i],
										  sd=sd[i])
			start = end

		return (X, y)

	def sample_cluster(self, cluster_size, mean, axes, sd):
		raise NotImplementedError('Cannot sample cluster from abstract DataDist. '+
								  'Choose a provided implementation, e.g. '
								  'GaussianDist, or code your own data distribution.')