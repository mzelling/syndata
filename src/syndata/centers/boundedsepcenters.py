import numpy as np
import scipy.stats as stats
from ..core import CenterGeom
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

class BoundedSepCenters(CenterGeom):
	"""
	Sample cluster centers by specifying maximum and minimum separation between centers.

	Cluster centers are generated using rejection sampling. Separation is computed with the
	Mahalanobis distance and is expressed in units of four times the standard deviation. 
	As a 1D example, a separation of 1.0 corresponds to two Gaussians that overlap at their
	2.5% quantiles.

	Parameters
	----------

	Attributes
	----------
	min_sep : float
		Minimum separation between clusters, in units of four times the standard deviation.
	max_sep : float
		Maximum separation between clusters, in units of four times the standard deviation.
	packing : float, default=0.1
		The ratio of total cluster volume over sampling volume. Cluster volume is computed
		from the maximum standard deviation among cluster shapes.
	"""
	
	def __init__(self, min_sep, max_sep=10, packing=.1):
		"""
		Creates a BoundedSepCenters object.
		"""
		self.min_sep = min_sep
		self.max_sep = max_sep
		self.packing = packing
		

	def place_centers(self, clusterdata):
		"""
		Place cluster centers sequentially with rejection sampling.

		Sample uniformly within a box and only accept new centers whose
		separation to all other clusters is greater than self.min_sep and 
		whose separation to at least one other cluster is less than 
		self.max_sep.
		"""

		# find the maximum sd
		cluster_sd = clusterdata.cluster_sd
		#max_sd = np.mean(np.array([np.mean(sd_vec) for sd_vec in cluster_sd]))
		max_sd = np.max(np.array([np.max(sd_vec) for sd_vec in cluster_sd]))

		cov_inv = clusterdata.cov_inv
		
		n_clusters = clusterdata.n_clusters
		n_dim = clusterdata.n_dim
		centers = np.zeros(shape=(n_clusters, n_dim))

		total_cluster_vol = n_clusters * ((4*max_sd)**n_dim)
		sampling_vol = total_cluster_vol / self.packing
		sampling_width = sampling_vol**(1/n_dim)

		# compute reference chi square value (take 0.95 quantile)
		chi_sq_ref = chi2.ppf(0.95, df=n_dim)
		
		for new_center in range(n_clusters):
			accept = False
			while not accept:
				proposed_center = sampling_width * np.random.uniform(size=n_dim)
				far_enough = True # Need to uphold minimum distance to other centers.
				close_enough = False # New center shouldn't be far away from all centers.

				# Check distances to previously selected centers
				for prev_ctr in range(new_center):
					chi_1 = mahalanobis(proposed_center, centers[prev_ctr], 
										 cov_inv[new_center]) ** 2
					chi_2 = mahalanobis(proposed_center, centers[prev_ctr], 
										 cov_inv[prev_ctr]) ** 2
					if (np.sqrt(np.min([chi_1, chi_2])) <= 2*np.sqrt(chi_sq_ref)*self.min_sep):
						far_enough = False 
						break

					if (np.sqrt(np.min([chi_1, chi_2])) <= 2*np.sqrt(chi_sq_ref)*self.max_sep):
						close_enough = True
				
				if ((far_enough and close_enough) or (new_center == 0)):
					accept = True
					#UNCOMMENT FOR DEBUGGING PURPOSES:
					#if not (new_center == 0): 
					#	print(chi_1, chi_2)
					centers[new_center,:] = proposed_center
					print('\t' + str(1+new_center) + '/' + str(n_clusters) + ' placed!')
				
		return(centers)