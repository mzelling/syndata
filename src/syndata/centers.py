"""
This module provides algorithms for sampling cluster centers for a 
ClusterData object.

CLASSES AND METHODS

	BoundedSepCenters : sample cluster centers obeying separation constraints
		__init__(self, min_sep, max_sep, ...)
		place_centers(self, clusterdata)
		
"""

import numpy as np
import scipy.stats as stats
from .core import CenterGeom
from scipy.spatial.distance import mahalanobis, pdist, squareform
from scipy.stats import chi2
from numpy.linalg import qr




class BoundedSepCenters(CenterGeom):
	"""
	Sample cluster centers by specifying maximum and minimum separation 
	between centers.

	The minimum separation constraint ensures that no clusters are too 
	close together. On the other hand, the maximum separation constraint 
	guarantees that no cluster is too far away from all other clusters

	Separation is expressed on a scale derived from Mahalanobis distance. 
	A separation of 1.0 means that two clusters are just touching (when 
	each cluster is considered as a hard sphere filling 95% of the 
	cluster's probability mass). Lower values of separation imply overlap 
	between the clusters; higher values cleanly separate the clusters. 

	Note that the above interpretation of separation value is strictly 
	valid only for Gaussian clusters; for clusters with other data 
	distributions, separation values below or above 1.0 may be necessary
	for two clusters to be "just touching."

	Attributes
	----------
	min_sep : float (positive)
		Minimum separation between clusters. Value 1.0 means "just touching",
		lower values allow overlaps. See discussion in the class description.
	max_sep : float (positive)
		Maximum separation between clusters. Value 1.0 means "just touching",
		lower values imply overlap. See discussion in the class description.
	packing : float between 0 and 1, default=0.1
		Ratio of total cluster volume to sampling volume, used in rejection
		sampling. Here, cluster volume is computed from the maximum standard
		deviation among cluster shapes.

	"""
	
	def __init__(self, alpha_min, alpha_max, packing=.1):
		"""
		Create a BoundedSepCenters object.

		Parameters
		----------
		min_sep : float (positive)
			Minimum separation between clusters. Value 1.0 means "just touching",
			lower values allow overlaps. See discussion in the class description.
			max_sep : float (positive)
		Maximum separation between clusters. Value 1.0 means "just touching",
			lower values imply overlap. See discussion in the class description.
		packing : float between 0 and 1, default=0.1
			Ratio of total cluster volume to sampling volume, used in rejection
			sampling. Here, cluster volume is computed from the maximum standard
			deviation among cluster shapes.

		Returns
		-------
		out : BoundedSepCenters
			Object for sampling cluster centers obeying separation constraints.
		"""
		self.alpha_min = alpha_min
		self.alpha_max = alpha_max
		self.packing = packing
		

	def place_centers(self, clusterdata, seed=None, verbose=True):
		"""
		Place cluster centers sequentially with rejection sampling.

		Sample possible cluster centers uniformly within a box. To achieve 
		desired separation between clusters, only accept new centers whose
		separation from all other clusters is greater than self.min_sep (ensure
		no clusters overlap too much). In addition, only accept new centers
		whose separation from at least one other cluster is less than 
		self.max_sep (ensure no cluster is far away from all other clusters).

		Parameters
		----------
		self : BoundedSepCenters
			This instance of BoundedSepCenters
		clusterdata : ClusterData
			The data generator

		Returns
		-------
		centers : ndarray
			Cluster centers arranged in a matrix. Each row is a cluster center.
		"""

		np.random.seed(seed)

		n_clusters = clusterdata.n_clusters
		n_dim = clusterdata.n_dim

		# find the maximum sd
		cluster_sd = clusterdata.cluster_sd
		max_sd = 2*np.prod(np.array([(np.prod(sd_vec))**(1/n_dim) for sd_vec in cluster_sd]))**(1/n_dim)
		#max_sd = np.max(np.array([np.max(sd_vec) for sd_vec in cluster_sd]))

		cov_inv = clusterdata.cov_inv
		
		centers = np.zeros(shape=(n_clusters, n_dim))

		total_cluster_vol = n_clusters * ((4*max_sd)**n_dim)
		sampling_vol = total_cluster_vol / self.packing
		sampling_width = sampling_vol**(1/n_dim)

		# compute reference chi square value (take 0.95 quantile)
		q_chi_sq_repel = chi2.ppf(1-self.alpha_max, df=n_dim)
		q_chi_sq_attract = chi2.ppf(1-self.alpha_min, df=n_dim)
		
		for new_center in range(n_clusters):
			accept = False
			while not accept:
				proposed_center = sampling_width * np.random.uniform(size=n_dim)
				far_enough = True # Need to uphold minimum distance to other centers.
				close_enough = False # New center shouldn't be far away from all centers.

				# Check distances to previously selected centers
				for prev_ctr in range(new_center):
					d_1 = mahalanobis(proposed_center, centers[prev_ctr], 
										 cov_inv[new_center])
					d_2 = mahalanobis(proposed_center, centers[prev_ctr], 
										 cov_inv[prev_ctr])

					# quantile corresponding to chi^2 distribution with df=n_dim
					q_sq = 1/(((1/d_1) + (1/d_2))**2)
					# min_sep = q_sq_(1-alpha_min), max_sep = q_sq_(1-alpha_max),
					# e.g. min_sep = 0.95, max_sep = 0.9999

					if (q_sq < q_chi_sq_repel):
						far_enough = False 
						break

					if (q_sq < q_chi_sq_attract):
						close_enough = True
				
				if ((far_enough and close_enough) or (new_center == 0)):
					accept = True
					#UNCOMMENT FOR DEBUGGING PURPOSES:
					#if not (new_center == 0): 
					#	print(chi_1, chi_2)
					centers[new_center,:] = proposed_center
					if verbose: print('\t' + str(1+new_center) + 
						'/' + str(n_clusters) + ' placed!')
				
		return(centers)



class FastHighDimCenters(CenterGeom):
	"""
	Sample cluster centers fast for data sets with high intrinsic dimensionality.
	"""
	
	def __init__(self, alpha_min=1e-5,alpha_max=0.05, 
				 packing_min=1e-2, packing_max=2,thold=1.05, max_runs=5):
		"""
		Instantiate a FastHighDimCenters object.
		"""
		self.max_runs = max_runs
		self.log_thold = np.log(thold)
		self.log_packing_min = np.log(packing_min)
		self.log_packing_max = np.log(packing_max)
		self.alpha_min = alpha_min
		self.alpha_max = alpha_max
		self.alpha_min_obs = None
		self.alpha_max_obs = None
	
	def place_centers(self, clusterdata, seed=None, verbose=False):
		"""
		Sample cluster centers repeatedly for different values of the
		packing parameter. Take the centers that best fit the desired
		minimum and maximum overlap values.
		"""
		# make adjustments for the dimensionality
		self.log_packing_min = (self.log_packing_min)*(clusterdata.n_dim/2)
		self.log_packing_max = (self.log_packing_max)*(clusterdata.n_dim/2)
		self.log_thold = (self.log_thold)*clusterdata.n_dim*2
		
		self.binary_search_packing(clusterdata,self.log_packing_min,self.log_packing_max,
								   attempts_remaining=self.max_runs,
								   verbose=verbose)
		return clusterdata.centers
		
	def overlap_error(alpha_min_obs,alpha_max_obs,alpha_min_desired,alpha_max_desired,
					  inf_proxy=100,verbose=True):
		"""
		Compute how far observed overlaps are outside the desired box [alpha_min,alpha_max]
		in log space and penalize the overhang linearly.
		"""
		if verbose:
			print('alpha_min_obs',alpha_min_obs,'alpha_max_obs',alpha_max_obs)

		if (alpha_min_obs == 0):
			diff2 = inf_proxy
		else:
			diff2 = np.log(alpha_min_desired) - np.log(alpha_min_obs)

		if (alpha_max_obs == 0):
			diff1 = inf_proxy
		else:
			diff1 = np.log(alpha_max_obs) - np.log(alpha_max_desired)
		
		return diff1*(diff1>0) + diff2*(diff2>0)
		
	def binary_search_packing(self, clusterdata, log_packing_min, 
							  log_packing_max, attempts_remaining,
							  n_restarts=5,verbose=True):
		"""
		"""
		avg_log_packing = (log_packing_min+log_packing_max)/2
		
		if verbose:
			print('log packing ->', avg_log_packing)
		
		seed_list = np.random.choice(n_restarts*1000,size=n_restarts,replace=False)
		# randomly choose cluster centers with current packing parameter
		min_error = np.Inf
		min_idx = None

		cov_save_list = []
		
		for i in range(n_restarts):
			# place cluster centers
			cov_save_list.append((clusterdata.cov,clusterdata.cov_inv, 
									clusterdata.cluster_axes, clusterdata.cluster_sd))
			self.attempt_to_place_centers(avg_log_packing, clusterdata, seed=seed_list[i],
										  modify_clusterdata=True)
			
			# report error
			overlap_mat = CenterGeom.overlap_matrix(clusterdata.centers,
															clusterdata.cov_inv)
			alpha_min = CenterGeom.alpha_min_obs(overlap_mat)
			alpha_max = CenterGeom.alpha_max_obs(overlap_mat)
			error = FastHighDimCenters.overlap_error(alpha_min_obs=alpha_min,
													 alpha_max_obs=alpha_max,
													 alpha_min_desired=self.alpha_min,
													 alpha_max_desired=self.alpha_max,
													 verbose=verbose)
			# update best error
			if (error < min_error):
				min_idx = i
				min_error = error
			
		# continue binary search
		if ((log_packing_max - log_packing_min) >= self.log_thold) and (attempts_remaining>0):
			cov_save_list = []
			if (alpha_max > self.alpha_max):
				# go left -> reduce packing
				self.binary_search_packing(clusterdata,log_packing_min,avg_log_packing,
										   n_restarts=n_restarts,
										   attempts_remaining=attempts_remaining-1,verbose=verbose)
			else:
				# go right -> increase packing
				self.binary_search_packing(clusterdata,avg_log_packing,log_packing_max,
										   n_restarts=n_restarts,
										   attempts_remaining=attempts_remaining-1,verbose=verbose)
		else:
			clusterdata.modify_cov_structure(*cov_save_list[min_idx])
			self.attempt_to_place_centers(log_packing=avg_log_packing, clusterdata=clusterdata, 
												 seed=seed_list[min_idx], verbose=verbose,
												 modify_clusterdata=True)
			# recompute overlap
			overlap_mat = CenterGeom.overlap_matrix(clusterdata.centers,
															clusterdata.cov_inv)
			self.alpha_min_obs = CenterGeom.alpha_min_obs(overlap_mat)
			self.alpha_max_obs = CenterGeom.alpha_max_obs(overlap_mat)
			
			# report packing
			self.log_packing = avg_log_packing
		
	
	def attempt_to_place_centers(self, log_packing, clusterdata, seed=None, verbose=True,
								 modify_clusterdata=True):
		"""
		Sample cluster centers and adjust covariance matrices.
		"""
		if seed:
			np.random.seed(seed)
		
		n_clusters = clusterdata.n_clusters
		n_dim = clusterdata.n_dim

		# find reference sd for the whole data
		cluster_sd = clusterdata.cluster_sd
		log_ref_sd = np.mean(np.array([np.mean(np.log(sd_vec)) for sd_vec in cluster_sd]))

		centers = np.zeros(shape=(n_clusters, n_dim))
	
		#total_cluster_vol = n_clusters * ((4*ref_sd)**n_dim)
		#print('total cluster vol', total_cluster_vol)
		#sampling_vol = total_cluster_vol / self.packing
		#sampling_width = sampling_vol**(1/n_dim)
		
		log_total_cluster_vol = n_dim*(np.log(4) + log_ref_sd)
		log_sampling_vol = log_total_cluster_vol - log_packing
		sampling_width = np.exp(log_sampling_vol/n_dim)

		centers = sampling_width * np.random.uniform(size=(n_clusters,n_dim))
		
		# adjust covariance structures to accommodate the new centers
		if modify_clusterdata:
			clusterdata.centers = centers
			FastHighDimCenters.adjust_cov(clusterdata)
		
		return centers

	
	def adjust_cov(clusterdata):
		"""
		Adjust covariance matrices after sampling cluster centers.
		"""

		# compute pairwise distances between cluster centers
		center_array = clusterdata.centers
		center_pw_dist = squareform(pdist(center_array))
		center_pw_dist[np.arange(center_pw_dist.shape[0]),np.arange(center_pw_dist.shape[0])] = np.Inf
		

		# for each cluster center, compute minimum distances to the nearest other center
		min_dist = np.min(center_pw_dist,axis=1)

		# compute volume for each covariance matrix
		vol_array = np.zeros(clusterdata.n_clusters)
		for k in range(clusterdata.n_clusters):
			vol_array[k] = np.prod(4*clusterdata.cluster_sd[k])

		# sort covariance matrices by volume, increasing order
		vol_order = np.argsort(vol_array)
		# sort cov in order of minimum distance to nearest other center, increasing order
		cov_order = np.argsort(np.argsort(min_dist)[np.arange(clusterdata.n_clusters)])

		new_cov_list = []
		new_cov_inv_list = []
		new_sd_list = []
		new_axes_list = []
		# assign smallest-volume covariance matrix to smallest-min-distance centers
		for k in range(clusterdata.n_clusters):
			#print('\n\nDoing center ', str(k), '...')
			ctr = center_array[k,:]
			cov = clusterdata.cov[vol_order[cov_order[k]]]
			axes = clusterdata.cluster_axes[vol_order[cov_order[k]]] 
			axes_sd = clusterdata.cluster_sd[vol_order[cov_order[k]]]

			# find vector corresponding to minimum distance, and normalize
			closest_ctr_idx = np.argmin(center_pw_dist[k,:])
			#print('closest other center:', closest_ctr_idx)
			min_dist_vec = center_array[closest_ctr_idx,:] - ctr
			min_dist_vec = min_dist_vec/np.sqrt(np.dot(min_dist_vec,min_dist_vec))
			#print('minimum distance vector:', min_dist_vec)

			# find longest principal axis
			longest_axis_idx = np.argmax(axes_sd)
			longest_axis = axes[longest_axis_idx,:]
			#print('longest axis:', longest_axis)

			# project longest axis away from the min dist vector
			projection_coef = np.dot(min_dist_vec,longest_axis)
			#print('proj coef:', projection_coef)
			new_longest_axis = longest_axis - projection_coef*min_dist_vec 
			#print('new longest axis:', new_longest_axis)

			# Step 1: replace longest axis by the new longest axis, place longest axis first
			# Comment: placing longest axis first is important for QR decomposition-- otherwise,
			# the decomposition will not leave the longest axis intact.
			axes[longest_axis_idx,:] = axes[0,:]
			axes[0,:] = new_longest_axis

			# Step 2: calculate QR decomp of the resulting matrix, use as new axes
			new_axes = np.transpose(qr(np.transpose(axes))[0])

			# Step 3: put together new cov matrix from new axes and old standard deviations
			axes_sd = np.sort(axes_sd)[::-1]
			cov = np.transpose(new_axes) @ np.diag(axes_sd**2) @ new_axes
			
			cov_inv = np.transpose(new_axes) @ np.diag(axes_sd**(-2)) @ new_axes

			# Step 4: reset cov matrix in cluster data
			new_cov_list.append(cov)
			new_cov_inv_list.append(cov_inv)
			new_sd_list.append(axes_sd)
			new_axes_list.append(new_axes)
			
		clusterdata.modify_cov_structure(new_cov_list, new_cov_inv_list, new_axes_list, new_sd_list)