import pytest
import numpy as np

from syndata.core import ClusterData
from syndata.maxmin import MaxMinClusters, MaxMinCov, MaxMinBal, maxmin_sampler

# Test Cases for maxmin_sampler

def test_maxmin_sampler():
	"""
	Make sure the sampling mechanism doesn't break when wrong inputs
	are supplied.
	"""

	# Test cases throwing exceptions
	args_causing_exception = [ # negative vals
							  {'n_samples': 10, 'ref': -2, 'min_val': 1, 'maxmin_ratio': 1.5},
							  {'n_samples': 10, 'ref': 2, 'min_val': -1, 'maxmin_ratio': 1.5},
							  {'n_samples': 10, 'ref': 2, 'min_val': 1, 'maxmin_ratio': -1.5},
							   # zeros vals
							  {'n_samples': 0, 'ref': 2, 'min_val': 1, 'maxmin_ratio': 1.5},
							  {'n_samples': 10, 'ref': 0, 'min_val': 1, 'maxmin_ratio': 1.5},
							  {'n_samples': 10, 'ref': 2, 'min_val': 0, 'maxmin_ratio': 1.5},
							  {'n_samples': 10, 'ref': 2, 'min_val': 1, 'maxmin_ratio': 0},
							   # ref < min
							  {'n_samples': 10, 'ref': 1, 'min_val': 2, 'maxmin_ratio': 1.5},
							   # ref > max
							  {'n_samples': 10, 'ref': 10, 'min_val': 1, 'maxmin_ratio': 1.5},
							   # maxmin_ratio < 1
							  {'n_samples': 10, 'ref': 2, 'min_val': 1, 'maxmin_ratio': 0.7},
							   # maxmin_ratio = 1, ref != min_val
							  {'n_samples': 10, 'ref': 2, 'min_val': 1, 'maxmin_ratio': 1},
							 ]

	with pytest.raises(ValueError):
		for args in args_causing_exception:
			args['f_constrain'] = lambda x: 2*args['ref'] - x
			maxmin_sampler(**args)


	# Test cases with appropriate inputs (randomized)
	args_appropriate_input = []
	max_ref_val = 10; max_min_val = 10

	for i in range(100):
		min_val = np.random.default_rng(seed=i).uniform(0,max_min_val)
		ref = np.random.uniform(min_val, max_ref_val)
		maxmin_ratio = np.random.uniform(ref/min_val, 10*(ref/min_val))
		args_appropriate_input.append(
			{
			# Do the first 10 tests on the edge case n_samples=1
			'n_samples': np.random.choice(np.arange(2,15)) if i>10 else 1,
			'min_val': min_val,
			'ref': ref,
			'maxmin_ratio': maxmin_ratio,
			}
			)

		print('making the args', 'ref', ref, 'min_val', min_val, 'max_val', min_val*maxmin_ratio)

	# Add test case with large sample size
	args_appropriate_input.append({'n_samples': 10000, 'ref': 2, \
									'min_val': 1, 'maxmin_ratio': 3})

	for args in args_appropriate_input:
		args['f_constrain'] = lambda x: 2*args['ref'] - x
		out = maxmin_sampler(**args)
		print(out)
		assert check_maxmin_sampler_output(out, 
				args['f_constrain'])


def check_maxmin_sampler_output(sampled_vals, f_constrain):
	"""
	Check that output satisfies lower and upper bounds.
	Check min, max values are related through the constraint.
	Check that output is sorted.
	"""

	return is_sorted(sampled_vals, order='ascending') \
		and (f_constrain(np.max(sampled_vals) == np.min(sampled_vals))) \
		and (f_constrain(np.min(sampled_vals) == np.max(sampled_vals)))


def is_sorted(vals, order='ascending'):
	"""
	Check if values are sorted.
	"""
	if order=='ascending':
		return np.all(vals[1:] - vals[:-1] >= 0)
	elif order=='descending':
		return np.all(vals[1:] - vals[:-1] <= 0)


# Test Cases for MaxMinCov

def test_init_maxmincov():
	"""
	Make sure that no illicit values can be used to construct MaxMinCov.
	"""

	# appropriate values of attributes
	interior_cases = np.random.uniform(1,10,size=(100,3)) # random appropriate values
	edge_cases = np.concatenate([2-np.eye(3),np.ones(3)[np.newaxis,:]],axis=0) # edge and corner cases
	Z_appropriate = np.concatenate([interior_cases,edge_cases],axis=0)
	args_appropriate = [{'ref_aspect': z[0], 'aspect_maxmin': z[1], 
						'radius_maxmin': z[2]} for z in Z_appropriate]
	for args in args_appropriate:
		my_maxmincov = MaxMinCov(**args)
		for attr in ['ref_aspect','aspect_maxmin','radius_maxmin']:
			assert hasattr(my_maxmincov, attr)

	# inappropriate values of attributes
	Z_inappropriate = np.concatenate([np.ones(3) - 0.5*np.eye(3), (1-0.01)*np.ones(3)[np.newaxis,:]])
	args_inappropriate = [{'ref_aspect': z[0], 'aspect_maxmin': z[1], 
						'radius_maxmin': z[2]} for z in Z_inappropriate]
	with pytest.raises(ValueError):
		for args in args_inappropriate:
			MaxMinCov(**args)


@pytest.fixture()
def setup_maxmincov():
	"""
	Initialize a valid MaxMinCov instance to test its methods.
	"""
	maxmincov = MaxMinCov(ref_aspect=1.5,
						  aspect_maxmin=1.5,
						  radius_maxmin=1.5)
	yield maxmincov


def test_make_cluster_aspects(setup_maxmincov):
	"""
	Make sure that valid cluster aspect ratios are sampled.

	Test the range of acceptable numbers of clusters, and
	make sure setting a seed works.
	"""
	maxmincov = setup_maxmincov

	with pytest.raises(ValueError):
		maxmincov.make_cluster_aspects(0,seed=None)
		maxmincov.make_cluster_aspects(0.99,seed=None)

	# test different numbers of clusters
	for n_clusters in range(1,100):
		cluster_aspects = maxmincov.make_cluster_aspects(n_clusters,seed=None)
		assert np.all(cluster_aspects >= 1)
		assert np.max(cluster_aspects) >= maxmincov.ref_aspect
		assert np.min(cluster_aspects) <= maxmincov.ref_aspect

	# test seed
	seed = 23
	for i in range(10):
		cluster_aspects_new = maxmincov.make_cluster_aspects(2,seed=23)
		# make sure that each successive output is the same as the previous output
		if i >= 1:
			assert np.all(cluster_aspects_new == cluster_aspects_prev)
		cluster_aspects_prev = cluster_aspects_new


def test_make_cluster_radii(setup_maxmincov):
	"""
	Make sure valid cluster radii are sampled.

	Test the range of acceptable inputs, and make sure setting a seed works.
	"""
	maxmincov = setup_maxmincov

	# test appropriate inputs
	interior_cases = np.concatenate([np.arange(1,20+1)[:,np.newaxis], 
									 np.random.uniform(0,10,size=20)[:,np.newaxis], 
									 np.random.choice(np.arange(2,100),size=20)[:,np.newaxis]], 
									 axis=1)
	edge_cases = np.array([[1,1e-3,2], [1,1e-3,1],[2,100,1]])
	Z_appropriate = np.concatenate([interior_cases, edge_cases],axis=0)
	args_appropriate = [{'n_clusters': z[0], 'ref_radius': z[1], 'n_dim': z[2]} for z in Z_appropriate]

	for args in args_appropriate:
		tol = 1e-12
		print(args)
		cluster_radii = maxmincov.make_cluster_radii(**args)
		print(cluster_radii)
		assert np.all(cluster_radii > 0)
		assert (np.min(cluster_radii) <= args['ref_radius'] + tol) and \
				(np.max(cluster_radii) >= args['ref_radius'] - tol)

	# test inappropriate inputs
	with pytest.raises(ValueError):
		maxmincov.make_cluster_radii(n_clusters=0, ref_radius=1, n_dim=10)
		maxmincov.make_cluster_radii(n_clusters=1, ref_radius=0, n_dim=10)
		maxmincov.make_cluster_radii(n_clusters=1, ref_radius=1, n_dim=0)

	# test seeds
	seed = 717
	for i in range(10):
		cluster_radii_new = maxmincov.make_cluster_radii(n_clusters=5,ref_radius=4,n_dim=25, seed=seed)
		if (i >= 1):
			assert np.all(cluster_radii_new == cluster_radii_prev)
		cluster_radii_prev = cluster_radii_new


def test_make_axis_sd(setup_maxmincov):
	"""
	Make sure valid standard deviations are sampled (>0).

	Ensure sure ref_sd is between min and max, and that the maxmin ratio
	equals the desired aspect ratio.
	"""
	maxmincov = setup_maxmincov

	# test appropriate inputs
	interior_cases = np.concatenate([np.arange(2,50+2)[:,np.newaxis], 
								 	 np.random.uniform(0,10,size=50)[:,np.newaxis], 
								 	 np.random.uniform(1,10,size=50)[:,np.newaxis]], 
								 	axis=1)
	edge_cases = np.array([[1,0.5,1.5], [1,0.5,1], [2,0.1,1]])
	Z_appropriate = np.concatenate([interior_cases, edge_cases],axis=0)
	args_appropriate = [{'n_axes': z[0], 'sd': z[1], 'aspect': z[2]} for z in Z_appropriate]

	for args in args_appropriate:
		out = maxmincov.make_axis_sd(**args)
		assert (np.min(out) <= args['sd']) and (np.max(out) >= args['sd']) 

	# test inappropriate inputs
	with pytest.raises(ValueError):
		maxmincov.make_axis_sd(n_axes=0, sd=1, aspect=2)
		maxmincov.make_axis_sd(n_axes=0.5, sd=0, aspect=2)
		maxmincov.make_axis_sd(n_axes=1, sd=1, aspect=0.5)
		maxmincov.make_axis_sd(n_axes=2, sd=1, aspect=-2)
		maxmincov.make_axis_sd(n_axes=2, sd=-1, aspect=2)

	# test seed
	seed = 123
	for i in range(10):
		axis_sd_new = maxmincov.make_axis_sd(n_axes=5,sd=4,aspect=25, seed=seed)
		if (i >= 1):
			assert np.all(axis_sd_new == axis_sd_prev)
		axis_sd_prev = axis_sd_new


def test_make_cov(setup_maxmincov, setup_clusterdata):
	"""
	Make sure axes are orthogonal
	Make sure cov = axis * sd**2 * axis', similar for cov_inv
	"""
	clusterdata = setup_clusterdata
	maxmincov = setup_maxmincov

	# ensure output makes mathematical sense
	for i in range(10):
		(axis, sd, cov, cov_inv) = maxmincov.make_cov(clusterdata)

		for cluster_idx in range(clusterdata.n_clusters):
			# test orthogonality of cluster axes
			assert np.all(np.allclose(axis[cluster_idx] @ np.transpose(axis[cluster_idx]),
					np.eye(axis[cluster_idx].shape[0])))

			# test covariance matrix is correct
			assert np.all(np.allclose(cov[cluster_idx],
					np.transpose(axis[cluster_idx]) @ np.diag(sd[cluster_idx]**2) \
						@ axis[cluster_idx]))

			# test inverse covariance matrix is correct
			assert np.all(np.allclose(cov_inv[cluster_idx], 
					np.transpose(axis[cluster_idx]) @ np.diag(sd[cluster_idx]**(-2)) \
						@ axis[cluster_idx]))


	# test seed
	seed = 123
	for i in range(10):
		cov_structure_new = maxmincov.make_cov(clusterdata, seed=seed)
		if (i >= 1):
			for cluster_idx in range(clusterdata.n_clusters):
				for j in range(4): # iterate through axis, sd, cov, cov_inv
					assert np.all(np.allclose(cov_structure_prev[j][cluster_idx],
										  	  cov_structure_new[j][cluster_idx]))
		# set previous covariance structure for next iteration:
		cov_structure_prev = cov_structure_new


# Test Cases for MaxMinBal

@pytest.fixture(params = np.linspace(1,10,10))
def setup_maxminbal(request):
	return MaxMinBal(request.param)


def test_init_maxminbal(setup_maxminbal):
	"""
	Ensure imbalance ratio is properly specified.
	"""
	maxminbal = setup_maxminbal
	assert maxminbal.imbal_ratio >= 1

	# test input check for inappropriate arguments
	with pytest.raises(ValueError):
		MaxMinBal(imbal_ratio = 0.5)
		MaxMinBal(imbal_ratio = -2)


def test_make_class_sizes(setup_maxminbal,setup_clusterdata):
	"""
	"""
	maxminbal = setup_maxminbal
	clusterdata = setup_clusterdata

	# test with appropriate input
	Z_appropriate = [[500,5],[200,1],[100,2],[1000,10],[1500,3], [100,100]]
	args_appropriate = [{'n_samples': z[0], 'n_clusters': z[1]} for z in Z_appropriate]
	for args in args_appropriate:
		clusterdata.n_samples = args['n_samples']
		clusterdata.n_clusters = args['n_clusters']
		out = maxminbal.make_class_sizes(clusterdata)
		assert np.issubdtype(out.dtype, np.integer) and np.all(out >= 1) and \
				(np.sum(out) == args['n_samples'])

	# test with inappropriate input
	Z_inappropriate = [[500,0],[0,10],[100,-1],[-0.5,5],[10,11]]
	args_inappropriate = [{'n_samples': z[0], 'n_clusters': z[1]} for z in Z_inappropriate]
	for args in args_inappropriate:
		with pytest.raises(ValueError):
			clusterdata.n_clusters = args['n_clusters']
			clusterdata.n_samples = args['n_samples']
			maxminbal.make_class_sizes(clusterdata)


def test_float_to_int(setup_maxminbal):
	"""
	float_class_sz, n_samples
	"""
	maxminbal = setup_maxminbal

	# test appropriate inputs
	for float_class_sz, n_samples in [(np.array([23.2, 254.7, 0.1, 35.6]), 100), \
									  (np.array([0.2, 0.7, 0.1, 0.5]), 10),
									  (np.array([2.5,1.5,5.2]), 3),
									  (np.array([0.5]), 1)]:
		out = maxminbal.float_to_int(float_class_sz,n_samples)
		print(len(float_class_sz), float_class_sz, n_samples)
		assert (np.sum(out) == n_samples) and (np.all(out >= 1)) \
				and np.issubdtype(out.dtype,np.integer)

	# test inputs that should be left unchanged
	assert np.all(maxminbal.float_to_int(np.array([5,10,25,7]), 5+10+25+7) \
					== np.sort(np.array([5,10,25,7])))

	# test inappropriate inputs
	for float_class_sz, n_samples in [(np.array([0.5,1.5]), 1),
									  (np.array([0.5,1.5]), 0),
									  (np.array([2.5,1.5,5.2]), 2)]:
		with pytest.raises(ValueError):
			maxminbal.float_to_int(float_class_sz,n_samples)



# Test Cases for MaxMinClusters





