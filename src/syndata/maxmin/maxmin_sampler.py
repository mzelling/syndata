import numpy as np

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