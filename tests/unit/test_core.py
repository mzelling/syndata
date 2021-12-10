import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from syndata.core import ClusterData
from syndata.maxmin import MaxMinClusters

def test_init(setup_clusterdata):
	"""
	Ensure a ClusterData object is initialized with all required attributes.
	"""
	for attr in ['n_clusters', 'n_dim', 'n_samples', 'class_bal', 'cov_geom', 'center_geom',
				 'data', 'labels', 'centers', 'class_sizes', 'cov', 'cov_inv', 'cluster_axes',
				 'cluster_sd', 'scale']:
		exec(r"assert hasattr(setup_clusterdata, '" + attr + "')")


@pytest.fixture()
def to_csv_setup(filename='to_csv_testfile_'):
	"""
	Ensure the .csv file does not exist prior to the test.
	"""
	# Create the .csv file
	files = os.listdir('.')
	if filename in files:
		raise Exception('Test file already present in the working directory.')

	yield filename

	# Delete the created .csv file
	files = os.listdir('.')
	if filename in files:
		os.remove(filename)

def test_to_csv(setup_clusterdata, to_csv_setup):
	"""
	Test creation of a .csv file.
	"""
	setup_clusterdata.to_csv(to_csv_setup)
	X = setup_clusterdata.data

	# load the created .csv file
	files = os.listdir('.')
	assert to_csv_setup in files
	X_loaded = np.loadtxt(to_csv_setup,delimiter=',')

	# make sure the created file gives the same data
	assert np.all(X == X_loaded)



def test_modify_cov_structure(setup_clusterdata):
	"""
	Ensure the function changes the cov structure as desired.
	"""
	cov_structure_before = (setup_clusterdata.cov,
							setup_clusterdata.cov_inv,
							setup_clusterdata.cluster_axes,
							setup_clusterdata.cluster_sd)
	
	new_cov = [(2**2)*np.eye(setup_clusterdata.n_dim) \
				for i in range(setup_clusterdata.n_clusters)]
	new_cov_inv = [(1/(2**2))*np.eye(setup_clusterdata.n_dim) \
				for i in range(setup_clusterdata.n_clusters)]
	new_cluster_axes = setup_clusterdata.n_dim
	new_cluster_sd = [2*np.ones(setup_clusterdata.n_dim) for i in \
					  range(setup_clusterdata.n_clusters)]

	cov_structure_new = (new_cov,new_cov_inv,new_cluster_axes,new_cluster_sd)

	setup_clusterdata.modify_cov_structure(*cov_structure_new)

	assert setup_clusterdata.cov == cov_structure_new[0]
	assert setup_clusterdata.cov_inv == cov_structure_new[1]
	assert setup_clusterdata.cluster_axes == cov_structure_new[2]
	assert setup_clusterdata.cluster_sd == cov_structure_new[3]


@pytest.fixture()
def setup_draw_model(setup_clusterdata):
	"""
	Run ClusterData.draw_model to setup a plot. Close the figure afterwards.
	"""
	(fig, ax) = setup_clusterdata.draw_model()
	yield (fig,ax)
	plt.close(fig)


def test_draw_model(setup_clusterdata, setup_draw_model):
	"""
	Make sure the function draws a figure, but don't check the image.
	"""
	# make sure the desired ranges of the parameter values work

	fig, ax = setup_draw_model
	assert True
	#self,s=30,alpha_max=0.125, sd_max_bare=2,sd_max_density=4, mode='bare', h=0.1,dr=0.1,
	#			   cluster_labels=True)


def test_generate_data():
	pass

def test_add_noise():
	pass



