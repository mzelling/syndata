import pytest
from syndata.maxmin import MaxMinClusters

@pytest.fixture()
def setup_clusterdata():
	"""
	"""
	clusterdata = MaxMinClusters()
	clusterdata.generate_data()

	return clusterdata
