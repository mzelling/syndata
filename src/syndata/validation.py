import numpy as np
import scipy.stats as stats
from .maxmin import MaxMinClusters
from .centers import FastHighDimCenters

class TestStream():
    """
    Iterator that randomly generates synthetic data.
    """
    
    def __init__(self,n_tests=20,seed=None,return_labels=False, sample_dim=lambda: np.random.exponential(25)):
        """
        Instantiate TestStream object.
        """
        self.n_tests = n_tests
        self.iter_count = 0
        self.return_labels = return_labels
        self.seed = seed
        self.sample_dim = sample_dim
        
    def __iter__(self):
        """
        Return iterator.
        """
        return self
        
    def __next__(self):
        """
        Sample next data set.
        """
        if (self.iter_count == self.n_tests):
            raise StopIteration
        
        clusterdata = MaxMinClusters()
        self.randomize_parameters(clusterdata, 
                                        seed=(None if not self.seed else self.seed + 4783*self.iter_count) % 7727)
        
        if (clusterdata.n_dim >= 20):
            clusterdata.center_geom = FastHighDimCenters()

        X, y = clusterdata.generate_data(verbose=False, 
                                         seed=(None if not self.seed else self.seed + 3529*self.iter_count) % 7727)
        self.iter_count += 1
        
        attr_dict = {'k': clusterdata.n_clusters, 'p': clusterdata.n_dim, 'n': clusterdata.n_samples,
                     'alpha_max': clusterdata.alpha_max_obs, 'alpha_min': clusterdata.alpha_min_obs}
        
        if self.return_labels:
            return (X,attr_dict,y)
        else:
            return (X,attr_dict)
    
    def randomize_parameters(self, clusterdata, seed=None, max_k=30, min_pts_per_cluster=50, 
                             max_sample_sz=2000):
        """
        Randomize parameters of ClusterData object.
        """
        if seed:
            np.random.seed(seed)
            
        clusterdata.n_clusters = np.random.choice(a=np.arange(2,max_k+1))
        clusterdata.n_samples = np.max([clusterdata.n_clusters*min_pts_per_cluster, 
                                        100*np.random.choice(np.arange(1,int(max_sample_sz/100)))])
        clusterdata.n_dim = np.max([2, int(self.sample_dim())]) #int(np.random.exponential(ref_dim))
        clusterdata.alpha_max = np.random.uniform(0.01,0.2)
        clusterdata.alpha_min = 1e-5
        clusterdata.aspect_ref = np.random.uniform(1,5)
        clusterdata.radius_maxmin = np.random.uniform(1,5)
        clusterdata.aspect_maxmin = np.random.uniform(1,5)
        clusterdata.imbal_maxmin = np.random.uniform(1,5)