SynData - Generate Synthetic Data Sets with Clusters
============================================================

SynData is a package for generating synthetic data sets. These data sets are particularly helpful for validating 
unsupervised cluster analysis techniques. 

This package differs from other generators in that it automatically
selects appropriate cluster centers and covariance matrices based on high-level geometric input from the
user: for instance, the user decides how much the clusters should overlap, what the ratio between
the largest and smallest cluster volume should be, how elliptical/spherical clusters should be on average, <i>etc.</i> 
The software then automatically creates data sets satisfying these criteria.

### Installation
SynData is hosted on PyPI, so you can install it using pip:
```
pip install syndata
```

### Getting Started
The following code snippet gives a taste of how SynData can be used.
```
import syndata as sd
import matplotlib.pyplot as plt

my_clusters = sd.MaxMinClusters(n_clusters=3,n_dim=2,n_samples=300,aspect_ref=2,
                                dist='exp',alpha_max=0.1,alpha_min=0.075)
X,y = my_clusters.generate_data()

plt.scatter(X[:,0],X[:,1])
plt.gca().set_aspect('equal', adjustable='box')
```
The plots below show the result of running the above code snippet three times. In defining the object `my_clusters`, we specify that we would
like to create data sets with three clusters in two dimensions (`n_clusters=3`, `n_dim=2`). Each data set has 300 samples in total (`n_samples=300`). 
In addition, we set the reference aspect ratio of a cluster to be 2, producing slightly oblong clusters (`aspect_ref=2`). As you increase the reference
aspect ratio, clusters become more elliptical on average. The parameters alpha_max and alpha_min specify the maximum and minimum allowed overlap between
clusters (`alpha_max=0.1`, `alpha_min=0.075`). Finally, we choose an exponential distribution for the clusters (`dist='exp'`).

<div>
<img src="https://github.com/mzelling/syndata/blob/main/my_clusters_test_0.png?raw=true" width="200px">
<img src="https://github.com/mzelling/syndata/blob/main/my_clusters_test_1.png?raw=true" width="200px">
<img src="https://github.com/mzelling/syndata/blob/main/my_clusters_test_2.png?raw=true" width="200px">
</div>


### References
We will host the documentation on a website soon. In the meantime, feel free to make use of the built-in help feature in Python. For example,
type
```
import syndata as sd
help(sd.MaxMinClusters)
```
into a Jupyter Notebook to access documentation for the ``MaxMinClusters`` class (used in the code example above)


### Support
SynData is currently in the late development phase. If you're already using the
package and would like to request support for a problem or tell us about a
new feature you'd like to see implemented, don't hesitate to tell us! Feel free
to submit an issue
<a href="https://github.com/mzelling/syndata/issues/new"> here</a>. We look forward to hearing from you.
