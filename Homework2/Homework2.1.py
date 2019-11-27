#!/usr/bin/env python
# coding: utf-8

# In[2]:


from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, mixture
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
import warnings
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')
np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 1797

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\tNMI\tHomogeneity\tCompleteness')

# Kmeans


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    # print(estimator.cluster_centers_)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))

# AffinityPropagation
def bench_AffinityPropagation(estimator, name, data):
    t0=time()
    estimator.fit(data)
    # print(estimator.cluster_centers_)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), estimator.n_iter_ ,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))

# Meanshift
def bench_MeanShift(estimator, name, data):
    t0=time()
    estimator.fit(data)
    # print(estimator.cluster_centers_)
    print('%-9s\t%.2fs\t%s\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), 'unknown',
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))



def bench_SpectralClustering(estimator, name, data):
    t0=time()
    estimator.fit(data)
    print(estimator.labels_)
    print('%-9s\t%.2fs\t%s\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), 'unknown',
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))



def bench_AgglomerativeClustering(estimator, name, data):
    t0=time()
    estimator.fit(data)
    # print(estimator.labels_ )
    print('%-9s\t%.2fs\t%s\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), 'unknown',
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))

def bench_DBSCAN(estimator, name, data):
    t0=time()
    estimator.fit(data)
    labels=estimator.labels_
    n_noise_=list(labels).count(-1)
    n_clusters_=len(set(labels)) - (1 if -1 in labels else 0)
    print('%-9s\t%.2fs\t%s\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), 'unknown',
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))

def bench_GaussianMixture(estimator, name, data):
    t0=time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.n_iter_,
            metrics.normalized_mutual_info_score(labels, estimator.predict(data)),
            metrics.homogeneity_score(labels, estimator.predict(data)),
            metrics.completeness_score(labels, estimator.predict(data))))



bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

bench_AffinityPropagation(AffinityPropagation(convergence_iter=20),
                          name="AP", data=data)

bench_MeanShift(MeanShift(), name="MeanShift", data=data)

# bench_SpectralClustering(SpectralClustering(),name="MeanShift", data=data)

bench_SpectralClustering(SpectralClustering(
    n_clusters=n_digits), name="Spectral", data=data)

bench_AgglomerativeClustering(AgglomerativeClustering(n_clusters=n_digits, linkage='ward', connectivity=None),
                         name="Ward-hier", data=data)

bench_AgglomerativeClustering(AgglomerativeClustering(n_clusters=n_digits, linkage='complete', connectivity=None),
                         name="Agglomerative", data=data)

bench_DBSCAN(DBSCAN(eps=5, min_samples=3), name="DBSCAN", data=data)

bench_GaussianMixture(mixture.GaussianMixture(n_components=n_digits, covariance_type='full'),
                      name="GaussMix", data=data)
# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca=PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')


# In[ ]:





# In[ ]:





# In[ ]:




