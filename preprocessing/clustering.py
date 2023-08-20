import math

import numpy as np
from sklearn.datasets import make_blobs

def _check_types(v1, v2):
    if isinstance(v1, (list, tuple)):
        if isinstance(v2, (list, tuple)):
                assert len(v1) == len(v2), "both vectors should be of the same length"
                return math.sqrt(sum(math.pow((v1[idx] - v2[idx]), 2) for idx in range(len(v1))))
        else:
            raise TypeError(f"[TYPE-MISMATCH]: arg v1 type -> {type(v1)} arg v2 type -> {type(v2)}")
    return

def distance(v1, v2):
    ans = _check_types(v1, v2)
    if ans is not None:
        return ans
    
    ans = _check_types(v2, v1)
    if ans is not None:
        return ans
    
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(v1 - v2)
    raise TypeError("types of args not supported yet!")
    
def fill_groups(grps, centroids, X):
    distances = []
    for idx in range(X.shape[0]):
        curr_sample = list(X[idx, :])
        for cent_idx in range(centroids.shape[0]):
            curr_centroid = list(centroids[cent_idx, :])
            distances.append(distance(curr_centroid, curr_sample))
        
        grps[distances.index(min(distances))].append(curr_sample)
        distances = []
    
    return grps

def run(k, n_samples=500, n_features=10, cluster_std=0.5, shuffle=True, random_state=1):
    # don't need y's for unsupervised learning in k-means clustering (a prototype-based grouping method)
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=k, cluster_std=cluster_std, shuffle=shuffle, random_state=random_state)
    initial_centroids = X[:k]
    X_rest = X[k:]
    
    groups = {key: [initial_centroids[key]] for key in range(k)}
    groups = fill_groups(groups, initial_centroids, X_rest)
    
    for i, val in enumerate(groups.values()):
        print(f"Number of elements in group {i}: {len(val)}")
    
    print("*" * 50)
    return

if __name__ == '__main__':
    Ks = [2, 3, 4, 5, 6, 7, 8]
    for k in Ks:
        run(k, random_state=123)
