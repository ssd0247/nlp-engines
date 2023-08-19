import math

import numpy as np
from sklearn.datasets import make_blobs

K_GRPS = 5

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

if __name__ == '__main__':
    X, y = make_blobs(
        n_samples=500,
        n_features=10,
        centers=K_GRPS,
        cluster_std=0.5,
        shuffle=True,
        random_state=123)

    initial_centroids = X[:K_GRPS]
    X_rest = X[K_GRPS:]
    groups = {key: [] for key in range(K_GRPS)}
    groups = fill_groups(groups, initial_centroids, X_rest)
    for val in groups.values():
        print(len(val))
