# NOTE:
# Can we extract some kind of rigid and peculiar pattern on 
# repeated script runs ??
# 
# One behaviour I found is stated below :
# Suppose, the true underlying clusters in the dataset are "x"
# Hyperparameter tuning would show us that:
# (1) It's more erratic number of iterations required to stabilize,
#     when far from true underlying k value.
# (2) As we approach nearer to "x", the number of iterations
#     fixate at an arbitrary constant value, just before and after the
#     true k value. The arbitrary constant is generally on the smaller
#     side of the number line.
# (3) Number of iterations changes drastically at true k value
#     and fixate at some new arbitrary constant.
# 
# This (and other) behaviour(s) can be visualized by setting the
# 'vis' boolean flag to True, as an argument to the run() function.

import math

import matplotlib.pyplot as plt
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

def _initialize_grps(X, k):
    initial_centroids = X[:k]
    X_rest = X[k:]
    
    groups = {key: [initial_centroids[key]] for key in range(k)}
    groups = fill_groups(initial_centroids, X_rest, grps=groups)
    return groups

def find_new_centroids(grps: dict) -> np.ndarray:
    print_group_details(grps)
    cents = []
    for samples in grps.values():
        cents.append([sum(x)/len(x) for x in zip(*samples)])
    return np.array(cents)

def fill_groups(centroids, X, grps=None) -> dict[int, list]:
    distances = []
    for idx in range(X.shape[0]):
        curr_sample = list(X[idx, :])
        for cent_idx in range(centroids.shape[0]):
            curr_centroid = list(centroids[cent_idx, :])
            distances.append(distance(curr_centroid, curr_sample))
        
        if grps is None:
            grps = {key: [] for key in range(len(centroids))}

        grps[distances.index(min(distances))].append(curr_sample)
        distances = []
    
    return grps

def group_changes(initial_grps: dict[int, list], final_grps: dict[int, list]) -> int:
    changes = 0
    for key in initial_grps.keys():
        samples_initial = initial_grps[key]
        samples_final = final_grps[key]
        for sample_initial in samples_initial:
            not_in_new = True
            for sample_final in samples_final:
                norm_final = np.linalg.norm(sample_final)
                norm_initial = np.linalg.norm(sample_initial)
                same_length = (norm_final == norm_initial)
                if same_length:
                    float_value = round(float(np.dot(sample_initial, sample_final) / (norm_initial * norm_final)), 1)
                    angle = math.degrees(math.acos(float_value))
                    if angle == 0:
                        not_in_new = False
                        break
            if not_in_new:
                changes += 1

    #assert math.modf(changes / 2)[0] == 0, "logic of algorithm to find number of changes is **WRONG**"
    #return int(changes / 2)
    return changes

def print_group_details(grps):
    for i, val in enumerate(grps.values()):
        print(f"Number of elements in group {i}: {len(val)}")
    print("*" * 50)
    return

def k_means_algorithm(X, k, min_threshold=4, last_times=4):
    groups = _initialize_grps(X, k)

    print("\nGroups Initialized!!\n")
    
    num_changes: list[int] = [float('-inf')]*(last_times-1) + [float('inf')]
    num_iters = 0
    are_same = False
    print("\nTraining starts...\n")
    while (not are_same) and num_changes[-1] > min_threshold:
        print(f"Iteration #{num_iters}")
        cents = find_new_centroids(groups)
        new_groups = fill_groups(cents, X)
        for grp in new_groups.values():
            if len(grp) == 0:
                print(f"\nDiscard {k} from the set -> not suitable\n")
                return 0
        num_changes.append(group_changes(groups, new_groups))
        groups = new_groups
        are_same = (len(set(num_changes[-last_times:])) == 1)
        num_iters += 1
    
    print("\nTraining stops...\n")
    print(f"It took {num_iters} iterations to form clusters.\n")
    return num_iters

def _initialize_dataset(k_orig, n_samples=500, n_features=10, cluster_std=0.5, shuffle=True, random_state=1):
    # don't need y's for unsupervised learning in k-means clustering (a prototype-based grouping method)
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=k_orig, cluster_std=cluster_std, shuffle=shuffle, random_state=random_state)
    return X

def vis_result(x, y, k_orig):
    plt.scatter(x, y)
    plt.xlabel("K (NUMBER OF CLUSTERS)")
    plt.ylabel("ITERATIONS TO STABLE CLUSTERS")

    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    plt.plot(x, p(y), "r--")

    plt.title(f"TRUE K VALUE : {k_orig}")

    plt.show()
    return

def run(vis=False):
    Ks = list(range(2, 30, 1))
    k_original = np.random.choice(Ks)
    print(f"Original K : {k_original}\n\n")
    X = _initialize_dataset(k_original, random_state=123)
    if vis:
        iters = []
        k_vals = []
        for k in Ks:
            print("--" * 10, f"Checking for k-value : {k}", "--" * 10)
            iters.append(k_means_algorithm(X, k))
            k_vals.append(k)
        vis_result(k_vals, iters, k_original)
        return
    
    for k in Ks:
        print("--" * 10, f"Checking for k-value : {k}", "--" * 10)
        _ = k_means_algorithm(X, k)
    return

if __name__ == '__main__':
    run(vis=True)