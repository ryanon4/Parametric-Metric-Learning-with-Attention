import sklearn
import numpy as np
from sklearn.cluster import KMeans
from functions import calculate_accuracy, calculate_accuracy_per_class
import time
def evaluate_vectors(reduced_vectors, target):
    accuracy = []
    n_clusters = len(np.unique(np.array(target)))

    for vectors in reduced_vectors:
        try:
            dimensionality = len(vectors[0])
            print("Dimensionality: " + str(dimensionality))

            cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)

            clusters = cluster_model.labels_

            accuracy.append(calculate_accuracy(target, clusters))
        except Exception as e:
            continue
    return accuracy

def evaluate_vectors_class(reduced_vectors, target):
    accuracy = []
    per_class_accuracy = []
    n_clusters = len(np.unique(np.array(target)))

    for vectors in reduced_vectors:
        try:
            dimensionality = len(vectors[0])
            print("Dimensionality: " + str(dimensionality))

            cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)

            clusters = cluster_model.labels_
            del cluster_model
            _accuracy, _per_class = calculate_accuracy_per_class(target, clusters)
            accuracy.append(_accuracy)
            per_class_accuracy.append(_per_class)
        except Exception as e:
            continue
    return accuracy, per_class_accuracy
def evaluate_vectors_timed(reduced_vectors, target):
    accuracy = []
    n_clusters = len(np.unique(np.array(target)))
    runtimes = []
    for vectors in reduced_vectors:
        try:
            dimensionality = len(vectors[0])
            print("Dimensionality: " + str(dimensionality))

            start_time = time.perf_counter()
            cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)
            finish_time = time.perf_counter()
            elapsed_time = finish_time - start_time

            runtimes.append(elapsed_time)

            clusters = cluster_model.labels_

            accuracy.append(calculate_accuracy(target, clusters))
        except Exception as e:
            continue
    return accuracy, runtimes

def evaluate_vectors_baseline(vectors, target):
    n_clusters = len(np.unique(np.array(target)))


    dimensionality = len(vectors[0])
    print("Dimensionality: " + str(dimensionality))
    cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)
    clusters = cluster_model.labels_
    accuracy = calculate_accuracy(target, clusters)

    return accuracy