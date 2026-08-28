import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DEFAULT_DATASET_PATH = "normalized_dataset.json"
#columns that are metadata/labels rather than clustering features, dropped if present
NON_FEATURE_COLUMNS = ["timestamp", "is_fault"]


#load an arbitrary number of numeric feature columns from a JSON-records dataset
def load_features(path, non_feature_columns=NON_FEATURE_COLUMNS):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.drop(columns=[c for c in non_feature_columns if c in df.columns])
    df = df.select_dtypes(include=[np.number])
    return df.to_numpy(), df.columns.tolist()


#define the gap statistic calculation function
def gap_statistic(X, k_max=10, n_refs=5):
    """Compute the Gap statistic for an nxm dataset in X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data to cluster.
    k_max : int
        The maximum number of clusters to test for.
    n_refs : int
        The number of reference datasets to generate.

    Returns
    -------
    gaps : array, shape (k_max,)
        The gap statistic values for each k.
    """
    #reference data should be uniform over X's principal axes, not its raw min/max box:
    #for correlated/high-dimensional features an axis-aligned box is a much larger, differently
    #shaped volume than the real data, which biases the gap statistic toward larger k (Tibshirani et al., 2001)
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_rotated = X_centered @ Vt.T
    rot_mins = X_rotated.min(axis=0)
    rot_maxs = X_rotated.max(axis=0)

    def generate_reference_data():
        rand_rotated = np.random.uniform(rot_mins, rot_maxs, size=X.shape)
        return rand_rotated @ Vt + X_mean

    gap_values = []

    #loop over a range of k values
    for k in range(1, k_max + 1):
        #fit kmeans to the original data
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        original_inertia = kmeans.inertia_

        #compute the average inertia for the reference datasets
        reference_inertias = []
        for _ in range(n_refs):
            random_data = generate_reference_data()
            kmeans.fit(random_data)
            reference_inertias.append(kmeans.inertia_)

        #calculate the gap statistic
        gap = np.log(np.mean(reference_inertias)) - np.log(original_inertia)
        gap_values.append(gap)

    return np.array(gap_values)


def run(dataset_path=DEFAULT_DATASET_PATH, k_max=10, n_refs=5, show=True, save_path=None):
    np.random.seed(42)

    #load the dataset (works for any number of numeric feature columns)
    X, feature_names = load_features(dataset_path)
    print(f"Loaded {X.shape[0]} instances x {X.shape[1]} features: {feature_names}")

    #calculate gap statistic for different values of k
    gap_values = gap_statistic(X, k_max=k_max, n_refs=n_refs)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k_max + 1), gap_values, marker='o')
    plt.title('Gap Statistic vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    #determine the optimal k
    optimal_k = np.argmax(np.diff(gap_values)) + 1
    print(f"Optimal number of clusters according to Gap Statistic: {optimal_k}")

    return optimal_k, gap_values


if __name__ == "__main__":
    run()
