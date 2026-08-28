from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import kmean_default

DEFAULT_DATASET_PATH = "normalized_dataset.json"
DEFAULT_BEST_K_PATH = "best_k.json"


def find_elbow_nd(k_values, *value_arrays):
    """Knee/elbow detection generalized to N dimensions: builds one point per k
    from (k, value_arrays[0][k], value_arrays[1][k], ...), normalizes each axis
    to [0, 1], and returns the k whose point is furthest from the straight line
    joining the first and last points."""
    k_arr = np.array(list(k_values), dtype=float)
    axes = [k_arr] + [np.array(v, dtype=float) for v in value_arrays]

    #normalize every axis to [0, 1] so distance isn't skewed by scale
    norm_axes = [(a - a.min()) / (a.max() - a.min()) for a in axes]
    points = np.stack(norm_axes, axis=1)

    p1 = points[0]
    p2 = points[-1]
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for p in points:
        v = p - p1
        proj = np.dot(v, line_vec_norm) * line_vec_norm
        distance = np.linalg.norm(v - proj)
        distances.append(distance)

    elbow_k = list(k_values)[np.argmax(distances)]
    return elbow_k


def run(dataset_path=DEFAULT_DATASET_PATH, k_range=range(1, 11), best_k_path=DEFAULT_BEST_K_PATH, show=True, save_prefix=None):
    """save_prefix, if given, saves three plots: <prefix>_distortion.png, <prefix>_inertia.png, <prefix>_3d.png."""
    #load the data
    df, X = kmean_default.load_dataset(dataset_path)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    excluded_names = {"timestamp"}
    feature_names = [name for name in data[0].keys() if name not in excluded_names]
    X = np.array([[row[name] for name in feature_names] for row in data], dtype=float)

    #clusterting model and calculating distortion and inertia
    K = list(k_range)
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    for k in K:
        labels, centers, _ = kmean_default.fit_kmeans(X, k)

        distortions.append(sum(np.min(cdist(X, kmean_default.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
        inertias.append(kmean_default.inertia_)

        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]

    #tabulating the results
    print("Distortion values:")
    for key, val in mapping1.items():
        print(f"{key} : {val:.4f}")

    print("\nInertia values:")
    for key, val in mapping2.items():
        print(f"{key} : {val:.4f}")

    best_k_distortion = find_elbow_nd(K, distortions)
    best_k_inertia = find_elbow_nd(K, inertias)
    best_k_combined = find_elbow_nd(K, distortions, inertias)

    print(f"\nElbow (best k) detected from distortion: {best_k_distortion}")
    print(f"Elbow (best k) detected from inertia: {best_k_inertia}")
    print(f"Elbow (best k) detected from combined 3D (k, distortion, inertia): {best_k_combined}")

    with open(best_k_path, "w") as f:
        json.dump({
            "best_k_distortion": best_k_distortion,
            "best_k_inertia": best_k_inertia,
            "best_k_combined_3d": best_k_combined,
        }, f, indent=2)
    print(f"Saved best k values to {best_k_path}")

    #build the 3 output paths (distortion/inertia/3d) from a shared prefix, e.g.
    #save_prefix="plots/elbow" -> "plots/elbow_distortion.png", "..._inertia.png", "..._3d.png"
    save_paths = {"distortion": None, "inertia": None, "3d": None}
    if save_prefix:
        root, ext = os.path.splitext(save_prefix)
        ext = ext or ".png"
        save_paths = {name: f"{root}_{name}{ext}" for name in save_paths}

    #visualizing the results
    plt.plot(K, distortions, 'bx-')
    plt.axvline(best_k_distortion, color='red', linestyle='--', label=f'elbow k={best_k_distortion}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.legend()
    if save_paths["distortion"]:
        plt.savefig(save_paths["distortion"])
        print(f"Saved plot to {save_paths['distortion']}")
    if show:
        plt.show()
    else:
        plt.close()

    plt.plot(K, inertias, 'bx-')
    plt.axvline(best_k_inertia, color='red', linestyle='--', label=f'elbow k={best_k_inertia}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.legend()
    if save_paths["inertia"]:
        plt.savefig(save_paths["inertia"])
        print(f"Saved plot to {save_paths['inertia']}")
    if show:
        plt.show()
    else:
        plt.close()

    #3D visualization: k, distortion, and inertia together
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(K, distortions, inertias, 'bx-')
    combined_idx = K.index(best_k_combined)
    ax.scatter(best_k_combined, distortions[combined_idx], inertias[combined_idx],
               color='red', s=80, label=f'elbow k={best_k_combined}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Distortion')
    ax.set_zlabel('Inertia')
    ax.set_title('The Elbow Method in 3D (k, Distortion, Inertia)')
    ax.legend()
    if save_paths["3d"]:
        fig.savefig(save_paths["3d"])
        print(f"Saved plot to {save_paths['3d']}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "best_k_distortion": best_k_distortion,
        "best_k_inertia": best_k_inertia,
        "best_k_combined_3d": best_k_combined,
    }


if __name__ == "__main__":
    run()
