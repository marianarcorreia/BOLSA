from sklearn.metrics import davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt
import kmean_default

DEFAULT_DATASET_PATH = "normalized_dataset.json"


def run(dataset_path=DEFAULT_DATASET_PATH, k_range=range(2, 11), show=True, save_path=None):
    """Fit kmean_default over k_range and score each fit with the Davies-Bouldin index."""
    df, X = kmean_default.load_dataset(dataset_path)

    k_values = list(k_range)
    scores = []
    for k in k_values:
        labels, centers, _ = kmean_default.fit_kmeans(X, k)
        score = davies_bouldin_score(X, labels)
        scores.append(score)
        print(f"k={k} : Davies-Bouldin score = {score:.4f}")

    best_k = k_values[np.argmin(scores)]
    print(f"\nBest k by Davies-Bouldin score: {best_k}")

    plt.plot(k_values, scores, 'bx-')
    plt.axvline(best_k, color='red', linestyle='--', label=f'best k={best_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score using custom k-means')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return best_k, scores


if __name__ == "__main__":
    run()
