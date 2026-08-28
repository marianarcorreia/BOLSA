#train k-means

import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

FEATURE_NAMES = ["sensor_1", "sensor_2", "sensor_3"]

#populated by fit_kmeans(), mirroring sklearn's KMeans fitted attributes
cluster_centers_ = None
inertia_ = None


def load_dataset(dataset_path):
    """Load the sensor dataset and return (df, sensor_number) where sensor_number
    is an (n_samples, 3) array of sensor_1/sensor_2/sensor_3 values."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    sensor_number = np.array([[row[name] for name in FEATURE_NAMES] for row in data])
    return df, sensor_number


def init_centers(data, k, seed=42):
    np.random.seed(seed)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return np.array([mins + np.random.random(data.shape[1]) * (maxs - mins) for _ in range(k)])


def assign_clusters(data, centers):
    dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_centers(data, labels, centers, k):
    new_centers = centers.copy()
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            new_centers[i] = points.mean(axis=0)
    return new_centers


def fit_kmeans(data, k, seed=42, max_iter=300):
    """From-scratch k-means (Lloyd's algorithm). Returns (labels, centers, history),
    where history is a list of (labels, centers) snapshots, one per iteration
    starting with the random init."""
    centers = init_centers(data, k, seed)
    history = [(assign_clusters(data, centers), centers.copy())]
    prev_labels = history[0][0]

    for _ in range(max_iter):
        labels = assign_clusters(data, centers)
        centers = update_centers(data, labels, centers, k)
        history.append((labels, centers.copy()))
        if np.array_equal(labels, prev_labels):
            break
        prev_labels = labels

    final_labels = assign_clusters(data, centers)

    global cluster_centers_, inertia_
    cluster_centers_ = centers
    inertia_ = sum(
        np.sum((data[final_labels == i] - centers[i]) ** 2)
        for i in range(k)
    )

    return final_labels, centers, history


def plot_3d_scatter(X, Y, Z, colors, centers=None, center_colors=None, title="3D Scatter Plot of Sensor Readings", save_path=None, show=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, c=colors, s=12)
    if centers is not None:
        if center_colors is None:
            center_colors = ["black"] * len(centers)
        for idx, (cx, cy, cz) in enumerate(centers):
            ax.scatter(cx, cy, cz, c=[center_colors[idx]], marker="X", s=200,
                       edgecolors="black", linewidths=1.5, label=f"Cluster {idx} centroid")
            ax.text(cx, cy, cz, f"({cx:.2f}, {cy:.2f}, {cz:.2f})", fontsize=9, color="black")
        ax.legend()
    ax.set_xlabel("Sensor 1")
    ax.set_ylabel("Sensor 2")
    ax.set_zlabel("Sensor 3")
    ax.set_title(title)
    ax.grid(True)
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run(dataset_path="normalized_dataset.json", frames_dir="kmeans_frames", video_path="kmeans_training.gif", k=3, show=True):
    """Train k-means on the dataset, saving one frame per iteration to frames_dir and
    stitching them into an animated gif at video_path. Returns (final_labels, final_centers)."""
    from PIL import Image

    #load the data
    df, sensor_number = load_dataset(dataset_path)

    #dataset overview: each sensor over time, green = no fault, red = fault
    timestamps = pd.to_datetime(df["timestamp"])
    sensor_names = [name for name in df.columns if name.startswith("sensor")]
    colors = np.where(df["is_fault"].astype(bool), "#C62828", "#2E7D32")

    fig_overview, axes = plt.subplots(len(sensor_names), 1, sharex=True, figsize=(12, 3 * len(sensor_names)))
    for ax, sensor in zip(axes, sensor_names):
        ax.scatter(timestamps, df[sensor], c=colors, s=12)
        ax.set_ylabel(sensor)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("timestamp")
    fig_overview.autofmt_xdate()

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E7D32", label="No fault", markersize=8),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#C62828", label="Fault", markersize=8),
    ]
    fig_overview.legend(handles=legend_handles, loc="upper right")
    fig_overview.suptitle("Sensor readings over time")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig_overview)

    X = sensor_number[:, 0]
    Y = sensor_number[:, 1]
    Z = sensor_number[:, 2]

    #create a graph with the data set with 3 dimension, where each dimension is a sensor, and the color is the fault or not
    plot_3d_scatter(X, Y, Z, colors, show=show)

    #give each cluster a fixed color that its points will always be drawn with
    cluster_cmap = plt.get_cmap("tab10")
    cluster_colors = [cluster_cmap(i % 10) for i in range(k)]

    def labels_to_colors(labels):
        return [cluster_colors[label] for label in labels]

    #random initialized centers with data points
    initial_centers = init_centers(sensor_number, k)
    plot_3d_scatter(X, Y, Z, colors, initial_centers, center_colors=cluster_colors, show=show)

    #run k-means, keeping a per-iteration history so the whole run can be turned into a video afterwards
    final_labels, final_centers, history = fit_kmeans(sensor_number, k)

    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    initial_labels, init_centers_snapshot = history[0]
    initial_frame_path = os.path.join(frames_dir, "frame_000_init.png")
    plot_3d_scatter(X, Y, Z, labels_to_colors(initial_labels), init_centers_snapshot, center_colors=cluster_colors,
                     title="Iteration 0 (random init)", save_path=initial_frame_path, show=False)
    frame_paths.append(initial_frame_path)

    for iteration, (labels, centers) in enumerate(history[1:], start=1):
        frame_path = os.path.join(frames_dir, f"frame_{iteration:03d}.png")
        plot_3d_scatter(X, Y, Z, labels_to_colors(labels), centers, center_colors=cluster_colors,
                         title=f"Iteration {iteration}", save_path=frame_path, show=False)
        frame_paths.append(frame_path)

    plot_3d_scatter(X, Y, Z, labels_to_colors(final_labels), final_centers, center_colors=cluster_colors,
                     title="Final clusters", show=show)

    #stitch the saved frames into an animated gif ("video") of the whole training run
    frame_images = [Image.open(p) for p in frame_paths]
    frame_images[0].save(
        video_path,
        save_all=True,
        append_images=frame_images[1:],
        duration=500,
        loop=0,
    )
    print(f"Saved {len(frame_paths)} frames to '{frames_dir}/' and training video to '{video_path}'")

    return final_labels, final_centers


if __name__ == "__main__":
    #let the user choose the names of the input/output files via command-line arguments
    parser = argparse.ArgumentParser(description="Train k-means on sensor data and export frames/video.")
    parser.add_argument("--dataset", default="normalized_dataset.json", help="Path to the input dataset JSON file")
    parser.add_argument("--frames-dir", default="kmeans_frames", help="Directory where per-iteration frame images are saved")
    parser.add_argument("--video", default="kmeans_training.gif", help="Path to the output training animation (gif)")
    args = parser.parse_args()

    run(args.dataset, args.frames_dir, args.video)
