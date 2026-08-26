#sites used: https://www.geeksforgeeks.org/k-nearest-neighbors/

# 0- Import libraries
from collections import Counter
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#plot classification report
def plot_classification_report(y_true, y_pred, save_path=None):
    fig = plt.figure()
    sns.heatmap(pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).iloc[:-1, :].T,
                annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    if save_path is None:
        save_path = "classification_report.png"
    plt.savefig(save_path)
    plt.close(fig)

# generic evaluation helper: prints accuracy/report, saves a classification-report
# heatmap and a confusion matrix, tagged by which classifier and which split
# (train/validation/test) produced the predictions. Used for all three splits
# so the diagnostics stay identical across them and are easy to compare.
def evaluate_predictions(method_label, split_label, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"[{method_label}] {split_label} Accuracy: {acc}")
    print(f"[{method_label}] {split_label} Classification Report:\n{classification_report(y_true, y_pred)}")
    plot_classification_report(y_true, y_pred, save_path=f"classification_report_{method_label}_{split_label}.png")

    print(f"[{method_label}] {split_label} Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=np.unique(y_true))
    disp.plot(cmap=plt.cm.Blues)
    disp.figure_.savefig(f"Confusion_Matrix_{method_label}_{split_label}.png")
    plt.close(disp.figure_)
    return acc


k = 3

#1 -Load the dataset
with open("C:\\Users\\maria\\OneDrive\\Desktop\\bolsa1\\KNN\\KNN_test_train_val_test\\random_random_appropriate_dataset_all_runs.json") as f:
    dataset = pd.DataFrame(json.load(f))

FEATURE_COLS = ["temperature", "o2", "pressure"]
FEATURE_LABELS = {"temperature": "Temperature", "o2": "Oxygen", "pressure": "Pressure"}

#raw, whole-dataset overview (all splits combined) - purely descriptive, not tied
#to any model or split, so it lives separately from the train/validation/test
#diagnostics below.
#marker shape encodes the per-sample "faulty" status; colour (assigned further
#down, once the KNN neighbourhoods are clustered) encodes which neighbourhood
#each point belongs to, so the two pieces of information don't compete for the
#same visual channel.
FAULT_LABELS = {0: "Not faulty", 1: "Faulty"}
FAULT_MARKERS = {0: "o", 1: "^"}
dataset_fault_markers = dataset["faulty"].map(FAULT_MARKERS)
#first slots of the validated categorical palette (blue, orange, ...) - these
#pass CVD/contrast checks pairwise, so neighbourhoods stay distinguishable
CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#builds legend proxy artists for the two independent encodings (colour for
#neighbourhood, marker shape for fault status) since neither is a single
#scatter series that matplotlib could legend automatically
def _cluster_fault_legend_handles(cluster_color_map, fault_marker_map, fault_labels):
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=color, markersize=10, label=f"Neighbourhood {cluster}")
        for cluster, color in cluster_color_map.items()
    ]
    handles += [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="none", markeredgecolor="black",
               color="black", markersize=9, label=fault_labels[fault_value])
        for fault_value, marker in fault_marker_map.items()
    ]
    return handles

#2 - Split the dataset into training, validation and testing sets

#normalize the features

sensor = dataset[FEATURE_COLS]
fault = dataset["faulty"]
# use the dataset's own train/validation/test split (grouped by run_id) instead
# of a random row-level split, since rows from the same run are highly
# correlated and a random split would leak near-duplicate readings across splits
train_mask = dataset["split"] == "train"
validation_mask = dataset["split"] == "val"
test_mask = dataset["split"] == "test"
X_train, X_validation, X_test = sensor[train_mask], sensor[validation_mask], sensor[test_mask]

y_train, y_validation, y_test = fault[train_mask], fault[validation_mask], fault[test_mask]

# fit the scaler on the training split only, then apply it to all three splits:
# temperature (~200-960), o2 (~3-7) and pressure (~-64--36) are on very
# different scales, so unscaled Euclidean distance is dominated almost
# entirely by temperature and the classifier collapses to predicting the
# majority class
scaler = StandardScaler()
scaler.fit(X_train)
train_features = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_COLS, index=X_train.index)
validation_features = pd.DataFrame(scaler.transform(X_validation), columns=FEATURE_COLS, index=X_validation.index)
test_features = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

#3 - Choose the k value
#evaluate knn 
def evaluate_knn(k, train_features, y_train, validation_features, y_validation):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, y_train)
    y_pred = knn.predict(validation_features)
    acc = accuracy_score(y_validation, y_pred)
    print(f"[k selection] k={k}, validation accuracy={acc}")
    return accuracy_score(y_validation, y_pred)

#train, score on validation, and plot the resulting elbow curve
#3. 1 elbow method
k_candidateselbow = list(range(1, 22, 2))
validation_accuracies = []
for k_candidate in k_candidateselbow:
    knn_k = KNeighborsClassifier(n_neighbors=k_candidate)
    knn_k.fit(train_features, y_train)
    validation_accuracies.append(accuracy_score(y_validation, knn_k.predict(validation_features)))

best_k_elbow = k_candidateselbow[int(np.argmax(validation_accuracies))]
best_acc_elbow = max(validation_accuracies)
print(f"[k selection] validation accuracy by k: {dict(zip(k_candidateselbow, validation_accuracies))}")
print(f"[k selection] best k on validation: {best_k_elbow}")

fig = plt.figure()
plt.plot(k_candidateselbow, validation_accuracies, marker="o")
plt.axvline(best_k_elbow, color="red", linestyle="--", label=f"best k = {best_k_elbow}")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Validation Accuracy")
plt.title("Elbow Method: Validation Accuracy vs k")
plt.xticks(k_candidateselbow)
plt.legend()
plt.grid(True)
plt.savefig("elbow_validation_accuracy_vs_k.png")
plt.close(fig)

# 3.2 cross-validation method
k_candidatescv = list(range(1, 22, 2))
cv_accuracies = []
for k_candidate in k_candidatescv:
    knn_k = KNeighborsClassifier(n_neighbors=k_candidate)
    cv_scores = cross_val_score(knn_k, train_features, y_train, cv=5, scoring="accuracy")
    cv_accuracies.append(np.mean(cv_scores))
best_k_cv = k_candidatescv[int(np.argmax(cv_accuracies))]
best_acc_cv = max(cv_accuracies)
print(f"[k selection] cross-validation accuracy by k: {dict(zip(k_candidatescv, cv_accuracies))}")
print(f"[k selection] best k on cross-validation: {best_k_cv}")

fig = plt.figure()
plt.plot(k_candidatescv, cv_accuracies, marker="o")
plt.axvline(best_k_cv, color="red", linestyle="--", label=f"best k = {best_k_cv}")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Cross-Validation Method: Accuracy vs k")
plt.xticks(k_candidatescv)
plt.legend()
plt.grid(True)
plt.savefig("cross_validation_accuracy_vs_k.png")
plt.close(fig)

#3.3 the gridsearch method
param_grid = {"n_neighbors": k_candidatescv}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(train_features, y_train)
best_k_grid = grid_search.best_params_["n_neighbors"]
best_acc_grid = grid_search.best_score_
print(f"[k selection] best k on grid search: {best_k_grid}")

#3.4 the randomized search CV method
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_grid, n_iter=10, cv=5, scoring="accuracy", random_state=42)
random_search.fit(train_features, y_train)
best_k_random = random_search.best_params_["n_neighbors"]
best_acc_random = random_search.best_score_
print(f"[k selection] best k on randomized search: {best_k_random}")

#3.5 compare the four methods and keep whichever k scored the highest
#accuracy (note: elbow/cross-validation report validation/cv accuracy while
#grid/randomized search report mean cv accuracy from sklearn - not perfectly
#apples-to-apples, but all four are accuracy scores on 0-1 so picking the max
#is a reasonable way to pick a single k to move forward with)
k_selection_results = {
    "elbow": (best_k_elbow, best_acc_elbow),
    "cross-validation": (best_k_cv, best_acc_cv),
    "grid search": (best_k_grid, best_acc_grid),
    "randomized search": (best_k_random, best_acc_random),
}
print(f"[k selection] best k/accuracy by method: {k_selection_results}")

best_method = max(k_selection_results, key=lambda method: k_selection_results[method][1])
k = k_selection_results[best_method][0]
print(f"[k selection] chosen method: {best_method}, chosen k: {k} (accuracy={k_selection_results[best_method][1]})")

#see the impact of k in the test and train datasets
#(loop variable renamed to neighbor_k - it used to be named k and silently
#clobbered the k chosen above via k_selection_results, so every classifier
#downstream was actually fit with k=19, the last value tried here, instead of
#the selected k)
neighbors = list(range(1, 20))
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, neighbor_k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbor_k)
    knn.fit(train_features, y_train)
    train_accuracy[i] = knn.score(train_features, y_train)
    test_accuracy[i] = knn.score(test_features, y_test)

plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

#3.6 - cluster the (scaled) sensor readings into neighbourhoods for the
#scatter plots below: reuses the same k chosen above so the number of
#neighbourhoods lines up with the KNN classifiers' own k, fit on the training
#split only (consistent with the scaler) and then applied to every split/the
#whole dataset
neighborhood_model = KMeans(n_clusters=k, random_state=42, n_init=10)
neighborhood_model.fit(train_features)
CLUSTER_COLORS = {cluster: CLUSTER_PALETTE[cluster % len(CLUSTER_PALETTE)] for cluster in range(k)}

dataset_scaled = pd.DataFrame(scaler.transform(dataset[FEATURE_COLS]), columns=FEATURE_COLS, index=dataset.index)
dataset_clusters = pd.Series(neighborhood_model.predict(dataset_scaled), index=dataset.index)
dataset_cluster_colors = dataset_clusters.map(CLUSTER_COLORS)

#4 - Calculate the distance

def euclidean_distance(x1,x2):
    return np.sqrt(sum((float(x1[col]) - float(x2[col])) ** 2 for col in FEATURE_COLS))

#5 - KNN prediction
#one method
def knn_predict1(train_data, train_labels, test_data, k):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(test_data, train_data.iloc[i])
        distances.append((dist, train_labels.iloc[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

# batch version of knn_predict1: same distance/majority-vote logic, but
# vectorized (and chunked) so it can run over thousands of points (a grid,
# a full split) in reasonable time instead of one row at a time
def knn_predict1_batch(train_data, train_labels, test_data, k, chunk_size=500):
    train_arr = train_data[FEATURE_COLS].to_numpy(dtype=float)
    if isinstance(test_data, pd.DataFrame):
        test_arr = test_data[FEATURE_COLS].to_numpy(dtype=float)
    else:
        test_arr = np.asarray(test_data, dtype=float)
    train_labels_arr = train_labels.to_numpy()

    preds = np.empty(len(test_arr), dtype=train_labels_arr.dtype)
    for start in range(0, len(test_arr), chunk_size):
        end = start + chunk_size
        batch = test_arr[start:end]
        dists = cdist(batch, train_arr)  # (batch_size, n_train)
        nearest_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        for i, idxs in enumerate(nearest_idx):
            labels, counts = np.unique(train_labels_arr[idxs], return_counts=True)
            preds[start + i] = labels[np.argmax(counts)]
    return preds

# adapter so knn_predict1's from-scratch logic can be plugged into
# plot_3d_scatter_with_centers wherever it expects an sklearn-style classifier
class ManualKNNClassifier:
    def __init__(self, train_data, train_labels, k):
        self.train_data = train_data
        self.train_labels = train_labels
        self.k = k

    def predict(self, data):
        return knn_predict1_batch(self.train_data, self.train_labels, data, self.k)

#6 - Fit both KNN implementations once, on the training split, using the k
#chosen above via the validation elbow curve
manual_knn = ManualKNNClassifier(train_features, y_train, k)

#second method {https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbor-algorithm-in-python/}
def fit_sklearn_knn(train_data, train_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    return knn

knn_model = fit_sklearn_knn(train_features, y_train, k)

#7 - Evaluate both classifiers on all three splits: train diagnostics show
#how well each model fits its own training data, validation diagnostics
#mirror that for the held-out validation split, and test diagnostics are the
#final, untouched-until-now evaluation
splits = {
    "train": (train_features, y_train),
    "validation": (validation_features, y_validation),
    "test": (test_features, y_test),
}

accuracies = {"manual": {}, "sklearn": {}}
for split_name, (X_split, y_split) in splits.items():
    manual_preds = manual_knn.predict(X_split)
    accuracies["manual"][split_name] = evaluate_predictions("manual", split_name, y_split, manual_preds)

    sklearn_preds = knn_model.predict(X_split)
    accuracies["sklearn"][split_name] = evaluate_predictions("sklearn", split_name, y_split, sklearn_preds)

#8 - Compare accuracy across splits and implementations, to visualize any
#train/validation/test generalization gap
split_order = ["train", "validation", "test"]
fig, ax = plt.subplots()
x = np.arange(len(split_order))
width = 0.35
manual_bars = ax.bar(x - width / 2, [accuracies["manual"][s] for s in split_order], width, label="manual knn_predict1")
sklearn_bars = ax.bar(x + width / 2, [accuracies["sklearn"][s] for s in split_order], width, label="sklearn KNeighborsClassifier")
ax.bar_label(manual_bars, fmt="%.2f")
ax.bar_label(sklearn_bars, fmt="%.2f")
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in split_order])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy Comparison: Train vs Validation vs Test")
ax.legend()
fig.tight_layout()
fig.savefig("accuracy_comparison_train_validation_test.png")
plt.close(fig)

def _draw_centroids(ax, centers, center_colors=None, annotate=False):
    if centers is None:
        return
    if center_colors is None:
        center_colors = ["black"] * len(centers)
    for idx, (cx, cy, cz) in enumerate(centers):
        ax.scatter(cx, cy, cz, c=[center_colors[idx]], marker="X", s=200,
                   edgecolors="black", linewidths=1.5, label=f"Cluster {idx} centroid")
        if annotate:
            ax.text(cx, cy, cz, f"({cx:.2f}, {cy:.2f}, {cz:.2f})", fontsize=9, color="black")

# combines any centroid legend entries (added directly on the axes) with the
# group/fault proxy handles passed in, so one call produces the full legend
def _apply_legend(ax, extra_handles=None):
    handles, _ = ax.get_legend_handles_labels()
    if extra_handles:
        handles = list(handles) + list(extra_handles)
    if handles:
        ax.legend(handles=handles)

def _finalize_3d_plot(fig, ax, cols, title, save_path, show):
    ax.set_xlabel(str(FEATURE_LABELS.get(cols[0], cols[0])))
    ax.set_ylabel(str(FEATURE_LABELS.get(cols[1], cols[1])))
    ax.set_zlabel(str(FEATURE_LABELS.get(cols[2], cols[2])))
    ax.set_title(title)
    ax.grid(True)
    if save_path is None:
        save_path = f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    fig.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_3d_scatter(X, Y, Z, colors, markers=None, centers=None, center_colors=None, cols=FEATURE_COLS,
                     title="3D Scatter Plot of Sensor Readings", save_path=None, show=False, legend_handles=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    colors = np.asarray(colors)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if markers is None:
        ax.scatter(xs=X, ys=Y, zs=Z, c=colors, s=12)  # type: ignore[arg-type]
    else:
        # matplotlib scatter takes one marker per call, not one per point, so
        # points are split into per-marker groups (one scatter call each)
        markers = np.asarray(markers)
        for marker in np.unique(markers):
            mask = markers == marker
            ax.scatter(xs=X[mask], ys=Y[mask], zs=Z[mask], c=colors[mask], s=20, marker=marker,  # type: ignore[arg-type]
                       edgecolors="black", linewidths=0.3)
    _draw_centroids(ax, centers, center_colors, annotate=True)
    _apply_legend(ax, legend_handles)
    _finalize_3d_plot(fig, ax, cols, title, save_path, show)

#9 - Plot the whole-dataset sensor readings: colour = neighbourhood (KMeans
#cluster), marker shape = fault status
#(a raw data overview, independent of any split, but coloured using the
#neighbourhood model fit on the training split above)
plot_3d_scatter(
    dataset[FEATURE_COLS[0]],
    dataset[FEATURE_COLS[1]],
    dataset[FEATURE_COLS[2]],
    dataset_cluster_colors,
    markers=dataset_fault_markers,
    cols=FEATURE_COLS,
    title="3D Scatter Plot of Sensor Readings (colour = neighbourhood, marker = fault status)",
    save_path="3d_scatter_plot.png",
    legend_handles=_cluster_fault_legend_handles(CLUSTER_COLORS, FAULT_MARKERS, FAULT_LABELS),
)

def plot_3d_scatter_with_centers(X, Y, Z, colors, classifier, markers=None, centers=None, center_colors=None, cols=FEATURE_COLS,
                             title="3D Scatter Plot of Sensor Readings with Cluster Centers", save_path=None,
                             show=False, n_grid=15, scaler=None, legend_handles=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    colors = np.asarray(colors)

    x_min, x_max = X.min() - 1, X.max() + 1
    y_min, y_max = Y.min() - 1, Y.max() + 1
    z_min, z_max = Z.min() - 1, Z.max() + 1

    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                                np.linspace(y_min, y_max, n_grid),
                                np.linspace(z_min, z_max, n_grid))

    #predict the labels for each point in the meshgrid
    grid_points = pd.DataFrame({
        cols[0]: xx.ravel(),
        cols[1]: yy.ravel(),
        cols[2]: zz.ravel(),
    })
    # the grid is built in raw sensor units, but the classifiers are fit on
    # scaled features, so it must go through the same scaler before predict
    if scaler is not None:
        grid_points = pd.DataFrame(scaler.transform(grid_points), columns=cols, index=grid_points.index)
    preds = classifier.predict(grid_points)
    boundary_colors = np.where(preds == 1, "red", "green")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    #faint voxel cloud showing the predicted class of the grid
    ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=boundary_colors, s=8, alpha=0.03, linewidths=0)
    #actual sensor readings on top: colour = neighbourhood, marker shape = fault status
    if markers is None:
        ax.scatter(xs=X, ys=Y, zs=Z, c=colors, s=25, edgecolors="black", linewidths=0.5)  # type: ignore[arg-type]
    else:
        markers = np.asarray(markers)
        for marker in np.unique(markers):
            mask = markers == marker
            ax.scatter(xs=X[mask], ys=Y[mask], zs=Z[mask], c=colors[mask], s=30, marker=marker,  # type: ignore[arg-type]
                       edgecolors="black", linewidths=0.5)

    _draw_centroids(ax, centers, center_colors, annotate=False)
    _apply_legend(ax, legend_handles)
    _finalize_3d_plot(fig, ax, cols, title, save_path, show)

# ===========================================================================
# 10 - TEST SET 3D DECISION BOUNDARY (kept separate from the train/validation
# diagnostics above): only the test rows are scattered here, against the
# decision boundary each classifier learned from the training split.
# ===========================================================================
test_cluster_colors = dataset_cluster_colors[test_mask]
test_fault_markers = dataset_fault_markers[test_mask]
cluster_fault_legend = _cluster_fault_legend_handles(CLUSTER_COLORS, FAULT_MARKERS, FAULT_LABELS)

plot_3d_scatter_with_centers(
    X_test[FEATURE_COLS[0]],
    X_test[FEATURE_COLS[1]],
    X_test[FEATURE_COLS[2]],
    test_cluster_colors,
    knn_model,
    markers=test_fault_markers,
    cols=FEATURE_COLS,
    title="3D Scatter Plot of Test Set with Decision Boundary (sklearn KNeighborsClassifier, colour = neighbourhood, marker = fault status)",
    save_path="3d_scatter_plot_sklearn.png",
    scaler=scaler,
    legend_handles=cluster_fault_legend,
)

plot_3d_scatter_with_centers(
    X_test[FEATURE_COLS[0]],
    X_test[FEATURE_COLS[1]],
    X_test[FEATURE_COLS[2]],
    test_cluster_colors,
    manual_knn,
    markers=test_fault_markers,
    cols=FEATURE_COLS,
    title="3D Scatter Plot of Test Set with Decision Boundary (manual knn_predict1, colour = neighbourhood, marker = fault status)",
    save_path="3d_scatter_plot_manual.png",
    scaler=scaler,
    legend_handles=cluster_fault_legend,
)
