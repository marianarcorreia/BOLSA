#sites used: https://www.geeksforgeeks.org/k-nearest-neighbors/

# 0- Import libraries
from collections import Counter
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.svm import SVC

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

#confusion matrix (with all the three)

# 1- Choose the k value
#1.1 Fixed k value
k = 3
#1.2 Cross validation to find the best k value

#1.3 Elbow method to find the best k value

#1.4 Odd value for k

#2 -Load the dataset
with open("C:\\Users\\maria\\OneDrive\\Desktop\\bolsa1\\KNN\\KNN_test_bias\\bias_PRESSURE_appropriate_dataset_all_runs.json") as f:
    dataset = pd.DataFrame(json.load(f))

#3 - Split the dataset into training and testing sets

FEATURE_COLS = ["temperature", "o2", "pressure"]
FEATURE_LABELS = {"temperature": "Temperature", "o2": "Oxygen", "pressure": "Pressure"}

#normalize the features

sensor = dataset[FEATURE_COLS]
fault = dataset["faulty"]
# use the dataset's own train/test split (grouped by run_id) instead of a
# random row-level split, since rows from the same run are highly correlated
# and a random split would leak near-duplicate readings across train/test
train_mask = dataset["split"] == "train"
test_mask = dataset["split"] == "test"
X_train, X_test = sensor[train_mask], sensor[test_mask]
y_train, y_test = fault[train_mask], fault[test_mask]
# change the "group" column to a numerical value
dataset["group"] = dataset["group"].astype("category").cat.codes

# fit the scaler on the training split only, then apply it to both splits:
# temperature (~200-960), o2 (~3-7) and pressure (~-64--36) are on very
# different scales, so unscaled Euclidean distance is dominated almost
# entirely by temperature and the classifier collapses to predicting the
# majority class
scaler = StandardScaler()
scaler.fit(X_train)
train_features = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_COLS, index=X_train.index)
test_features = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

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

prediction = knn_predict1(train_features, y_train, test_features.iloc[0], k)

# batch version of knn_predict1: same distance/majority-vote logic, but
# vectorized (and chunked) so it can run over thousands of points (a grid,
# a full test set) in reasonable time instead of one row at a time
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
        #clf = SVC(random_state=42)
        #clf.fit(self.train_data, self.train_labels)
        return knn_predict1_batch(self.train_data, self.train_labels, data, self.k)

manual_knn = ManualKNNClassifier(train_features, y_train, k)

# apply knn_predict1 (via the batched implementation) to the whole test
# set, mirroring what knn_predict2 reports below, so the two
# implementations can be compared like-for-like
manual_test_preds = manual_knn.predict(test_features)
print(f"[knn_predict1] Accuracy: {accuracy_score(y_test, manual_test_preds)}")

print(f"[knn_predict1] Classification Report:\n{classification_report(y_test, manual_test_preds)}")
plot_classification_report(y_test, manual_test_preds, save_path="classification_report_manual.png")
print(f"[knn_predict1] Confusion Matrix:\n{confusion_matrix(y_test, manual_test_preds)}")
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, manual_test_preds), display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
disp.figure_.savefig("Confusion_Matrix_knn_predict1.png")
plt.close(disp.figure_)

#second method {https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbor-algorithm-in-python/}
def knn_predict2(train_data, train_labels, test_data, test_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    #clf = SVC(random_state=42)
    #clf.fit(self.train_data, self.train_labels)
    knn.fit(train_data, train_labels)

    y_pred = knn.predict(test_data)
    print(f"Accuracy: {accuracy_score(test_labels, y_pred)}")
    print(f"Classification Report:\n{classification_report(test_labels, y_pred)}")
    plot_classification_report(test_labels, y_pred, save_path="classification_report_sklearn.png")

    print(f"[knn_predict2] Confusion Matrix:\n{confusion_matrix(test_labels, y_pred)}")
    disp = ConfusionMatrixDisplay(confusion_matrix(test_labels, y_pred), display_labels=np.unique(test_labels))
    disp.plot(cmap=plt.cm.Blues)
    disp.figure_.savefig("Confusion_Matrix_knn_predict2_sklearn.png")
    plt.close(disp.figure_)
    return knn

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
    ax.legend()

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

def plot_3d_scatter(X, Y, Z, colors, centers=None, center_colors=None, cols=FEATURE_COLS,
                     title="3D Scatter Plot of Sensor Readings", save_path=None, show=False):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs=X, ys=Y, zs=Z, c=colors, s=12)  # type: ignore[arg-type]
    _draw_centroids(ax, centers, center_colors, annotate=True)
    _finalize_3d_plot(fig, ax, cols, title, save_path, show)

#6 - Plot the sensor readings, colored by fault status
fault_colors = dataset["faulty"].map({0: "green", 1: "red"})
plot_3d_scatter(
    dataset[FEATURE_COLS[0]],
    dataset[FEATURE_COLS[1]],
    dataset[FEATURE_COLS[2]],
    fault_colors,
    cols=FEATURE_COLS,
    title="3D Scatter Plot of Sensor Readings (green = ok, red = faulty)",
    save_path="3d_scatter_plot.png",
)

knn_model = knn_predict2(train_features, y_train, test_features, y_test, k)

def plot_3d_scatter_with_centers(X, Y, Z, colors, classifier, centers=None, center_colors=None, cols=FEATURE_COLS,
                             title="3D Scatter Plot of Sensor Readings with Cluster Centers", save_path=None,
                             show=False, n_grid=15, scaler=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

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
    #actual sensor readings on top
    ax.scatter(xs=X, ys=Y, zs=Z, c=colors, s=25, edgecolors="black", linewidths=0.5) # type: ignore

    _draw_centroids(ax, centers, center_colors, annotate=False)
    _finalize_3d_plot(fig, ax, cols, title, save_path, show)

plot_3d_scatter_with_centers(
    dataset[FEATURE_COLS[0]],
    dataset[FEATURE_COLS[1]],
    dataset[FEATURE_COLS[2]],
    fault_colors,
    knn_model,
    cols=FEATURE_COLS,
    title="3D Scatter Plot with Decision Boundary (sklearn KNeighborsClassifier)",
    save_path="3d_scatter_plot_sklearn.png",
    scaler=scaler,
)

plot_3d_scatter_with_centers(
    dataset[FEATURE_COLS[0]],
    dataset[FEATURE_COLS[1]],
    dataset[FEATURE_COLS[2]],
    fault_colors,
    manual_knn,
    cols=FEATURE_COLS,
    title="3D Scatter Plot with Decision Boundary (manual knn_predict1)",
    save_path="3d_scatter_plot_manual.png",
    scaler=scaler,
)
