#sites used: https://www.geeksforgeeks.org/k-nearest-neighbors/

# 0- Import libraries
import argparse
from collections import Counter
import io
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, homogeneity_score, rand_score, v_measure_score, mutual_info_score
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
        save_path = out_path("classification_report.png")
    plt.savefig(save_path)
    plt.close(fig)

# generic evaluation helper: prints accuracy/report, saves a classification-report
# heatmap and a confusion matrix, tagged by which classifier and which split
# (train/validation/test) produced the predictions. Used for all three splits
# so the diagnostics stay identical across them and are easy to compare.
def evaluate_predictions(method_label, split_label, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    homogeneity = homogeneity_score(y_true, y_pred)
    mutual_info = mutual_info_score(y_true, y_pred)
    rand = rand_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    print(f"[{method_label}] {split_label} Accuracy: {acc}")
    print(f"[{method_label}] {split_label} Precision: {precision}")
    print(f"[{method_label}] {split_label} Recall: {recall}")
    print(f"[{method_label}] {split_label} F1 Score: {f1}")
    print(f"[{method_label}] {split_label} Homogeneity: {homogeneity}")
    print(f"[{method_label}] {split_label} Mutual Information: {mutual_info}")
    print(f"[{method_label}] {split_label} Rand Index: {rand}")
    print(f"[{method_label}] {split_label} V-Measure: {v_measure}")
    print(f"[{method_label}] {split_label} Classification Report:\n{classification_report(y_true, y_pred)}")
    plot_classification_report(y_true, y_pred, save_path=out_path(f"classification_report_{method_label}_{split_label}.png"))

    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    print(f"[{method_label}] {split_label} Confusion Matrix:\n{cm}")

    # per-class right/wrong counts: "actual" reads across each row (of the
    # true-labeled samples, how many landed on the diagonal vs elsewhere),
    # "predicted" reads down each column (of the samples given that predicted
    # label, how many were actually right vs wrong)
    correct = np.diag(cm)
    actual_totals = cm.sum(axis=1)
    actual_wrong = actual_totals - correct
    predicted_totals = cm.sum(axis=0)
    predicted_wrong = predicted_totals - correct

    summary_lines = ["Actual class (row) right/wrong:"]
    summary_lines += [f"  {lbl}: {c} right, {w} wrong" for lbl, c, w in zip(labels, correct, actual_wrong)]
    summary_lines.append("Predicted class (col) right/wrong:")
    summary_lines += [f"  {lbl}: {c} right, {w} wrong" for lbl, c, w in zip(labels, correct, predicted_wrong)]
    print(f"[{method_label}] {split_label} " + " | ".join(summary_lines))

    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.figure_.text(0.98, 0.5, "\n".join(summary_lines), fontsize=8,
                       va="center", ha="left", family="monospace")
    fig_w, fig_h = disp.figure_.get_size_inches()
    disp.figure_.set_size_inches(fig_w + 3.2, fig_h)
    disp.figure_.savefig(out_path(f"Confusion_Matrix_{method_label}_{split_label}.png"), bbox_inches="tight")
    plt.close(disp.figure_)
    return acc, report_dict


def parse_args():
    parser = argparse.ArgumentParser(description="KNN fault detection pipeline")
    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument("--k", type=int, default=None,
                          help="Use this fixed k instead of auto-selecting one")
    k_group.add_argument("--optimal", action="store_true",
                          help="Auto-select k via elbow/cross-validation/grid/randomized search (default if --k is not given)")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Path to the dataset JSON to run (defaults to the noise_O2 dataset)")
    return parser.parse_args()

args = parse_args()
k = args.k if args.k is not None else 3

#1 -Load the dataset
DATASET_PATH = Path(args.dataset) if args.dataset is not None else Path("C:\\Users\\maria\\OneDrive\\Desktop\\bolsa1\\KNN\\KNN4_test\\datasets\\bias_TEMP_appropriate_dataset_all_runs_with_split.json")
with open(DATASET_PATH) as f:
    dataset = pd.DataFrame(json.load(f))

# every generated file (charts, reports, confusion matrices) is written under
# "[faulty_type]_results/[sensor]" (e.g. drift_results/temperature), derived from the
# dataset filename ("{faulty_type}_{SENSOR}_appropriate_dataset_all_runs.json")
# so the destination always matches the dataset actually being run, and those
# folders already exist for every faulty_type/sensor combination
SENSOR_DIR_NAMES = {"O2": "o2", "PRESSURE": "pressure", "TEMP": "temperature"}
_faulty_type, _sensor_code = DATASET_PATH.stem.split("_")[:2]
FAULTY_TYPE = _faulty_type.lower()
SENSOR = SENSOR_DIR_NAMES.get(_sensor_code.upper(), _sensor_code.lower())
RESULTS_DIR = Path(f"{FAULTY_TYPE}_results") / SENSOR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def out_path(filename):
    return str(RESULTS_DIR / filename)

# capture everything printed from here on so the full console output of this
# run (data-split summary, k selection, accuracy/metrics tables, neighbourhood
# breakdowns, ...) can be saved as one PNG next to the plots/CSVs it explains,
# instead of only living in terminal scrollback
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

_console_log = io.StringIO()
sys.stdout = _Tee(sys.stdout, _console_log)

def save_console_log_png(save_path, font_size=14):
    lines = _console_log.getvalue().splitlines() or [""]
    font = None
    for font_path in ("C:\\Windows\\Fonts\\consola.ttf", "C:\\Windows\\Fonts\\cour.ttf"):
        if Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_height = int(font_size * 1.35)
    max_width = int(max((measurer.textbbox((0, 0), line, font=font)[2] for line in lines), default=0))

    padding = 20
    img = Image.new("RGB", (max_width + padding * 2, line_height * len(lines) + padding * 2), "white")
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill="black")
        y += line_height
    img.save(save_path)

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
FAULT_COLORS = {0: "#1baf7a", 1: "#e34948"}  # no_fault = green, fault = red
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

# each run_id is one "store" (a full time-series of sensor readings from a
# single deployment): every sample in a run must land in the same split, or
# validation/test would be scored on near-duplicate data the model already
# trained on. This also confirms the train split actually contains runs from
# both the "fault" and "no_fault" groups, so it gets to learn what a normal
# case and a faulty case each look like, while validation/test - held out at
# the run level - are purely there to check whether that generalizes.
runs_per_split = dataset.groupby("run_id")["split"].nunique()
leaking_runs = runs_per_split[runs_per_split > 1].index.tolist()
assert not leaking_runs, f"run_id(s) (stores) spanning multiple splits: {leaking_runs}"

runs_by_split = dataset.groupby("split")["run_id"].nunique()
runs_by_split_and_group = dataset.groupby(["split", "group"])["run_id"].nunique().unstack(fill_value=0)
print(f"[data split] runs (stores) per split: {runs_by_split.to_dict()}")
print(f"[data split] runs (stores) per split by group:\n{runs_by_split_and_group}")
assert (runs_by_split_and_group > 0).all(axis=None), \
    f"a split is missing runs from one of the fault/no_fault groups:\n{runs_by_split_and_group}"

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

if args.k is not None:
    print(f"[k selection] using user-provided k: {k}")
else:
    #train, score on validation, and plot the resulting elbow curve
    #3. 1 elbow method
    k_candidateselbow = list(range(2, 10, 2))
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
    plt.savefig(out_path("elbow_validation_accuracy_vs_k.png"))
    plt.close(fig)

    # 3.2 cross-validation method
    k_candidatescv = list(range(2, 10, 2))
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
    plt.savefig(out_path("cross_validation_accuracy_vs_k.png"))
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

#see the impact of k in the validation and train datasets
#(loop variable renamed to neighbor_k - it used to be named k and silently
#clobbered the k chosen above via k_selection_results, so every classifier
#downstream was actually fit with k=19, the last value tried here, instead of
#the selected k)
#(uses validation, not test, so the test set stays untouched until section 7's
#final evaluation - it was previously scored against test_features/y_test here,
#which meant test accuracy was already being inspected across 9 different k
#values before the model was even finalized)
neighbors = list(range(2, 10, 2))
train_accuracy = np.empty(len(neighbors))
validation_accuracy = np.empty(len(neighbors))

for i, neighbor_k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbor_k)
    knn.fit(train_features, y_train)
    train_accuracy[i] = knn.score(train_features, y_train)
    validation_accuracy[i] = knn.score(validation_features, y_validation)

fig = plt.figure()
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, validation_accuracy, label="Validation Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.savefig(out_path("accuracy_train_validation_vs_neighbors.png"))
plt.close(fig)

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

accuracies = {"manual": {}}
reports = {"manual": {}}
predictions = {"manual": {}}
for split_name, (X_split, y_split) in splits.items():
    manual_preds = manual_knn.predict(X_split)
    predictions["manual"][split_name] = manual_preds
    accuracies["manual"][split_name], reports["manual"][split_name] = evaluate_predictions("manual", split_name, y_split, manual_preds)

    #sklearn_preds = knn_model.predict(X_split)
    #accuracies["sklearn"][split_name], reports["sklearn"][split_name] = evaluate_predictions("sklearn", split_name, y_split, sklearn_preds)

#8 - Compare accuracy across splits and implementations, to visualize any
#train/validation/test generalization gap
split_order = ["train", "validation", "test"]
fig, ax = plt.subplots()
x = np.arange(len(split_order))
width = 0.35
manual_bars = ax.bar(x - width / 2, [accuracies["manual"][s] for s in split_order], width, label="manual knn_predict1")
#sklearn_bars = ax.bar(x + width / 2, [accuracies["sklearn"][s] for s in split_order], width, label="sklearn KNeighborsClassifier")
ax.bar_label(manual_bars, fmt="%.2f")
#ax.bar_label(sklearn_bars, fmt="%.2f")
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in split_order])
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy Comparison: Train vs Validation vs Test")
ax.legend()
fig.tight_layout()
fig.savefig(out_path("accuracy_comparison_train_validation_test.png"))
plt.close(fig)

#8a - Consolidated precision/recall/f1 table across splits and methods: the
#accuracy bar chart above hides per-class behaviour (e.g. one split doing
#worse specifically on the "Faulty" class), so this pulls the per-class
#numbers out of the classification reports collected in section 7 into one
#table that puts train/validation/test side by side.
FAULT_CLASS_LABELS = {"0": "Not faulty", "1": "Faulty"}
metrics_rows = []
for method_label, split_reports in reports.items():
    for split_name, report_dict in split_reports.items():
        for class_key, class_label in FAULT_CLASS_LABELS.items():
            class_metrics = report_dict.get(class_key)
            if class_metrics is None:
                continue
            metrics_rows.append({
                "method": method_label, "split": split_name, "class": class_label,
                "precision": class_metrics["precision"], "recall": class_metrics["recall"],
                "f1-score": class_metrics["f1-score"], "support": class_metrics["support"],
            })
        macro = report_dict.get("macro avg")
        if macro is not None:
            metrics_rows.append({
                "method": method_label, "split": split_name, "class": "macro avg",
                "precision": macro["precision"], "recall": macro["recall"],
                "f1-score": macro["f1-score"], "support": macro["support"],
            })

metrics_table = pd.DataFrame(metrics_rows).set_index(["method", "split", "class"]).sort_index()
print(f"[metrics comparison] per-class precision/recall/f1 by method/split:\n{metrics_table}")
metrics_table.to_csv(out_path("metrics_comparison_train_validation_test.csv"))

fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(metrics_table))))
sns.heatmap(metrics_table[["precision", "recall", "f1-score"]], annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Precision / Recall / F1 by Method, Split and Class")
fig.tight_layout()
fig.savefig(out_path("metrics_comparison_train_validation_test.png"))
plt.close(fig)

#8b - Compare how the KMeans neighbourhoods (section 3.6) are populated
#across splits: if train/validation/test draw very different proportions of
#points from each neighbourhood, that distribution shift is itself a likely
#explanation for any accuracy/metric gap seen above.
neighbourhood_split_counts = pd.DataFrame({
    split_name: dataset_clusters[mask].value_counts(normalize=True)
    for split_name, mask in zip(split_order, [train_mask, validation_mask, test_mask])
}).reindex(range(k)).fillna(0.0)
print(f"[neighbourhood comparison] proportion of each split's points per neighbourhood:\n{neighbourhood_split_counts}")
neighbourhood_split_counts.to_csv(out_path("neighbourhood_distribution_train_validation_test.csv"))

fig, ax = plt.subplots()
x = np.arange(k)
width = 0.8 / len(split_order)
for offset, split_name in enumerate(split_order):
    ax.bar(x + (offset - (len(split_order) - 1) / 2) * width, neighbourhood_split_counts[split_name],
           width, label=split_name.capitalize())
ax.set_xticks(x)
ax.set_xticklabels([str(c) for c in x], fontsize=8)
ax.set_xlabel("Neighbourhood (KMeans cluster id)")
ax.set_ylabel("Proportion of split's points")
ax.set_title("Neighbourhood Distribution: Train vs Validation vs Test")
ax.legend()
fig.tight_layout()
fig.savefig(out_path("neighbourhood_distribution_train_validation_test.png"))
plt.close(fig)

#8c - Compare, for each split, how "pure" the k nearest training neighbours
#are for each of its points: fault_label_purity is the fraction of a point's
#k neighbours that share the majority fault label, and
#neighbourhood_cluster_purity is the fraction that fall in the majority
#KMeans neighbourhood. Low purity on validation/test relative to train means
#those points sit in messier neighbourhoods (near a class boundary or in an
#under-represented region), which is a direct, per-point explanation for
#misclassifications rather than just a summary accuracy number.
#Only the sklearn model's neighbour search is used - the manual knn_predict1
#implementation uses the same Euclidean distance and the same k, so it always
#selects the same k neighbours, just less efficiently.
train_cluster_labels = neighborhood_model.predict(train_features)

def neighbor_purity(query_features, exclude_self):
    query_k = k + 1 if exclude_self else k
    _, neighbor_idx = knn_model.kneighbors(query_features, n_neighbors=query_k)
    if exclude_self:
        neighbor_idx = neighbor_idx[:, 1:]
    neighbor_fault = y_train.to_numpy()[neighbor_idx]  # (n_query, k)
    neighbor_cluster = train_cluster_labels[neighbor_idx]  # (n_query, k)
    fault_mean = neighbor_fault.mean(axis=1)
    fault_purity = np.maximum(fault_mean, 1 - fault_mean)
    cluster_purity = np.array([
        np.unique(row, return_counts=True)[1].max() / len(row) for row in neighbor_cluster
    ])
    return fault_purity, cluster_purity

purity_frames = []
for split_name, X_split in zip(split_order, [train_features, validation_features, test_features]):
    fault_purity, cluster_purity = neighbor_purity(X_split, exclude_self=(split_name == "train"))
    purity_frames.append(pd.DataFrame({
        "split": split_name.capitalize(),
        "fault_label_purity": fault_purity,
        "neighbourhood_cluster_purity": cluster_purity,
    }))
purity_df = pd.concat(purity_frames, ignore_index=True)

purity_summary = purity_df.groupby("split")[["fault_label_purity", "neighbourhood_cluster_purity"]].mean()
print(f"[neighbourhood comparison] mean neighbour purity by split (sklearn KNN, k={k}):\n{purity_summary}")
purity_df.to_csv(out_path("neighbor_purity_train_validation_test.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
split_order_cap = [s.capitalize() for s in split_order]
sns.boxplot(data=purity_df, x="split", y="fault_label_purity", order=split_order_cap, ax=axes[0])
axes[0].set_title("Fault-label purity of each point's k neighbours")
axes[0].set_ylabel("Fraction of k neighbours sharing the majority fault label")
axes[0].set_ylim(0.4, 1.02)

sns.boxplot(data=purity_df, x="split", y="neighbourhood_cluster_purity", order=split_order_cap, ax=axes[1])
axes[1].set_title("Neighbourhood-cluster purity of each point's k neighbours")
axes[1].set_ylabel("Fraction of k neighbours in the majority KMeans cluster")
axes[1].set_ylim(0, 1.02)

fig.suptitle(f"KNN Neighbour Composition (sklearn, k={k}): Train vs Validation vs Test")
fig.tight_layout()
fig.savefig(out_path("neighbor_purity_train_validation_test.png"))
plt.close(fig)

#8d - Compare, for each neighbourhood (KMeans cluster), how many faulty vs
#not-faulty points it actually contains. A neighbourhood that is almost
#entirely one class is "easy" for KNN - any point landing there is
#surrounded by same-label neighbours. A neighbourhood close to a 50/50 split
#is where misclassifications are more likely, since a query point's k
#nearest neighbours will disagree on the label.
#Computed once for the whole dataset (all splits combined) and again
#restricted to just the validation split and just the test split, so the
#fault mix each held-out split actually draws from each neighbourhood can be
#compared against the whole-dataset picture (and against each other).
def neighbourhood_fault_composition(clusters, faulty, label, file_suffix):
    counts = pd.crosstab(clusters, faulty)
    counts = counts.rename(columns=FAULT_LABELS).reindex(range(k), fill_value=0)
    for fault_label in FAULT_LABELS.values():
        if fault_label not in counts:
            counts[fault_label] = 0
    counts = counts[list(FAULT_LABELS.values())]
    counts["total"] = counts.sum(axis=1)
    proportions = counts[list(FAULT_LABELS.values())].div(counts["total"].replace(0, np.nan), axis=0).fillna(0.0)

    print(f"[neighbourhood comparison] ({label}) fault vs not-faulty counts per neighbourhood:\n{counts}")
    print(f"[neighbourhood comparison] ({label}) fault vs not-faulty proportions per neighbourhood:\n{proportions}")
    counts.to_csv(out_path(f"neighbourhood_fault_counts{file_suffix}.csv"))
    proportions.to_csv(out_path(f"neighbourhood_fault_proportions{file_suffix}.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(k)
    for ax, data, ylabel, title, annotate_pct in (
        (axes[0], counts, "Number of points", "Fault vs Not-faulty counts per neighbourhood", False),
        (axes[1], proportions, "Proportion of neighbourhood's points", "Fault vs Not-faulty proportions per neighbourhood", True),
    ):
        bottom = np.zeros(k)
        for fault_value, fault_label in FAULT_LABELS.items():
            values = data[fault_label].to_numpy()
            ax.bar(x, values, bottom=bottom, color=FAULT_COLORS[fault_value], label=fault_label)
            if annotate_pct:
                for xi, (segment_bottom, segment_value) in enumerate(zip(bottom, values)):
                    if segment_value <= 0:
                        continue
                    ax.text(xi, segment_bottom + segment_value / 2, f"{segment_value:.0%}",
                            ha="center", va="center", fontsize=8, color="black")
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in x], fontsize=8)
        ax.set_xlabel("Neighbourhood (KMeans cluster id)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(f"Fault Composition of Each KNN Neighbourhood ({label})")
    fig.tight_layout()
    fig.savefig(out_path(f"neighbourhood_fault_composition{file_suffix}.png"))
    plt.close(fig)
    return counts, proportions

neighbourhood_fault_counts, neighbourhood_fault_proportions = neighbourhood_fault_composition(
    dataset_clusters, dataset["faulty"], "whole dataset", "")
neighbourhood_fault_composition(
    dataset_clusters[validation_mask], dataset.loc[validation_mask, "faulty"], "validation set", "_validation")
neighbourhood_fault_composition(
    dataset_clusters[test_mask], dataset.loc[test_mask, "faulty"], "test set", "_test")

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
        save_path = out_path(f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png")
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
    save_path=out_path("3d_scatter_plot.png"),
    legend_handles=_cluster_fault_legend_handles(CLUSTER_COLORS, FAULT_MARKERS, FAULT_LABELS),
)

# core boundary + scatter drawing, factored out of plot_3d_scatter_with_centers so
# it can be reused per-subplot by plot_3d_scatter_with_centers_multi below (one
# figure/legend/title shared across several splits' axes, instead of duplicating
# the grid/predict/scatter logic per split)
def _draw_3d_scatter_with_boundary(ax, X, Y, Z, colors, classifier, markers=None, centers=None, center_colors=None,
                                    cols=FEATURE_COLS, n_grid=15, scaler=None):
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

def plot_3d_scatter_with_centers(X, Y, Z, colors, classifier, markers=None, centers=None, center_colors=None, cols=FEATURE_COLS,
                             title="3D Scatter Plot of Sensor Readings with Cluster Centers", save_path=None,
                             show=False, n_grid=15, scaler=None, legend_handles=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_3d_scatter_with_boundary(ax, X, Y, Z, colors, classifier, markers=markers, centers=centers,
                                    center_colors=center_colors, cols=cols, n_grid=n_grid, scaler=scaler)
    _apply_legend(ax, legend_handles)
    _finalize_3d_plot(fig, ax, cols, title, save_path, show)

# same as plot_3d_scatter_with_centers, but draws several splits (e.g. test and
# validation) as side-by-side 3D subplots in one figure/one PNG instead of one
# file per split, so the same colour/marker encoding can be compared directly
# without flipping between images. `panels` is a list of dicts with keys
# label, X, Y, Z, colors, and optionally markers/centers/center_colors.
def plot_3d_scatter_with_centers_multi(panels, classifier, cols=FEATURE_COLS, suptitle="",
                                        save_path=None, show=False, n_grid=15, scaler=None,
                                        legend_handles=None, return_image=False):
    fig = plt.figure(figsize=(10 * len(panels), 8))
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        _draw_3d_scatter_with_boundary(ax, panel["X"], panel["Y"], panel["Z"], panel["colors"], classifier,
                                        markers=panel.get("markers"), centers=panel.get("centers"),
                                        center_colors=panel.get("center_colors"), cols=cols, n_grid=n_grid,
                                        scaler=scaler)
        ax.set_xlabel(str(FEATURE_LABELS.get(cols[0], cols[0])))
        ax.set_ylabel(str(FEATURE_LABELS.get(cols[1], cols[1])))
        ax.set_zlabel(str(FEATURE_LABELS.get(cols[2], cols[2])))
        ax.set_title(panel["label"])
        ax.grid(True)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 4))
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # return_image renders in-memory instead of writing its own file, so callers
    # (e.g. the by-neighbourhood/by-fault pair in section 10) can stitch several
    # of these figures into one combined PNG rather than leaving them as
    # separate files
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if save_path is None:
        save_path = out_path(f"{suptitle.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png")
    fig.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return None

# ===========================================================================
# 10 - TEST & VALIDATION SET 3D DECISION BOUNDARIES (kept separate from the
# train diagnostics above): the test and validation rows are each scattered
# here, against the decision boundary each classifier learned from the
# training split.
#
# For every split, two plot "types" are produced:
#   - by neighbourhood: colour = KMeans neighbourhood, marker = fault status
#     (same encoding as the whole-dataset overview above)
#   - by fault status: colour = fault/no_fault directly, as a single group of
#     two classes. "no_fault" is the group that teaches the model what
#     normal, error-free readings look like; "fault" is the group that
#     teaches it what a fault looks like. Both KNN models (manual and
#     sklearn) are always fit on the two groups together as one classifier -
#     fault/no_fault is only ever the label, never a separate model - this
#     view just makes that split visible on the scatter plot.
# ===========================================================================
def _fault_color_legend_handles(fault_color_map, fault_labels):
    return [
        Line2D([0], [0], marker="s", linestyle="", color=color, markersize=10, label=fault_labels[fault_value])
        for fault_value, color in fault_color_map.items()
    ]

dataset_fault_colors = dataset["faulty"].map(FAULT_COLORS)
cluster_fault_legend = _cluster_fault_legend_handles(CLUSTER_COLORS, FAULT_MARKERS, FAULT_LABELS)
fault_color_legend = _fault_color_legend_handles(FAULT_COLORS, FAULT_LABELS)

eval_splits = {
    "test": (X_test, test_mask),
    "validation": (X_validation, validation_mask),
}
classifiers = {
    "manual": manual_knn,
}

# one figure per classifier per encoding (neighbourhood / fault), with test and
# validation side by side as subplots - instead of one file per
# split-x-encoding combination, so the two splits can be compared directly
for clf_name, clf in classifiers.items():
    neighbourhood_panels = []
    fault_panels = []
    for split_name, (X_split, split_mask) in eval_splits.items():
        split_cluster_colors = dataset_cluster_colors[split_mask]
        split_fault_markers = dataset_fault_markers[split_mask]
        split_fault_colors = dataset_fault_colors[split_mask]

        neighbourhood_panels.append({
            "label": f"{split_name.capitalize()} Set",
            "X": X_split[FEATURE_COLS[0]], "Y": X_split[FEATURE_COLS[1]], "Z": X_split[FEATURE_COLS[2]],
            "colors": split_cluster_colors, "markers": split_fault_markers,
        })
        fault_panels.append({
            "label": f"{split_name.capitalize()} Set",
            "X": X_split[FEATURE_COLS[0]], "Y": X_split[FEATURE_COLS[1]], "Z": X_split[FEATURE_COLS[2]],
            "colors": split_fault_colors,
        })

    # render the by-neighbourhood and by-fault rows in-memory, then stack them
    # into a single PNG (one image per classifier covering both splits and
    # both colour encodings) instead of leaving them as two separate files
    neighbourhood_row = plot_3d_scatter_with_centers_multi(
        neighbourhood_panels, clf, cols=FEATURE_COLS,
        suptitle=f"3D Scatter Plot of Test vs Validation Sets with Decision Boundary ({clf_name} KNN, colour = neighbourhood, marker = fault status)",
        scaler=scaler,
        legend_handles=cluster_fault_legend,
        return_image=True,
    )

    fault_row = plot_3d_scatter_with_centers_multi(
        fault_panels, clf, cols=FEATURE_COLS,
        suptitle=f"3D Scatter Plot of Test vs Validation Sets with Decision Boundary ({clf_name} KNN, colour = fault status)",
        scaler=scaler,
        legend_handles=fault_color_legend,
        return_image=True,
    )

    if neighbourhood_row is not None and fault_row is not None:
        combined_width = max(neighbourhood_row.width, fault_row.width)
        combined = Image.new("RGB", (combined_width, neighbourhood_row.height + fault_row.height), "white")
        combined.paste(neighbourhood_row, (0, 0))
        combined.paste(fault_row, (0, neighbourhood_row.height))
        combined.save(out_path(f"3d_scatter_plot_test_validation_{clf_name}.png"))

#11 - Per-sample classification scatter: unlike the 3D scatters above (feature
#vs feature vs feature), this plots each sensor variable on its own axis
#against sample position within the split, so misclassifications can be read
#off sample-by-sample instead of only as an aggregate accuracy number.
#Per point: marker shape = true fault status (same as FAULT_MARKERS
#elsewhere), fill colour = KMeans neighbourhood (same as CLUSTER_COLORS), and
#edge colour = whether the classifier's prediction matched the true label
#(black = correct, red = misclassified) - so all three pieces of information
#that matter for diagnosing a wrong prediction (which variable, which
#neighbourhood, which fault group) are visible in one plot.
def _correctness_legend_handles():
    return [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black",
               markeredgewidth=1.5, markersize=9, label="Correctly classified"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="red",
               markeredgewidth=1.5, markersize=9, label="Misclassified"),
    ]

def plot_sample_classification_scatter(X_split, y_true, y_pred, cluster_colors, fault_markers,
                                        cols=FEATURE_COLS, suptitle="", save_path=None, legend_handles=None):
    x_idx = np.arange(len(X_split))
    correct = np.asarray(y_true) == np.asarray(y_pred)
    edge_colors = np.where(correct, "black", "red")
    edge_widths = np.where(correct, 0.6, 1.8)
    marker_arr = np.asarray(fault_markers)
    color_arr = np.asarray(cluster_colors)

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        y_vals = X_split[col].to_numpy(dtype=float)
        # misclassified points drawn last (on top) within each marker group so
        # a red-edged error is never hidden under a correctly-classified point
        for marker in np.unique(marker_arr):
            mask = marker_arr == marker
            for is_correct in (True, False):
                sub = mask & (correct == is_correct)
                if not sub.any():
                    continue
                ax.scatter(x_idx[sub], y_vals[sub], c=color_arr[sub], marker=marker, s=35,
                           edgecolors=edge_colors[sub], linewidths=edge_widths[sub], zorder=1 if is_correct else 2)
        ax.set_ylabel(str(FEATURE_LABELS.get(col, col)))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sample index (within split)")

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 4), fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if save_path is None:
        save_path = out_path(f"{suptitle.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png")
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

sample_scatter_legend = cluster_fault_legend + _correctness_legend_handles()
for clf_name in classifiers:
    for split_name, (X_split, split_mask) in eval_splits.items():
        plot_sample_classification_scatter(
            X_split, fault[split_mask], predictions[clf_name][split_name],
            dataset_cluster_colors[split_mask], dataset_fault_markers[split_mask],
            cols=FEATURE_COLS,
            suptitle=f"Per-Sample Classification Scatter ({split_name.capitalize()} Set, {clf_name} KNN)",
            legend_handles=sample_scatter_legend,
        )

#12 - Save everything this run printed (data split summary, k selection,
#accuracy/metrics tables, neighbourhood breakdowns, ...) as one PNG in
#RESULTS_DIR, so it stays alongside the plots/CSVs instead of only living in
#terminal scrollback
save_console_log_png(out_path("run_log.png"))
