from cProfile import label
import json
import argparse
from pathlib import Path
from statistics import mean
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import winsorize
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn import preprocessing

with open("C:\\Users\\maria\\OneDrive\\Desktop\\bolsa1\\KMEANS\\PREPROCESSINGDATA\\BIAS\\bias_TEMP_appropriate_dataset_all_runs.json") as f:
    dataset = pd.DataFrame(json.load(f))


SENSOR_COLUMNS = ["temperature", "pressure", "o2"]
GROUP = ["no_fault", "faulty"]
SPLIT = ["train", "test"]

#0- Make it readable and understandable, rename columns, change data types, etc.

dataset["group_encoded"] = (dataset["group"] != "no_fault").astype(int)
dataset["split_encoded"] = (dataset["split"] != "train").astype(int)
dataset = dataset.drop(columns=["group", "split"])

output_path = Path(__file__).parent / "fault_detection_numerical_dataset.json"
dataset.to_json(output_path, orient="records", indent=2, date_format="iso")
# test if there are no mistakes in the encoding
assert (dataset["group_encoded"] == 0).equals(dataset["group_encoded"] == 0)
assert (dataset["group_encoded"] == 1).equals(dataset["group_encoded"] == 1)
assert (dataset["split_encoded"] == 0).equals(dataset["split_encoded"] == 0)
assert (dataset["split_encoded"] == 1).equals(dataset["split_encoded"] == 1)

#1-DATA CLEANING
#1.1-remove duplicates, used the {https://www.linkedin.com/pulse/data-cleaning-python-handling-duplicates-pandas-bennett-alexander-ay7xe/}
data_noduplicates = dataset.drop_duplicates()
data_noduplicates.head()
print(f'The number of duplicates removed: {len(dataset) - len(data_noduplicates)}   ')

#1.2- remove irrelevant rows and columns
data_12 = data_noduplicates.drop(columns=['split_encoded', 'group_encoded'])

#1.3- handle inconsistent values
def _flag_stuck_values(series: pd.Series, run_ids: pd.Series, min_repeats: int = 5) -> pd.Series:
    """Flag samples that are part of a run of >= min_repeats identical consecutive values within the same run_id (a 'stuck'/flatlined sensor)."""
    tmp = pd.DataFrame({'val': series.values, 'run': run_ids.values}, index=series.index)
    same_as_prev = tmp.groupby('run')['val'].transform(lambda s: s.eq(s.shift()).fillna(False))
    block_id = (~same_as_prev).groupby(tmp['run']).cumsum()
    block_size = tmp.groupby(['run', block_id])['val'].transform('size')
    return block_size >= min_repeats


def check_inconsistent_values(df: pd.DataFrame, temp_col='temperature', pressure_col='pressure',
                               o2_col='o2', faulty_col='faulty', group_col='group', split_col='split',
                               run_col='run_id', temp_range=(150, 950), o2_range=(0, 25),
                               stuck_min_repeats=5) -> dict[str, pd.Series]:
    issues: dict[str, pd.Series] = {}

    if temp_col in df.columns:
        issues[f'{temp_col}_out_of_range'] = ~df[temp_col].between(*temp_range)

    if o2_col in df.columns:
        issues[f'{o2_col}_out_of_range'] = ~df[o2_col].between(*o2_range)

    if pressure_col in df.columns:
        sign = pd.Series(np.sign(df[pressure_col].to_numpy()), index=df.index)
        common_sign = sign.mode().iloc[0] if not sign.mode().empty else 0
        issues[f'{pressure_col}_sign_mismatch'] = sign != common_sign

    if faulty_col in df.columns and group_col in df.columns:
        expected_faulty = (df[group_col] != 'no_fault').astype(int)
        issues['faulty_group_mismatch'] = df[faulty_col] != expected_faulty

    for col in [group_col, split_col]:
        if col in df.columns:
            as_str = df[col].astype(str)
            issues[f'{col}_whitespace'] = as_str != as_str.str.strip()
            print(f"Unique values in '{col}': {sorted(as_str.unique())}")

    if run_col in df.columns:
        for col in [temp_col, pressure_col, o2_col]:
            if col in df.columns:
                issues[f'{col}_stuck'] = _flag_stuck_values(df[col], df[run_col], stuck_min_repeats)

    any_issue = pd.Series(False, index=df.index)
    print('\nInconsistent value report:')
    for name, mask in issues.items():
        count = int(mask.sum())
        print(f'  {name}: {count} row(s)')
        any_issue |= mask
    issues['any_issue'] = any_issue
    print(f'Total rows with at least one inconsistency: {int(any_issue.sum())} / {len(df)}')

    return issues


inconsistency_report = check_inconsistent_values(data_noduplicates)

#1.4- handle outliers and noise
# colors: slot 1 (blue) = IQR, slot 2 (orange) = Z-score
COLOR_IQR = "#2a78d6"
COLOR_ZSCORE = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"

#detect outliers using z-score method
def detect_outliers_zscore(dataset, threshold=3):
    df = dataset[SENSOR_COLUMNS]
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = z_scores > threshold
    # Ensure a pandas DataFrame is returned with the original index/columns
    return pd.DataFrame(outliers, index=df.index, columns=df.columns)

#detect outliers using IQR method
def detect_outliers_iqr(dataset):
    df = dataset[SENSOR_COLUMNS]
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df < lower_bound) | (df > upper_bound)
    # Ensure a pandas DataFrame is returned with the original index/columns
    return pd.DataFrame(outliers, index=df.index, columns=df.columns)


def _plot_outliers_panel(ax, dataset, title):
    """Draw one IQR-vs-Z-score outlier panel for SENSOR_COLUMNS onto ax."""
    df = dataset[SENSOR_COLUMNS]
    zscore_outliers = detect_outliers_zscore(dataset)
    iqr_outliers = detect_outliers_iqr(dataset)

    positions = np.arange(0, len(SENSOR_COLUMNS))

    df_long = df.melt(value_vars=SENSOR_COLUMNS, var_name="sensor", value_name="value")

    sns.boxplot(
        data=df_long,
        x="sensor",
        y="value",
        order=SENSOR_COLUMNS,
        ax=ax,
        width=0.5,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor=COLOR_MUTED, linewidth=1.5),
        medianprops=dict(color=COLOR_INK, linewidth=2),
        whiskerprops=dict(color=COLOR_MUTED, linewidth=1.5),
        capprops=dict(color=COLOR_MUTED, linewidth=1.5),
    )

    rng = np.random.default_rng(0)
    for i, col in enumerate(SENSOR_COLUMNS):
        x_center = positions[i]
        iqr_mask = iqr_outliers[col]
        zscore_mask = zscore_outliers[col]

        iqr_only = df.loc[iqr_mask & ~zscore_mask, col]
        zscore_only = df.loc[zscore_mask & ~iqr_mask, col]
        both = df.loc[iqr_mask & zscore_mask, col]

        for values, color, marker, label in [
            (iqr_only, COLOR_IQR, "o", "IQR outlier"),
            (zscore_only, COLOR_ZSCORE, "^", "Z-score outlier"),
            (both, COLOR_INK, "D", "Both methods"),
        ]:
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                x_center + 0.35 + jitter,
                values,
                color=color,
                marker=marker,
                s=35,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
                label=label,
            )

        iqr_pct = iqr_mask.mean() * 100
        zscore_pct = zscore_mask.mean() * 100
        ax.text(
            x_center,
            df[col].max(),
            f"IQR: {iqr_pct:.1f}%\nZ-score: {zscore_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_INK,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([c.replace("_", " ").title() for c in SENSOR_COLUMNS])
    ax.set_xlabel("")
    ax.set_ylabel("Sensor reading")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper right")


def plot_outliers_boxplot(dataset, output_path="outliers_boxplot_bias_temp_test.png", facet_col=None):
    """Plot IQR vs. Z-score outliers for SENSOR_COLUMNS.

    If facet_col is given (e.g. "group_encoded" or "split_encoded"), draws one
    panel per numeric category in dataset[facet_col], each computed only from
    that subset's sensor readings.
    """
    if facet_col is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _plot_outliers_panel(ax, dataset, "Sensor readings: IQR vs. Z-score outlier detection")
    else:
        categories = sorted(dataset[facet_col].dropna().unique())
        fig, axes = plt.subplots(1, len(categories), figsize=(8 * len(categories), 5.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, category in zip(axes, categories):
            subset = dataset.loc[dataset[facet_col] == category]
            _plot_outliers_panel(ax, subset, f"{facet_col} = {category}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved boxplot to {output_path}")
    plt.close(fig)


def export_outliers_dataset(dataset, output_path="outliers_dataset_bias_temp_test.csv"):
    df = dataset[SENSOR_COLUMNS]
    zscore_outliers = detect_outliers_zscore(dataset)
    iqr_outliers = detect_outliers_iqr(dataset)

    result = dataset.copy()
    any_outlier = pd.Series(False, index=dataset.index)
    for col in SENSOR_COLUMNS:
        result[f"{col}_iqr_outlier"] = iqr_outliers[col]
        result[f"{col}_zscore_outlier"] = zscore_outliers[col]
        any_outlier |= iqr_outliers[col] | zscore_outliers[col]

    result = result.loc[any_outlier]
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} outlier rows to {output_path}")
    return result


#winsorize each sensor column independently, clipping the given tail proportions
def winsorize_sensors(dataset, limits=(0.05, 0.05)):
    df = dataset[SENSOR_COLUMNS]
    result = df.copy()
    for col in SENSOR_COLUMNS:
        clipped = winsorize(df[col].to_numpy(), limits=limits)
        # winsorize returns a MaskedArray; unwrap it back into a plain float column
        result[col] = np.asarray(clipped)
    return result


#compare winsorizing vs. deleting the same tail rows, per sensor column
def compare_winsorize_vs_deletion(dataset, limits=(0.05, 0.05)):
    df = dataset[SENSOR_COLUMNS]
    winsorized = winsorize_sensors(dataset, limits=limits)
    expected_changed = int(len(df) * sum(limits))

    for col in SENSOR_COLUMNS:
        lower_q = df[col].quantile(limits[0])
        upper_q = df[col].quantile(1 - limits[1])
        mask = (df[col] >= lower_q) & (df[col] <= upper_q)
        kept_after_deletion = df[col][mask]

        n_changed = int((df[col].to_numpy() != winsorized[col].to_numpy()).sum())
        original_mean = df[col].mean()
        winsorized_mean = winsorized[col].mean()
        deleted_mean = kept_after_deletion.mean()

        print(f"{col}:")
        print(f"  values changed by winsorizing: {n_changed} (predicted ~{expected_changed})")
        print(f"  mean original:    {original_mean:.4f}")
        print(f"  mean winsorized:  {winsorized_mean:.4f}  (shift {winsorized_mean - original_mean:+.4f})")
        print(f"  mean if deleted:  {deleted_mean:.4f}  (shift {deleted_mean - original_mean:+.4f}, n={len(kept_after_deletion)})")
        print()


#boxplot of each sensor column before vs. after winsorizing, side by side
def plot_winsorize_boxplot(dataset, limits=(0.05, 0.05), output_path="boxplot_bias_temp_test.png"):
    df = dataset[SENSOR_COLUMNS]
    winsorized = winsorize_sensors(dataset, limits=limits)

    original_long = df.melt(value_vars=SENSOR_COLUMNS, var_name="sensor", value_name="value")
    original_long["stage"] = "original"
    winsorized_long = winsorized.melt(value_vars=SENSOR_COLUMNS, var_name="sensor", value_name="value")
    winsorized_long["stage"] = "winsorized"
    combined = pd.concat([original_long, winsorized_long], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(
        data=combined,
        x="sensor",
        y="value",
        hue="stage",
        order=SENSOR_COLUMNS,
        ax=ax,
        width=0.6,
        palette={"original": COLOR_MUTED, "winsorized": COLOR_IQR},
    )

    ax.set_xticks(np.arange(len(SENSOR_COLUMNS)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in SENSOR_COLUMNS])
    ax.set_xlabel("")
    ax.set_ylabel("Sensor reading")
    ax.set_title(f"Before vs. after winsorizing (limits={limits})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved boxplot to {output_path}")
    plt.close(fig)



#export the full dataset with sensor columns replaced by their winsorized values
def export_winsorized_dataset(dataset, limits=(0.05, 0.05), output_path="dataset_bias_temp_test.csv", json_output_path="dataset_bias_temp_test.json"):
    df = dataset.copy()
    winsorized = winsorize_sensors(dataset, limits=limits)

    result = df.copy()
    for col in SENSOR_COLUMNS:
        result[col] = winsorized[col]

    result.to_csv(output_path, index=False)
    print(f"Saved winsorized dataset ({len(result)} rows) to {output_path}")

    result.to_json(json_output_path, orient="records", indent=2, date_format="iso")
    print(f"Saved winsorized dataset ({len(result)} rows) to {json_output_path}")
    return result

plot_outliers_boxplot(data_12)
plot_outliers_boxplot(data_noduplicates, output_path="outliers_boxplot_by_group.png", facet_col="group_encoded")
plot_outliers_boxplot(data_noduplicates, output_path="outliers_boxplot_by_split.png", facet_col="split_encoded")
export_outliers_dataset(data_12)
compare_winsorize_vs_deletion(data_12)
plot_winsorize_boxplot(data_12)
export_winsorized_dataset(data_12)
compare_winsorize_vs_deletion(data_12)
plot_winsorize_boxplot(data_12)

#2- Missing data
#2.1/locate and count gaps missing values {https://www.kaggle.com/code/alexisbcook/handling-missing-values}
def analyze_data_quality(df):
    missing_values_count = df.isnull().sum()
    nan_values_count = df.isna().sum()
    zero_values_count = (df == 0).sum()
    print(f'Missing values count:\n{missing_values_count}')
    print(f'NaN values count:\n{nan_values_count}')
    print(f'Zero values count:\n{zero_values_count}')

    print(f'The number of missing values: {missing_values_count.sum()}')
    print(f'The number of NaN values: {nan_values_count.sum()}')
    print(f'The number of zero values: {zero_values_count.sum()}')

    total_cells = np.prod(df.shape)
    total_missing = missing_values_count.sum()
    print(f'The percentage of missing values: {(total_missing / total_cells) * 100:.2f}%')
    print(f'The percentage of NaN values: {(nan_values_count.sum() / total_cells) * 100:.2f}%')
    print(f'The percentage of zero values: {(zero_values_count.sum() / total_cells) * 100:.2f}%')

    return missing_values_count, nan_values_count, zero_values_count


missing_values_count, nan_values_count, zero_values_count = analyze_data_quality(data_12[SENSOR_COLUMNS])

#2.2 Random or patterned (duvida)

#2.3 impute using interpolation, mean, median or drop - there are no missing values in the dataset, so no imputation is needed.
#handle missing values by interpolation
#data_interpolated = data_12.sort_values(by='timestamp').copy()
#data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']] = data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']].interpolate(method='linear', limit_direction='both')

#output_path_interpolated = Path(__file__).parent / "fault_detection_interpolated_dataset.json"
#data_interpolated.to_json(output_path_interpolated, orient="records", date_format="iso", indent=2)

#missing_values_count, nan_values_count, zero_values_count = analyze_data_quality(data_interpolated)

#state = data_interpolated['state']

#3 - Data Integration (already done)

#4 - Data transformation
#4.1 Feature construction - Create new features from existing ones, such as ratios, differences, or aggregations. For example, you could create a new feature that represents the difference between temperature and pressure, or the ratio of o2 to temperature. This can help capture relationships between variables that may not be apparent in the original features.
#{https://medium.com/@morepravin1989/feature-engineering-feature-construction-and-feature-fitting-explained-with-example-824128e9d886}
#The features already constructed are: group_encoded and split_encoded, which are binary features representing the group and split columns, respectively. These features can be used in machine learning models to predict the target variable (faulty or no_fault) based on the sensor readings.
#do i need more?

#4.2 Fix Skewness and Kurtosis
#4.2.1 Fix Skewness - If it is symmetric or nor.
def check_skewness(data_12):
    for col in ["temperature", "pressure", "o2"]:
        skewness = skew(data_12[col])
        print(f"{col}: Skewness of {col} = {skewness:.4f}")

    #it is said that the are in all of them more weight in the right tail
    for col in ["temperature", "pressure", "o2"]:
        mean_skew = data_12[col].mean()
        median_skew = data_12[col].median()
        std_skew = data_12[col].std()
        pearson_skew = 3 * (mean_skew - median_skew) / std_skew if std_skew != 0 else 0
        print(f"{col}: Pearson's skewness of {col} = {pearson_skew:.4f}")

    for col in ["temperature", "pressure", "o2"]:
        mean_skew = data_12[col].mean()
        median_skew = data_12[col].median()
        mode_skew = data_12[col].mode().iloc[0]
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data_12[col], fill=True)
        plt.axvline(mean_skew, color='r', linestyle='--', label='Mean')
        plt.axvline(median_skew, color='g', linestyle='-', label='Median')
        plt.axvline(mode_skew, color='b', linestyle=':', label='Mode')
        plt.title(f'Distribution of {col} with Mean, Median, and Mode')
        plt.xlabel(col)
        plt.legend()
        plt.show()

#they are all symetrical, but not the temperature, however that is explained by the normal increase of it. 
check_skewness(data_12)
#4.2.2 Fix Kurtosis - If it is normal or not.
def check_kurtosis(data_12):
    for col in ["temperature", "pressure", "o2"]:
        kurt = kurtosis(data_12[col])
        print(f"{col}: Kurtosis of {col} = {kurt:.4f}")

    for col in ["temperature", "pressure", "o2"]:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data_12[col], fill=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.show()

check_kurtosis(data_12)
#4.3 Normalized and standardized
#4.3.1 Normalization - Scale the features to a specific range, such as [0, 1] or [-1, 1]. This can help ensure that all features contribute equally to the distance calculations in K-Means clustering.
#def normalize_data(data_12):

#4.3.2 Standardization - Transform the features to have zero mean and unit variance, which is useful for algorithms that assume normally distributed data.
def standarlizer(robust_df):
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(robust_df[SENSOR_COLUMNS])
    normalized_df = robust_df.copy()
    normalized_df[SENSOR_COLUMNS] = normalized_values

    output_path_standardized = Path(__file__).parent / "fault_detection_standardized_dataset.json"
    normalized_df.to_json(output_path_standardized, orient="records", indent=2, date_format="iso")
    print(f"Standardized {normalized_df.shape[0]} instances x {len(SENSOR_COLUMNS)} features")
    print(f"Saved standardized data to {output_path_standardized}")



#4.3.3 Robust Scaling - Scale the features using statistics that are robust to outliers, such as the median and interquartile range. This can help reduce the influence of extreme values on the scaling process.
def robust_scaler(data_12): #uso o robust scaler para reduzir a influência de outliers, mas não sei se é o melhor para o meu caso, pois os outliers podem ser importantes para a detecção de falhas.
    scaler = preprocessing.RobustScaler()
    robust_values = scaler.fit_transform(data_12[SENSOR_COLUMNS])
    robust_df = data_12.copy()
    robust_df[SENSOR_COLUMNS] = robust_values

    output_path_robust = Path(__file__).parent / "fault_detection_robust_dataset.json"
    robust_df.to_json(output_path_robust, orient="records", indent=2, date_format="iso")
    print(f"Robust scaled {robust_df.shape[0]} instances x {len(SENSOR_COLUMNS)} features")
    print(f"Saved robust scaled data to {output_path_robust}")
    return robust_df


robust_df = robust_scaler(data_12)
standarlizer(robust_df)

#5 Data Reduction
#5.1 Drop correlated variables
#Pearson Correlaction
pearson_corr = robust_df[SENSOR_COLUMNS].corr(method='pearson')
print(pearson_corr)

sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

#no correlated variables to drop

#5.2 Feature selection {https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/}
#i dont think that is needed, but ask

#5.3 PCA (or other dimensionality reduction techniques)
#apply the standard PCA
def apply_pca(robust_df, n_components=3):
    pca = PCA(n_components=3)
    for col in SENSOR_COLUMNS:
        
    X_pca = pca.fit_transform(robust_df[SENSOR_COLUMNS])

X_train, X_test, y_train, y_test = train_test_split(X_pca, robust_df['group_encoded'], test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fault', 'Faulty'], yticklabels=['No Fault', 'Faulty'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


