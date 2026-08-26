from cProfile import label
import json
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

import numpy as np
 
with open("C:\\Users\\maria\\OneDrive\\Desktop\\bolsa1\\KMEANS\\PREPROCESSINGDATA\\bias_TEMP_appropriate_dataset_all_runs.json") as f:
    dataset = pd.DataFrame(json.load(f))


SENSOR_COLUMNS = ["temperature", "pressure", "o2"]
#0- Make it like a table with with timestamp, three columns that represent the three sensors and a column that represents the type of fault that is present in the system. The timestamp will be generated using the sample number and the sampling frequency (100 Hz). The three columns will be filled with the values of the three sensors for each sample.

#1-DATA CLEANING
#1.1-remove duplicates, used the {https://www.linkedin.com/pulse/data-cleaning-python-handling-duplicates-pandas-bennett-alexander-ay7xe/}
data_noduplicates = dataset.drop_duplicates()
data_noduplicates.head()
print(f'The number of duplicates removed: {len(dataset) - len(data_noduplicates)}   ')

#1.2- remove irrelevant rows and columns
data_12 = data_noduplicates.drop(columns=['split', 'group'])

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


def plot_outliers_boxplot(dataset, output_path="outliers_boxplot.png"):
    df = dataset[SENSOR_COLUMNS]
    zscore_outliers = detect_outliers_zscore(dataset)
    iqr_outliers = detect_outliers_iqr(dataset)

    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    ax.set_title("Sensor readings: IQR vs. Z-score outlier detection")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved boxplot to {output_path}")
    plt.close(fig)


def export_outliers_dataset(dataset, output_path="outliers_dataset.csv"):
    df = dataset.copy()
    zscore_outliers = detect_outliers_zscore(dataset)
    iqr_outliers = detect_outliers_iqr(dataset)

    result = df.copy()
    any_outlier = pd.Series(False, index=df.index)
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
def plot_winsorize_boxplot(dataset, limits=(0.05, 0.05), output_path="winsorize_boxplot.png"):
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
def export_winsorized_dataset(dataset, limits=(0.05, 0.05), output_path="winsorized_dataset.csv", json_output_path="winsorized_dataset.json"):
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

#2.2 Random or patterned

#2.3 impute using interpolation, mean, median or drop
#handle missing values by interpolation
#data_interpolated = data_12.sort_values(by='timestamp').copy()
#data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']] = data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']].interpolate(method='linear', limit_direction='both')

#output_path_interpolated = Path(__file__).parent / "fault_detection_interpolated_dataset.json"
#data_interpolated.to_json(output_path_interpolated, orient="records", date_format="iso", indent=2)

#missing_values_count, nan_values_count, zero_values_count = analyze_data_quality(data_interpolated)

#state = data_interpolated['state']

#3 - Data Integration (already done)

#4 - Data transformation
#4.1 Feature construction 

#4.2 Fix Skewness and Kurtosis

#4.3 Normalized and standardized

#5 Data Reduction
#5.1 Drop correlated variables
#5.2 Feature selection
#5.3 PCA (or other dimensionality reduction techniques)




