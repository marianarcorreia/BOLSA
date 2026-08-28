import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.ensemble import IsolationForest
from scipy.stats.mstats import winsorize


dataset_path = "fault_detection_interpolated.json"
SENSOR_COLUMNS = ["sensor_1", "sensor_2", "sensor_3"]

# colors: slot 1 (blue) = IQR, slot 2 (orange) = Z-score
COLOR_IQR = "#2a78d6"
COLOR_ZSCORE = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"


#detect outliers using z-score method
def detect_outliers_zscore(dataset_path, threshold=3):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = z_scores > threshold
    # Ensure a pandas DataFrame is returned with the original index/columns
    return pd.DataFrame(outliers, index=df.index, columns=df.columns)
    


#detect outliers using IQR method
def detect_outliers_iqr(dataset_path):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df < lower_bound) | (df > upper_bound)
    # Ensure a pandas DataFrame is returned with the original index/columns
    return pd.DataFrame(outliers, index=df.index, columns=df.columns)


def plot_outliers_boxplot(dataset_path, output_path="outliers_boxplot.png"):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    zscore_outliers = detect_outliers_zscore(dataset_path)
    iqr_outliers = detect_outliers_iqr(dataset_path)

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


def export_outliers_dataset(dataset_path, output_path="outliers_dataset.csv"):
    df = pd.read_json(dataset_path, convert_dates=False)
    zscore_outliers = detect_outliers_zscore(dataset_path)
    iqr_outliers = detect_outliers_iqr(dataset_path)

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
def winsorize_sensors(dataset_path, limits=(0.05, 0.05)):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    result = df.copy()
    for col in SENSOR_COLUMNS:
        clipped = winsorize(df[col].to_numpy(), limits=limits)
        # winsorize returns a MaskedArray; unwrap it back into a plain float column
        result[col] = np.asarray(clipped)
    return result


#compare winsorizing vs. deleting the same tail rows, per sensor column
def compare_winsorize_vs_deletion(dataset_path, limits=(0.05, 0.05)):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    winsorized = winsorize_sensors(dataset_path, limits=limits)
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
def plot_winsorize_boxplot(dataset_path, limits=(0.05, 0.05), output_path="winsorize_boxplot.png"):
    df = pd.read_json(dataset_path, convert_dates=False)[SENSOR_COLUMNS]
    winsorized = winsorize_sensors(dataset_path, limits=limits)

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
def export_winsorized_dataset(dataset_path, limits=(0.05, 0.05), output_path="winsorized_dataset.csv", json_output_path="winsorized_dataset.json"):
    df = pd.read_json(dataset_path, convert_dates=False)
    winsorized = winsorize_sensors(dataset_path, limits=limits)

    result = df.copy()
    for col in SENSOR_COLUMNS:
        result[col] = winsorized[col]

    result.to_csv(output_path, index=False)
    print(f"Saved winsorized dataset ({len(result)} rows) to {output_path}")

    result.to_json(json_output_path, orient="records", indent=2, date_format="iso")
    print(f"Saved winsorized dataset ({len(result)} rows) to {json_output_path}")
    return result


if __name__ == "__main__":
    plot_outliers_boxplot(dataset_path)
    export_outliers_dataset(dataset_path)
    compare_winsorize_vs_deletion(dataset_path)
    plot_winsorize_boxplot(dataset_path)
    export_winsorized_dataset(dataset_path)
    compare_winsorize_vs_deletion(dataset_path)
    plot_winsorize_boxplot(dataset_path)


