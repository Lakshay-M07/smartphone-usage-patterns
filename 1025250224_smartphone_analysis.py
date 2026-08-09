import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASET_PATH = Path(__file__).resolve().parent / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
ROLL = "smartphone_usage"

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def zscore(series):
    values = np.asarray(series, dtype=float)
    return (values - values.mean()) / values.std(ddof=0)


def main():
    print("=" * 65)
    print("  SMARTPHONE USAGE & ADDICTION — DATA ANALYSIS REPORT")
    print("=" * 65)

    # 1. Load data
    df_raw = pd.read_csv(DATASET_PATH)
    print(f"\n[1] Dataset loaded → {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print("\nColumn names & dtypes:")
    print(df_raw.dtypes.to_string())

    # 2. Data cleaning
    print("\n" + "-" * 65)
    print("[2] DATA CLEANING")
    print("-" * 65)
    df = df_raw.copy()
    df.drop(columns=["transaction_id", "user_id"], inplace=True, errors="ignore")
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    df["addiction_level"] = df["addiction_level"].fillna("Unknown")
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    df["addiction_score"] = df["addiction_level"].map(
        {"Unknown": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    )
    df["stress_score"] = df["stress_level"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    df["academic_impact"] = (df["academic_work_impact"] == "Yes").astype(int)

    num_cols = [
        "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "work_study_hours", "sleep_hours", "notifications_per_day",
        "app_opens_per_day", "weekend_screen_time",
    ]
    print("\nBasic statistics after cleaning:")
    print(df[num_cols].describe().round(2).to_string())

    # 3. Demographics
    print("\n" + "-" * 65)
    print("[3] EXPLORATORY DATA ANALYSIS — DEMOGRAPHICS")
    print("-" * 65)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Fig 1 — Demographic Overview", fontweight="bold")

    gender_counts = df["gender"].value_counts()
    axes[0].pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Gender Distribution")

    axes[1].hist(df["age"], bins=18, edgecolor="white")
    axes[1].set_title("Age Distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Count")

    addiction_counts = df["addiction_level"].value_counts().reindex(
        ["Mild", "Moderate", "Severe", "Unknown"], fill_value=0
    )
    axes[2].bar(addiction_counts.index, addiction_counts.values)
    axes[2].set_title("Addiction Level Distribution")
    axes[2].set_xlabel("Addiction Level")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig1_demographics.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 1 saved.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 2 — Stress Level & Academic Impact", fontweight="bold")
    stress_counts = df["stress_level"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    axes[0].bar(stress_counts.index, stress_counts.values)
    axes[0].set_title("Stress Level Distribution")
    axes[0].set_ylabel("Count")
    impact_counts = df["academic_work_impact"].value_counts()
    axes[1].pie(impact_counts, labels=impact_counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Academic / Work Impact")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig2_stress_impact.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 2 saved.")

    # 4. Covariance and correlation
    print("\n" + "-" * 65)
    print("[4] COVARIANCE AND CORRELATION ANALYSIS")
    print("-" * 65)
    cov_cols = [
        "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "sleep_hours", "notifications_per_day",
    ]
    print("\nCovariance Matrix:\n")
    print(df[cov_cols].cov().round(3).to_string())
    print("\nCorrelation Matrix:\n")
    print(df[cov_cols].corr().round(3).to_string())

    # 5. Outlier detection
    print("\n" + "-" * 65)
    print("[5] OUTLIER DETECTION")
    print("-" * 65)
    print("\nOutliers via IQR method:")
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        print(f"  {col:<30} → {mask.sum()} outliers ({mask.mean() * 100:.2f}%)")

    print("\nOutliers via Z-score (|z| > 3):")
    for col in num_cols:
        z = np.abs(zscore(df[col].dropna()))
        print(f"  {col:<30} → {(z > 3).sum()} outliers")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle("Fig 3 — Box Plots for Outlier Visualisation", fontweight="bold")
    for ax, col in zip(axes.flatten(), num_cols):
        ax.boxplot(df[col].dropna().values, vert=True, patch_artist=True)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig3_boxplots.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 3 saved.")

    # 6. Group-wise comparisons
    print("\n" + "-" * 65)
    print("[6] GROUP-WISE COMPARISONS")
    print("-" * 65)
    usage_cols = [
        "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "work_study_hours", "sleep_hours", "app_opens_per_day",
    ]
    known = df[df["addiction_level"] != "Unknown"]
    means = known.groupby("addiction_level")[usage_cols].mean().reindex(
        ["Mild", "Moderate", "Severe"]
    )
    print("\nMean usage metrics by addiction level:\n")
    print(means.round(2).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fig 4 — Mean Usage Metrics by Addiction Level", fontweight="bold")
    for ax, col in zip(axes.flatten(), usage_cols):
        ax.bar(means.index, means[col])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Addiction Level")
        ax.set_ylabel("Mean Value")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig4_group_means.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 4 saved.")

    # 7. Weekend vs weekday
    print("\n" + "-" * 65)
    print("[7] WEEKEND VS WEEKDAY SCREEN TIME")
    print("-" * 65)
    weekday = df["daily_screen_time_hours"].mean()
    weekend = df["weekend_screen_time"].mean()
    print(f"\nWeekday avg: {weekday:.2f} h | Weekend avg: {weekend:.2f} h | Difference: {weekend - weekday:.2f} h")
    difference = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(difference, bins=40, edgecolor="white")
    ax.axvline(0, linestyle="--", linewidth=1.5, label="No difference")
    ax.axvline(difference.mean(), linestyle="-.", linewidth=1.5,
               label=f"Mean diff = {difference.mean():.2f} h")
    ax.set_title("Fig 5 — Weekend vs Weekday Screen Time Difference", fontweight="bold")
    ax.set_xlabel("Difference (hours)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig5_weekend_vs_weekday.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 5 saved.")

    # 8. Screen-time breakdown
    print("\n" + "-" * 65)
    print("[8] SCREEN TIME BREAKDOWN BY CATEGORY")
    print("-" * 65)
    categories = ["social_media_hours", "gaming_hours", "work_study_hours"]
    labels = ["Social Media", "Gaming", "Work/Study"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig 6 — Screen Time Breakdown by Activity", fontweight="bold")
    axes[0].pie(df[categories].mean(), labels=labels, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Average Split of Screen Time")
    breakdown = known.groupby("addiction_level")[categories].mean().reindex(
        ["Mild", "Moderate", "Severe"]
    )
    x = np.arange(len(breakdown))
    width = 0.25
    for i, col in enumerate(categories):
        axes[1].bar(x + i * width, breakdown[col], width=width, label=labels[i])
    axes[1].set_title("Activity Hours by Addiction Level")
    axes[1].set_ylabel("Mean Hours")
    axes[1].set_xlabel("Addiction Level")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(breakdown.index)
    axes[1].legend(title="Activity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{ROLL}_fig6_usage_breakdown.png", bbox_inches="tight")
    plt.close()
    print("  → Fig 6 saved.")

    # 9. Summary
    print("\n" + "-" * 65)
    print("[9] FINAL SUMMARY STATISTICS")
    print("-" * 65)
    summary = df[num_cols].agg(["mean", "median", "std", "min", "max"]).T
    summary.columns = ["Mean", "Median", "Std Dev", "Min", "Max"]
    print("\n", summary.round(3).to_string())

    print("\n" + "=" * 65)
    print("  CONCLUSIONS")
    print("=" * 65)
    print("""
  1. Screen time is strongly associated with addiction level.
  2. Social media contributes substantially to non-work screen time.
  3. Sleep shows an inverse relationship with heavier usage patterns.
  4. Screen time, social media, and gaming tend to move together.
  5. Weekend usage can be compared directly with weekday usage.
  6. High notification and app-opening counts highlight heavy users.
""")
    print("  All figures saved. Analysis complete.")


if __name__ == "__main__":
    main()
