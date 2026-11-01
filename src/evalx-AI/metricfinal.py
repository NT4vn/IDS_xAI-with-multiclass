from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# metric.py writes one workbook per dataset/model. Average numeric scores over
# models so each dataset has one comparable value per metric and XAI method.
RESULT_DIR = Path("metric1")
OUTPUT_DIR = Path("xai_metric_plots")
METHODS = ["KernelSHAP", "TreeSHAP", "DeepSHAP", "LIME"]

files = sorted(RESULT_DIR.glob("*_FINAL.xlsx"))
if not files:
    raise FileNotFoundError(
        f"No per-model XAI results found in {RESULT_DIR.resolve()}; run metric.py first."
    )

frames = []
for path in files:
    df = pd.read_excel(path)
    required = {"Model", "Metric", *METHODS}
    if not required.issubset(df.columns):
        print(f"Skipping {path}: missing expected columns")
        continue
    dataset = path.name.split("_", 1)[0]
    df["Dataset"] = dataset
    for method in METHODS:
        df[method] = pd.to_numeric(df[method], errors="coerce")
    frames.append(df[["Dataset", "Model", "Metric", *METHODS]])

if not frames:
    raise ValueError(f"No usable XAI result workbooks found in {RESULT_DIR.resolve()}.")

all_results = pd.concat(frames, ignore_index=True)
avg = all_results.groupby(["Dataset", "Metric"], as_index=False)[METHODS].mean(numeric_only=True)
avg.to_excel(RESULT_DIR / "FINAL_AVG_BY_DATASET.xlsx", index=False)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
metrics = avg["Metric"].dropna().unique().tolist()

for metric in metrics:
    plot_df = avg[avg["Metric"] == metric].melt(
        id_vars=["Dataset", "Metric"], value_vars=METHODS,
        var_name="XAI_Method", value_name="Score"
    ).dropna(subset=["Score"])
    if plot_df.empty:
        continue
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Dataset", y="Score", hue="XAI_Method")
    plt.title(f"{metric} Comparison Across XAI Methods")
    plt.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(metric))
    plt.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=300)
    plt.close()

cols = 3
rows = max(1, (len(metrics) + cols - 1) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows), squeeze=False)
axes = axes.flatten()
for i, metric in enumerate(metrics):
    plot_df = avg[avg["Metric"] == metric].melt(
        id_vars=["Dataset", "Metric"], value_vars=METHODS,
        var_name="XAI_Method", value_name="Score"
    ).dropna(subset=["Score"])
    if not plot_df.empty:
        sns.barplot(data=plot_df, x="Dataset", y="Score", hue="XAI_Method", ax=axes[i])
        axes[i].set_title(metric)
        axes[i].tick_params(axis="x", rotation=20)
for ax in axes[len(metrics):]:
    fig.delaxes(ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "All_XAI_Metrics.png", dpi=300)
plt.close(fig)
print(f"Saved dataset averages to {RESULT_DIR / 'FINAL_AVG_BY_DATASET.xlsx'}")
print(f"Saved plots to {OUTPUT_DIR}")
