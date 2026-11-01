import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# ────────────────────────────────────────────────
# CẤU HÌNH THƯ MỤC & ĐƯỜNG DẪN
# ────────────────────────────────────────────────
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DATASET_FILES = [
    {
        "name": "CIC17",
        "Y_test": "preprocessed_multi_cic17_y_test.npy",
        "class_names": ["Botnet / Infiltration", "Brute Force", "DDoS", "DoS", "Normal", "Other", "Port Scan", "Web Attack"],
    },
    {
        "name": "NSL-KDD",
        "Y_test": "preprocessed_multi_nsl-kdd_y_test.npy",
        "class_names": ["DoS", "Normal", "Other", "Probe", "R2L", "U2R"],
    },
    {
        "name": "SIGNAL2022",
        "Y_test": "preprocessed_multi_signal22_y_test.npy",
        "class_names": ["DoS", "Malware", "Normal", "Port Scan"],
    },
]

# List danh sách các mô hình trong bài nghiên cứu
MODEL_NAMES = ["AdaBoost", "Random Forest", "LightGBM", "MLP", "DNN", "CNN", "LSTM"]
RESULT_FILES = [
    RESULTS_DIR / "model_results.xlsx",
    RESULTS_DIR / "deep_learning_results.xlsx",
]

# ────────────────────────────────────────────────
# HÀM TÍNH CÁC CHỈ SỐ EVALUATION (IDS METRICS)
# ────────────────────────────────────────────────
def compute_ids_metrics(y_true, y_pred):
    """Return macro one-vs-rest IDS metrics and per-class metrics."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    dr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) != 0)
    faf1 = np.divide(2 * tp, 2 * tp + fn + 2 * fp, out=np.zeros_like(tp), where=(2 * tp + fn + 2 * fp) != 0)
    summary = {
        "Detection Rate": float(np.mean(dr)),
        "FPR": float(np.mean(fpr)),
        "FAF1": float(np.mean(faf1)),
        "FAR": float(np.mean(fpr)),
        "Precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    per_class = {}
    for i, label in enumerate(labels):
        per_class[f"Class_{label}_DR"] = float(dr[i])
        per_class[f"Class_{label}_FPR"] = float(fpr[i])
        per_class[f"Class_{label}_FAF1"] = float(faf1[i])
    return summary, per_class

def normalize_dataset_name(value):
    if value is None:
        return value
    value = str(value).strip()
    normalized = value.lower().replace("_", "-")
    if normalized == "cic17":
        return "CIC17"
    if normalized in {"nsl-kdd", "nsl kdd"}:
        return "NSL-KDD"
    if normalized in {"signal2022", "signal-2022", "signal22"}:
        return "SIGNAL2022"
    return value

def normalize_model_name(value):
    if value is None:
        return value
    value = str(value).strip()
    if value.lower() in {"randomforest", "random forest"}:
        return "Random Forest"
    if value.lower() == "adaboost":
        return "AdaBoost"
    return value

def load_accuracy_results_from_excel():
    """Đọc dữ liệu Accuracy và các metric thực tế từ các file Excel đã đánh giá xong."""
    frames = []

    for file_path in RESULT_FILES:
        if not file_path.exists():
            continue

        try:
            df = pd.read_excel(file_path)
        except Exception as exc:
            print(f"⚠️ Không đọc được {file_path}: {exc}")
            continue

        if df.empty or {"Dataset", "Model"} - set(df.columns):
            continue

        keep_cols = ["Dataset", "Model"]
        if "Accuracy" in df.columns:
            keep_cols.append("Accuracy")
        # Thêm các metric thực tế nếu có trong file
        metric_cols = [
            "Detection Rate", "DetectionRate",
            "FPR", "FAF1", "FAR",
            "Recall", "Precision", "TrainTime"
        ]
        for col in metric_cols:
            if col in df.columns and col not in keep_cols:
                keep_cols.append(col)

        df = df[keep_cols].copy()

        if "DetectionRate" in df.columns and "Detection Rate" not in df.columns:
            df.rename(columns={"DetectionRate": "Detection Rate"}, inplace=True)

        df["Dataset"] = df["Dataset"].map(normalize_dataset_name)
        df["Model"] = df["Model"].map(normalize_model_name)
        df = df.dropna(subset=["Dataset", "Model"]).copy()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Không tìm thấy file kết quả Accuracy. Hãy chạy quá trình huấn luyện/evaluation trước khi vẽ biểu đồ."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Dataset", "Model"], keep="last")

    dataset_order = ["CIC17", "NSL-KDD", "SIGNAL2022"]
    model_order = ["AdaBoost", "Random Forest", "LightGBM", "MLP", "DNN", "CNN", "LSTM"]

    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values(["Dataset", "Model"]).reset_index(drop=True)
    return df

# ────────────────────────────────────────────────
# HÀM VẼ BIỂU ĐỒ SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY)
# ────────────────────────────────────────────────
def plot_accuracy_comparison(df_results):
    """
    Vẽ biểu đồ cột nhóm (Grouped Bar Chart) so sánh Accuracy của các mô hình trên 2 tập dữ liệu
    """
    df_results = df_results[(df_results["Dataset"].isin(["CIC17", "NSL-KDD", "SIGNAL2022"]))].copy()
    df_results["Model"] = pd.Categorical(df_results["Model"], categories=MODEL_NAMES, ordered=True)
    df_results = df_results.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=df_results,
        x="Model",
        y="Accuracy",
        hue="Dataset",
        palette=["#4E79A7", "#F28E2B", "#59A14F"],
        order=MODEL_NAMES
    )

    plt.title("So sánh Độ chính xác (Accuracy) của các Mô hình trên 2 Tập dữ liệu", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mô hình (Model)", fontsize=12, labelpad=10)
    plt.ylabel("Độ chính xác (Accuracy)", fontsize=12)
    plt.ylim(0, 1.02)
    plt.legend(title="Tập dữ liệu", title_fontsize="11", loc="lower right")
    plt.xticks(rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9, color="black")

    plt.tight_layout()

    save_path = PLOT_DIR / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ so sánh độ chính xác tại: {save_path}")

# ────────────────────────────────────────────────
# LUỒNG CHÍNH (MAIN EVALUATION)
# ────────────────────────────────────────────────
def evaluate_models(predictions_dict=None):
    """
    predictions_dict: Dictionary chứa kết quả dự đoán y_pred đã lưu từ trước.
    Ví dụ cấu trúc: predictions_dict['CIC17']['Random Forest'] = y_pred_array
    """
    if predictions_dict is not None:
        results = []

        for ds in DATASET_FILES:
            ds_name = ds["name"]
            print(f"\n================ Đang tính toán trên tập: {ds_name} ================")

            y_test_path = DATA_DIR / ds["Y_test"]
            if not y_test_path.exists():
                print(f"⚠️  Không tìm thấy tệp nhãn: {y_test_path}")
                continue

            y_test = np.load(y_test_path)
            if y_test.ndim > 1:
                y_test = np.argmax(y_test, axis=1)

            for model_name in MODEL_NAMES:
                if ds_name not in predictions_dict or model_name not in predictions_dict[ds_name]:
                    print(f"⚠️ Không có dự đoán cho {ds_name}/{model_name}; bỏ qua.")
                    continue
                y_pred = np.asarray(predictions_dict[ds_name][model_name])
                if y_pred.ndim > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                acc = accuracy_score(y_test, y_pred)
                summary, per_class = compute_ids_metrics(y_test, y_pred)

                results.append({
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Accuracy": acc,
                    **summary,
                    **per_class,
                })

        df_results = pd.DataFrame(results)
        excel_path = RESULTS_DIR / "ids_evaluation_metrics.xlsx"
        df_results.to_excel(excel_path, index=False)
        print(f"\n✅ Đã lưu bảng chỉ số đánh giá ra tệp Excel: {excel_path}")
        print("\n" + "="*70)
        print(" BẢNG ACCURACY ĐƯỢC TÍNH TỪ DỰ ĐOÁN ")
        print("="*70)
        print(df_results[["Dataset", "Model", "Accuracy"]].to_string(index=False))
        plot_accuracy_comparison(df_results[["Dataset", "Model", "Accuracy"]])
        return df_results

    try:
        df_results = load_accuracy_results_from_excel()
    except FileNotFoundError as exc:
        print(f"⚠️ {exc}")
        df_results = pd.DataFrame(columns=["Dataset", "Model", "Accuracy"])

    print("\n" + "="*70)
    print(" BẢNG ACCURACY ĐƯỢC TẢI TỪ FILE KẾT QUẢ ")
    print("="*70)
    print(df_results[["Dataset", "Model", "Accuracy"]].to_string(index=False))
    plot_accuracy_comparison(df_results[["Dataset", "Model", "Accuracy"]])
    return df_results[["Dataset", "Model", "Accuracy"]]

if __name__ == "__main__":
    evaluate_models()
