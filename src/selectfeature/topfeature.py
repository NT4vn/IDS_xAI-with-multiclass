import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

# ==========================
# DATA DIR & OUTPUT DIR
# ==========================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
OUTPUT_DIR = Path(r"D:\NLATTT\kqua\selection")  # Lưu kết quả đồ thị / bảng biểu trực tiếp vào thư mục này

# Tạo thư mục đầu ra nếu chưa tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# DATASET FILE MAP
# ==========================
DATASET_FILES = [
    {"name": "CIC17", "prefix": "preprocessed_multi_cic17"},
    {"name": "NSL-KDD", "prefix": "preprocessed_multi_nsl-kdd"},
    {"name": "SIGNAL2022", "prefix": "preprocessed_multi_signal22"},
]

# ==========================
# LOAD DATA
# ==========================
def load_dataset(info):
    name = info["name"]
    prefix = info["prefix"]

    X_train_path = DATA_DIR / f"{prefix}_X_train.npy"
    Y_train_path = DATA_DIR / f"{prefix}_y_train.npy"
    feature_path = DATA_DIR / f"{prefix}_feature_names.pkl"

    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)

    # Load Tên Features Thật
    if feature_path.exists():
        feature_names = joblib.load(feature_path)
        print("✅ Loaded real feature names")
    else:
        print("⚠️ Không tìm thấy feature_names → Fallback feat_i")
        feature_names = [f"feat_{i+1}" for i in range(X_train.shape[1])]

    print("\n" + "=" * 50)
    print("DATASET MULTI-CLASS:", name)
    print("=" * 50)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    # Chuyển đổi X_train thành DataFrame với đúng tên cột
    X_train = pd.DataFrame(X_train, columns=feature_names)

    return X_train, Y_train

# ==========================
# FEATURE SCORING (MULTI-CLASS OPTIMIZED)
# ==========================
def select_features(X_train, Y_train):
    # Xử lý Y_train để phù hợp với f_classif (cần nhãn 1D categorical / integer)
    if Y_train.ndim > 1:
        if Y_train.shape[1] > 1:
            # Trường hợp Y_train đang là One-Hot Encoded
            Y_train = np.argmax(Y_train, axis=1)
        else:
            # Trường hợp Y_train shape (N, 1)
            Y_train = Y_train.ravel()

    # f_classif (ANOVA F-value) hoạt động rất tốt cho phân loại đa lớp (multi-class)
    selector = SelectKBest(f_classif, k="all")
    selector.fit(X_train, Y_train)

    scores = selector.scores_

    # Thay thế NaN / Inf nếu có bằng 0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame({"feature": X_train.columns, "score": scores})

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df

# ==========================
# VISUALIZATION FUNCTIONS
# ==========================
def plot_bar(df, dataset, top_k):
    top = df.head(top_k)

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["score"])
    plt.gca().invert_yaxis()

    plt.title(f"Top {top_k} Multi-Class Feature Importance ({dataset})")
    plt.xlabel("ANOVA F-Score")
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"{dataset}_bar_top{top_k}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(df, dataset, top_k):
    top = df.head(top_k).set_index("feature")

    plt.figure(figsize=(6, 8))
    sns.heatmap(top[["score"]], annot=True, fmt=".2f", cmap="viridis")

    plt.title(f"Multi-Class Feature Heatmap ({dataset})")
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"{dataset}_heatmap_top{top_k}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_table(df, dataset, top_k):
    top = df.head(top_k)

    table_df = pd.DataFrame(
        {
            "Rank": range(1, top_k + 1),
            "Feature": top["feature"],
            "F-Score": np.round(top["score"], 2),
        }
    )

    fig, ax = plt.subplots(figsize=(12, top_k * 0.45))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title(f"Multi-Class Feature Ranking ({dataset})", pad=20)

    save_path = OUTPUT_DIR / f"{dataset}_feature_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    TOP_K = 20

    for i, dataset_info in enumerate(DATASET_FILES):
        name = dataset_info["name"]
        print(f"\nProcessing dataset index {i}: {name}")

        try:
            # 1. LOAD DATA
            X_train, Y_train = load_dataset(dataset_info)

            # 2. FEATURE SCORING
            df_scores = select_features(X_train, Y_train)

            # 3. PRINT TOP-K FEATURES
            print("\n" + "=" * 70)
            print(f"TOP {TOP_K} MULTI-CLASS FEATURES - {name}")
            print("=" * 70)

            top_features = df_scores.head(TOP_K)
            for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
                print(f"{rank:2d}. {row['feature']:<40} Score: {row['score']:.2f}")

            # 4. VISUALIZATION
            plot_bar(df_scores, name, TOP_K)
            plot_heatmap(df_scores, name, TOP_K)
            plot_table(df_scores, name, TOP_K)

            print(f"\n✅ Finished processing & saved plots for: {name}")

        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
