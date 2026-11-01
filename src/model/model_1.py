import os
import pickle
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

# Ẩn các cảnh báo không cần thiết từ Scikit-learn & LightGBM
warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────
# DATA & DIRECTORY PATHS
# ────────────────────────────────────────────────
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
MODEL_DIR = Path("saved_models")
PLOT_DIR = Path("plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR / "confusion", exist_ok=True)
os.makedirs(PLOT_DIR / "roc", exist_ok=True)
os.makedirs(PLOT_DIR / "loss", exist_ok=True)

# 📌 CLASS_NAMES_MAP ĐƯỢC ĐỒNG BỘ THEO DỮ LIỆU CỦA BẠN
CLASS_NAMES_MAP = {
    "CIC17": [
        "Botnet / Infiltration",  # Class 0
        "Brute Force",  # Class 1
        "DDoS",  # Class 2
        "DoS",  # Class 3
        "Normal",  # Class 4
        "Other",  # Class 5
        "Port Scan",  # Class 6
        "Web Attack",  # Class 7
    ],
    "NSL-KDD": [
        "DoS",  # Class 0
        "Normal",  # Class 1
        "Other",  # Class 2
        "Probe",  # Class 3
        "R2L",  # Class 4
        "U2R",  # Class 5
    ],
    "SIGNAL2022": [
        "DoS",  # Class 0
        "Malware",  # Class 1
        "Normal",  # Class 2
        "Port Scan",  # Class 3
    ],
}

DATASET_FILES = [
    {
        "name": "CIC17",
        "X_train": "preprocessed_multi_cic17_X_train.npy",
        "X_test": "preprocessed_multi_cic17_X_test.npy",
        "Y_train": "preprocessed_multi_cic17_y_train.npy",
        "Y_test": "preprocessed_multi_cic17_y_test.npy",
    },
    {
        "name": "NSL-KDD",
        "X_train": "preprocessed_multi_nsl-kdd_X_train.npy",
        "X_test": "preprocessed_multi_nsl-kdd_X_test.npy",
        "Y_train": "preprocessed_multi_nsl-kdd_y_train.npy",
        "Y_test": "preprocessed_multi_nsl-kdd_y_test.npy",
    },
    {
        "name": "SIGNAL2022",
        "X_train": "preprocessed_multi_signal22_X_train.npy",
        "X_test": "preprocessed_multi_signal22_X_test.npy",
        "Y_train": "preprocessed_multi_signal22_y_train.npy",
        "Y_test": "preprocessed_multi_signal22_y_test.npy",
    },
]

# ────────────────────────────────────────────────
# METRICS & EVALUATION
# ────────────────────────────────────────────────
def compute_ids_metrics(y_true, y_pred, num_classes):
    """Tính FPR và FAP theo one-vs-rest, rồi lấy trung bình macro."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    with np.errstate(divide="ignore", invalid="ignore"):
        fpr_per_class = np.divide(
            fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) != 0
        )
        fap_per_class = np.divide(
            fp, fp + tp, out=np.zeros_like(fp, dtype=float), where=(fp + tp) != 0
        )

    return {
        "FPR_macro": float(np.mean(fpr_per_class)),
        "FAP_macro": float(np.mean(fap_per_class)),
    }

def load_dataset(ds):
    print(f"\n→ Loading dataset: {ds['name']}")
    X_train = np.load(DATA_DIR / ds["X_train"])
    X_test = np.load(DATA_DIR / ds["X_test"])
    y_train = np.load(DATA_DIR / ds["Y_train"])
    y_test = np.load(DATA_DIR / ds["Y_test"])

    if y_train.ndim > 1:
        y_train = (
            np.argmax(y_train, axis=1) if y_train.shape[1] > 1 else y_train.ravel()
        )
    if y_test.ndim > 1:
        y_test = (
            np.argmax(y_test, axis=1) if y_test.shape[1] > 1 else y_test.ravel()
        )

    print(
        f" Shape → X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
    )
    return X_train, X_test, y_train, y_test

# ────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ────────────────────────────────────────────────
def plot_confusion_matrix(cm, dataset, model_name, class_names=None):
    plt.figure(figsize=(10, 8))
    if class_names is None or len(class_names) != cm.shape[0]:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{dataset} – {model_name} Multi-Class Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "confusion" / f"{dataset}_{model_name}_cm.png", dpi=150
    )
    plt.close()

def plot_roc(y_true, y_score, dataset, model_name, num_classes):
    if y_score is None:
        return
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if num_classes == 2 and y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Micro-avg AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{dataset} – {model_name} ROC Curve (Multi-Class)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc" / f"{dataset}_{model_name}_roc.png", dpi=150)
    plt.close()

def plot_loss(history, dataset, model):
    if history is None:
        return
    hist = history.history
    plt.figure(figsize=(6, 5))
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.title(f"{dataset} - {model} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss" / f"{dataset}_{model}_loss.png", dpi=120)
    plt.close()

# ────────────────────────────────────────────────
# DEEP LEARNING MODEL BUILDERS
# ────────────────────────────────────────────────
def train_dnn(X_train, y_train, X_test, num_classes):
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train_cat,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    pred = model.predict(X_test, verbose=0)
    return model, pred, history

def train_cnn(X_train, y_train, X_test, num_classes):
    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(64, 3, activation="relu", padding="same"),
        Conv1D(128, 3, activation="relu", padding="same"),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train_r,
        y_train_cat,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    pred = model.predict(X_test_r, verbose=0)
    return model, pred, history

def train_lstm(X_train, y_train, X_test, num_classes):
    X_train_r = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_r = X_test.reshape(X_test.shape[0], 1, X_train.shape[1])
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        LSTM(96),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train_r,
        y_train_cat,
        epochs=15,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    pred = model.predict(X_test_r, verbose=0)
    return model, pred, history

def get_models(num_classes):
    lgb_params = {
        "n_estimators": 150,
        "max_depth": 10,
        "learning_rate": 0.1,
        "n_jobs": 4,
        "verbose": -1,
        "random_state": 42,
        "objective": "multiclass" if num_classes > 2 else "binary",
    }
    if num_classes > 2:
        lgb_params["num_class"] = num_classes

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=4, random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(**lgb_params),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=80, random_state=42),
        "DNN": "DNN",
        "CNN": "CNN",
        "LSTM": "LSTM",
    }

# ────────────────────────────────────────────────
# MAIN EXECUTION WITH MODEL REUSE LOGIC
# ────────────────────────────────────────────────
def main():
    results = []
    dl_models = ["DNN", "CNN", "LSTM"]

    for ds in DATASET_FILES:
        print("\n" + "═" * 80)
        print(f" DATASET MULTI-CLASS: {ds['name']}")
        print("═" * 80)

        try:
            X_train, X_test, y_train, y_test = load_dataset(ds)
        except Exception as e:
            print(f"❌ Không thể tải dữ liệu tập {ds['name']}: {e}")
            continue

        num_classes = len(np.unique(y_train))
        models = get_models(num_classes)
        class_names = CLASS_NAMES_MAP.get(
            ds["name"], [f"Class {i}" for i in range(num_classes)]
        )

        for name, model_spec in models.items():
            print(f"\n🔥 EVALUATING MODEL: {name}")
            start = time.time()

            keras_file = MODEL_DIR / f"BEST_{name}_{ds['name']}.keras"
            pkl_file = MODEL_DIR / f"{ds['name']}_{name}.pkl"

            try:
                # 1. NẾU LÀ MÔ HÌNH HỌC SÂU (DNN, CNN, LSTM) VÀ ĐÃ CÓ FILE CHECKPOINT -> LOAD LẠI
                if name in dl_models and keras_file.exists():
                    print(
                        f" [Deep Learning] Phát hiện mô hình đã lưu: {keras_file.name} -> Đang nạp..."
                    )
                    model_obj = load_model(keras_file)

                    if name == "CNN":
                        X_test_input = X_test.reshape(
                            X_test.shape[0], X_test.shape[1], 1
                        )
                    elif name == "LSTM":
                        X_test_input = X_test.reshape(
                            X_test.shape[0], 1, X_test.shape[1]
                        )
                    else:
                        X_test_input = X_test

                    y_score = model_obj.predict(X_test_input, verbose=0)
                    y_pred = np.argmax(y_score, axis=1)

                # 2. NẾU LÀ MÔ HÌNH HỌC SÂU NHƯNG CHƯA CÓ FILE -> CHẠY TRAIN MỚI
                elif name in dl_models:
                    print(
                        f" [Deep Learning] Chưa có file lưu sẵn, tiến hành huấn luyện mới mô hình {name}..."
                    )
                    if name == "DNN":
                        model_obj, y_score, history = train_dnn(
                            X_train, y_train, X_test, num_classes
                        )
                    elif name == "CNN":
                        model_obj, y_score, history = train_cnn(
                            X_train, y_train, X_test, num_classes
                        )
                    elif name == "LSTM":
                        model_obj, y_score, history = train_lstm(
                            X_train, y_train, X_test, num_classes
                        )

                    model_obj.save(keras_file)
                    plot_loss(history, ds["name"], name)
                    y_pred = np.argmax(y_score, axis=1)

                # 3. MÔ HÌNH MACHINE LEARNING TRUYỀN THỐNG -> LUÔN LUÔN TRAIN MỚI
                else:
                    print(
                        f"⚡ [Machine Learning] Tiến hành huấn luyện mới mô hình {name}..."
                    )
                    model_spec.fit(X_train, y_train)
                    y_pred = model_spec.predict(X_test)
                    y_score = (
                        model_spec.predict_proba(X_test)
                        if hasattr(model_spec, "predict_proba")
                        else None
                    )
                    with open(pkl_file, "wb") as f:
                        pickle.dump(model_spec, f)

                train_time = time.time() - start
                print(f"✅ Hoàn tất {name} trong {train_time:.1f}s")

                # TÍNH TOÁN VÀ VẼ ĐỒ THỊ
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
                rec = recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
                f1_macro = f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
                ids_metrics = compute_ids_metrics(y_test, y_pred, num_classes)

                cm = confusion_matrix(
                    y_test, y_pred, labels=range(num_classes)
                )
                plot_confusion_matrix(
                    cm, ds["name"], name, class_names=class_names
                )

                if y_score is not None:
                    plot_roc(y_test, y_score, ds["name"], name, num_classes)

                results.append({
                    "Dataset": ds["name"],
                    "Model": name,
                    "Accuracy": acc,
                    "Precision_macro": prec,
                    "Recall_macro": rec,
                    "F1_macro": f1_macro,
                    "FPR_macro": ids_metrics["FPR_macro"],
                    "FAP_macro": ids_metrics["FAP_macro"],
                    "TrainTime_seconds": train_time,
                })

            except Exception as e:
                print(f"❌ LỖI KHI XỬ LÝ MODEL {name}: {e}")
                continue

    if results:
        df = pd.DataFrame(results)
        output_dir = Path(r"D:\NLATTT\results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "model_results.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\n✔ ĐÃ HOÀN THÀNH VÀ LƯU BÁO CÁO VÀO {output_file}")

if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
