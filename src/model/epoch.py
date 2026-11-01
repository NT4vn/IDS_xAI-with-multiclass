import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ────────────────────────────────────────────────
# PATH CONFIGURATION
# ────────────────────────────────────────────────
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")

MODEL_DIR = Path("saved_models")
PLOT_DIR = Path("plots")
RESULT_DIR = Path("results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR / "loss", exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# DATASET FILES CONFIGURATION
# ────────────────────────────────────────────────
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
# LOAD DATASET (MULTI-CLASS HANDLER)
# ────────────────────────────────────────────────
def load_dataset(ds):
    print(f"\n→ Loading dataset: {ds['name']}")

    X_train = np.load(DATA_DIR / ds["X_train"])
    X_test = np.load(DATA_DIR / ds["X_test"])
    y_train = np.load(DATA_DIR / ds["Y_train"])
    y_test = np.load(DATA_DIR / ds["Y_test"])

    # Đảm bảo nhãn 1D categorical cho multi-class
    if y_train.ndim > 1:
        if y_train.shape[1] > 1:
            y_train = np.argmax(y_train, axis=1)
        else:
            y_train = y_train.ravel()

    if y_test.ndim > 1:
        if y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.ravel()

    print(
        f"  X_train: {X_train.shape} | X_test: {X_test.shape} | Classes count: {len(np.unique(y_train))}"
    )

    return X_train, X_test, y_train, y_test

# ────────────────────────────────────────────────
# SAVE HISTORY
# ────────────────────────────────────────────────
def save_history_csv(history, dataset, model):
    hist_df = pd.DataFrame(history.history)
    save_path = PLOT_DIR / "loss" / f"{dataset}_{model}_history.csv"
    hist_df.to_csv(save_path, index_label="epoch")
    print(f"📁 Saved history: {save_path}")

# ────────────────────────────────────────────────
# PLOT METRICS (LOSS / ACC / MSE)
# ────────────────────────────────────────────────
def plot_metrics(history, dataset, model):
    hist = history.history

    # LOSS
    plt.figure(figsize=(7, 5))
    plt.plot(
        pd.Series(hist["loss"]).rolling(5, min_periods=1).mean(),
        label="Train Loss",
    )
    plt.plot(
        pd.Series(hist["val_loss"]).rolling(5, min_periods=1).mean(),
        label="Validation Loss",
    )
    plt.title(f"{dataset} - {model} Multi-Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss" / f"{dataset}_{model}_loss.png", dpi=300)
    plt.close()

    # ACCURACY
    if "accuracy" in hist:
        plt.figure(figsize=(7, 5))
        plt.plot(
            pd.Series(hist["accuracy"]).rolling(5, min_periods=1).mean(),
            label="Train Accuracy",
        )
        plt.plot(
            pd.Series(hist["val_accuracy"]).rolling(5, min_periods=1).mean(),
            label="Validation Accuracy",
        )
        plt.title(f"{dataset} - {model} Multi-Class Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / "loss" / f"{dataset}_{model}_accuracy.png", dpi=300
        )
        plt.close()

    # MSE
    if "mse" in hist:
        plt.figure(figsize=(7, 5))
        plt.plot(
            pd.Series(hist["mse"]).rolling(5, min_periods=1).mean(),
            label="Train MSE",
        )
        plt.plot(
            pd.Series(hist["val_mse"]).rolling(5, min_periods=1).mean(),
            label="Validation MSE",
        )
        plt.title(f"{dataset} - {model} Multi-Class MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "loss" / f"{dataset}_{model}_mse.png", dpi=300)
        plt.close()

# ────────────────────────────────────────────────
# COMPARE ALL MODELS
# ────────────────────────────────────────────────
def compare_histories(dataset_name):
    history_files = list(
        (PLOT_DIR / "loss").glob(f"{dataset_name}_*_history.csv")
    )

    if not history_files:
        return

    # LOSS COMPARE
    plt.figure(figsize=(8, 6))
    for file in history_files:
        df = pd.read_csv(file)
        model_name = (
            file.stem.replace(f"{dataset_name}_", "").replace("_history", "")
        )
        plt.plot(df["loss"], label=model_name)

    plt.title(f"{dataset_name} - Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "loss" / f"{dataset_name}_ALL_LOSS_COMPARE.png", dpi=300
    )
    plt.close()

    # ACCURACY COMPARE
    plt.figure(figsize=(8, 6))
    for file in history_files:
        df = pd.read_csv(file)
        if "accuracy" in df.columns:
            model_name = (
                file.stem.replace(f"{dataset_name}_", "").replace(
                    "_history", ""
                )
            )
            plt.plot(df["accuracy"], label=model_name)

    plt.title(f"{dataset_name} - Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "loss" / f"{dataset_name}_ALL_ACC_COMPARE.png", dpi=300
    )
    plt.close()

# ────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────
def get_callbacks(model_name, dataset_name):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(
                MODEL_DIR / f"BEST_{dataset_name}_{model_name}.keras"
            ),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

# ────────────────────────────────────────────────
# DNN MODEL (MULTI-CLASS)
# ────────────────────────────────────────────────
def train_dnn(X_train, y_train, X_test, dataset_name):
    num_classes = len(np.unique(y_train))
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
        metrics=["accuracy", "mse"],
    )

    history = model.fit(
        X_train,
        y_train_cat,
        epochs=300,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=get_callbacks("DNN", dataset_name),
    )

    pred = model.predict(X_test, verbose=0)
    return model, np.argmax(pred, axis=1), history

# ────────────────────────────────────────────────
# CNN MODEL (MULTI-CLASS)
# ────────────────────────────────────────────────
def train_cnn(X_train, y_train, X_test, dataset_name):
    num_classes = len(np.unique(y_train))

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
        metrics=["accuracy", "mse"],
    )

    history = model.fit(
        X_train_r,
        y_train_cat,
        epochs=350,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=get_callbacks("CNN", dataset_name),
    )

    pred = model.predict(X_test_r, verbose=0)
    return model, np.argmax(pred, axis=1), history

# ────────────────────────────────────────────────
# LSTM MODEL (MULTI-CLASS)
# ────────────────────────────────────────────────
def train_lstm(X_train, y_train, X_test, dataset_name):
    num_classes = len(np.unique(y_train))

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
        metrics=["accuracy", "mse"],
    )

    history = model.fit(
        X_train_r,
        y_train_cat,
        epochs=400,
        batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=get_callbacks("LSTM", dataset_name),
    )

    pred = model.predict(X_test_r, verbose=0)
    return model, np.argmax(pred, axis=1), history

# ────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────
def main():
    results = []

    for ds in DATASET_FILES:
        print("\n" + "=" * 80)
        print(f"DATASET MULTI-CLASS: {ds['name']}")
        print("=" * 80)

        try:
            X_train, X_test, y_train, y_test = load_dataset(ds)
        except Exception as e:
            print(f"❌ Error loading dataset {ds['name']}: {e}")
            continue

        models = {"DNN": train_dnn, "CNN": train_cnn, "LSTM": train_lstm}

        for name, train_func in models.items():
            print(f"\n🔥 TRAINING {name}")

            start = time.time()

            try:
                model, y_pred, history = train_func(
                    X_train, y_train, X_test, ds["name"]
                )

                train_time = time.time() - start
                acc = accuracy_score(y_test, y_pred)

                print(f"✅ Accuracy: {acc:.4f}")
                print(f"⏱ Time: {train_time:.2f}s")

                # SAVE MODEL (.keras format)
                model.save(MODEL_DIR / f"{ds['name']}_{name}.keras")

                # SAVE HISTORY CSV
                save_history_csv(history, ds["name"], name)

                # PLOT LOSS/ACC/MSE
                plot_metrics(history, ds["name"], name)

                results.append({
                    "Dataset": ds["name"],
                    "Model": name,
                    "Accuracy": acc,
                    "TrainTime": train_time,
                })

            except Exception as e:
                print(f"❌ Error training {name} on {ds['name']}: {e}")
                continue

        # COMPARE HISTORIES AFTER ALL MODELS TRAINED
        compare_histories(ds["name"])

    # SAVE RESULTS TO EXCEL
    if results:
        df = pd.DataFrame(results)
        df.to_excel(RESULT_DIR / "deep_learning_results.xlsx", index=False)
        print("\n✔ FINISHED ALL DEEP LEARNING TRAINING")

# ────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 START MULTI-CLASS DEEP LEARNING TRAINING...")
    tf.get_logger().setLevel("ERROR")
    main()
