import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ============================================================
# 1. CONFIGURATION
# ============================================================

PREPROCESSED_DIR = Path(
    r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped"
)

MODEL_DIR = Path("saved_models")

SAVE_DIR = Path("XAI_Results3") / "global" / "SIGNAL2022"

DATASET_NAME = "SIGNAL2022"
DATASET_PREFIX = "signal22"

N_CLASSES_EXPECTED = 4

# Cố định số mẫu để các model và SHAP method dùng cùng dữ liệu.
N_GLOBAL = 160
N_BACKGROUND = 30

# LIME sử dụng các mẫu thuộc tập explain cố định.
N_LIME_PER_CLASS = 50
LIME_NUM_SAMPLES = 1000

# KernelSHAP có thể tốn thời gian đáng kể.
KERNEL_NSAMPLES = 200

RANDOM_SEED = 42

CLASS_NAMES = [
    "DoS",
    "Malware",
    "Normal",
    "Port Scan",
]

MODELS = [
    "RandomForest",
    "LightGBM",
    "AdaBoost",
    "MLP",
    "DNN",
    "CNN",
    "LSTM",
]

DEEP_MODELS = {"DNN", "CNN", "LSTM"}

TREE_MODELS = {"RandomForest", "LightGBM"}

X_TRAIN_CANDIDATES = [
    "preprocessed_multi_signal22_X_train.npy",
    "preprocessed_signal22_X_train.npy",
]

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def find_existing_file(directory, filenames):
    for filename in filenames:
        path = directory / filename
        if path.is_file():
            return path
    return None

def load_pickle(path):
    return joblib.load(path)

def safe_filename(value):
    result = str(value)

    for char in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        result = result.replace(char, "_")

    return result


def save_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def close_plot():
    plt.close("all")


def dataset_fingerprint(X, y):
    h = hashlib.sha256()

    h.update(str(X.shape).encode("utf-8"))
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(X).view(np.uint8))

    h.update(str(y.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(y).view(np.uint8))

    return h.hexdigest()


# ============================================================
# 3. LOAD FEATURE NAMES
# ============================================================

def load_feature_names(n_features):
    candidates = [
        "preprocessed_multi_signal22_feature_names.pkl",
        "preprocessed_multi_signal22_feature_names.npy",
        "preprocessed_multi_signal22_selected_features.pkl",
        "signal22_selected_features.pkl",
    ]

    path = find_existing_file(PREPROCESSED_DIR, candidates)

    if path is None:
        raise FileNotFoundError(
            "Không tìm thấy feature metadata cho SIGNAL2022.\n"
            f"Thư mục: {PREPROCESSED_DIR}\n"
            f"Đã kiểm tra: {candidates}\n"
            "Không tự tạo Feature_0, Feature_1 hoặc cắt feature."
        )

    try:
        if path.suffix.lower() == ".npy":
            names = np.load(path, allow_pickle=True)
        else:
            names = load_pickle(path)

    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được metadata {path.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(names, dict):
        found = False

        for key in [
            "feature_names",
            "selected_features",
            "features",
            "columns",
        ]:
            if key in names:
                names = names[key]
                found = True
                break

        if not found:
            raise ValueError(
                f"{path.name} là dictionary nhưng không tìm thấy "
                "key chứa danh sách feature."
            )

    names = list(names)

    if len(names) != n_features:
        raise ValueError(
            f"Feature metadata không khớp dữ liệu.\n"
            f"File: {path}\n"
            f"Metadata: {len(names)} feature\n"
            f"X_test: {n_features} feature\n"
            "Kiểm tra đúng artifact được tạo từ pipeline preprocessing."
        )

    names = [str(name) for name in names]

    if len(set(names)) != len(names):
        print("[WARN] Metadata có tên feature trùng nhau.")

    print(f"[INFO] Feature metadata: {path}")

    return names


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset():
    x_path = (
        PREPROCESSED_DIR
        / "preprocessed_multi_signal22_X_test.npy"
    )

    y_path = (
        PREPROCESSED_DIR
        / "preprocessed_multi_signal22_y_test.npy"
    )

    if not x_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {x_path}")

    if not y_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {y_path}")

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test phải có dạng (N, F), nhận {X_test.shape}."
        )

    if y_test.ndim > 1:
        if y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.reshape(-1)

    y_test = np.asarray(y_test).astype(int)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Số dòng X_test và y_test không bằng nhau."
        )

    if len(y_test) == 0:
        raise ValueError("Test set rỗng.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test có NaN hoặc Inf.")

    if np.any(y_test < 0) or np.any(y_test >= N_CLASSES_EXPECTED):
        raise ValueError(
            "Nhãn y_test phải nằm trong khoảng 0..3. "
            "Hãy kiểm tra thứ tự mã hóa lớp của SIGNAL2022."
        )

    feature_names = load_feature_names(X_test.shape[1])

    test_support = np.bincount(
        y_test,
        minlength=N_CLASSES_EXPECTED,
    )

    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(
            f"  Class {class_idx}: {class_name}; "
            f"test support = {test_support[class_idx]}"
        )

    return X_test.astype(np.float32), y_test, feature_names, test_support


def load_X_train_if_available(n_features):
    path = find_existing_file(
        PREPROCESSED_DIR,
        X_TRAIN_CANDIDATES,
    )

    if path is None:
        print(
            "[WARN] Không tìm thấy X_train. "
            "LIME sẽ dùng X_test làm dữ liệu tham chiếu."
        )
        return None

    X_train = np.load(path)

    if X_train.ndim != 2:
        raise ValueError(
            f"{path.name} phải có dạng (N, F), "
            f"nhận {X_train.shape}."
        )

    if X_train.shape[1] != n_features:
        raise ValueError(
            f"{path.name} có {X_train.shape[1]} feature, "
            f"trong khi X_test có {n_features}."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(f"{path.name} có NaN hoặc Inf.")

    print(f"[INFO] LIME training reference: {path}")

    return X_train.astype(np.float32)


# ============================================================
# 5. LOAD MODELS
# ============================================================

def load_model(model_name):
    if model_name in DEEP_MODELS:
        candidates = [
            f"{DATASET_NAME}_{model_name}.keras",
            f"{DATASET_NAME}_{model_name}.h5",
        ]

        path = find_existing_file(MODEL_DIR, candidates)

        if path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model Keras {model_name}.\n"
                f"Thư mục: {MODEL_DIR}\n"
                f"Đã kiểm tra: {candidates}"
            )

        model = tf.keras.models.load_model(
            path,
            compile=False,
        )

        model_type = "keras"

    else:
        candidates = [
            f"{DATASET_NAME}_{model_name}.pkl",
        ]

        path = find_existing_file(MODEL_DIR, candidates)

        if path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model sklearn {model_name}.\n"
                f"Thư mục: {MODEL_DIR}\n"
                f"Đã kiểm tra: {candidates}"
            )

        try:
            model = load_pickle(path)
        except Exception as exc:
            raise RuntimeError(
                f"Không đọc được model {path.name}: {exc}"
            ) from exc

        model_type = "sklearn"

        if hasattr(model, "n_jobs"):
            try:
                model.n_jobs = 1
            except Exception:
                pass

    print(f"[INFO] Loaded {model_name}: {path}")

    return model, model_type


# ============================================================
# 6. MODEL INPUT AND PROBABILITY
# ============================================================

def prepare_model_input(X, model_type, model_name):
    if model_type != "keras":
        return X

    if model_name == "CNN":
        return X.reshape(X.shape[0], X.shape[1], 1)

    if model_name == "LSTM":
        return X.reshape(X.shape[0], 1, X.shape[1])

    return X


def reorder_sklearn_probabilities(model, probabilities):
    if not hasattr(model, "classes_"):
        raise ValueError(
            f"{type(model).__name__} không có classes_; "
            "không thể xác minh thứ tự xác suất."
        )

    model_classes = np.asarray(model.classes_)

    try:
        class_ids = model_classes.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"classes_ không phải mã lớp nguyên: "
            f"{model_classes.tolist()}"
        ) from exc

    if sorted(class_ids.tolist()) != list(range(N_CLASSES_EXPECTED)):
        raise ValueError(
            f"Model classes_={model_classes.tolist()} không khớp "
            "các mã lớp 0..3. Dừng để tránh giải thích sai lớp."
        )

    reordered = np.zeros(
        (probabilities.shape[0], N_CLASSES_EXPECTED),
        dtype=float,
    )

    for source_col, class_id in enumerate(class_ids):
        reordered[:, class_id] = probabilities[:, source_col]

    return reordered


def predict_probabilities(model, X, model_type, model_name):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"Input cần dạng (N, F), nhận {X.shape}."
        )

    X_model = prepare_model_input(
        X,
        model_type,
        model_name,
    )

    if model_type == "keras":
        probabilities = model.predict(
            X_model,
            verbose=0,
        )

    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_model)

    else:
        raise TypeError(
            f"{model_name} không hỗ trợ predict_proba."
        )

    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.shape != (
        len(X),
        N_CLASSES_EXPECTED,
    ):
        raise ValueError(
            f"{model_name} output shape={probabilities.shape}; "
            f"mong đợi {(len(X), N_CLASSES_EXPECTED)}."
        )

    if model_type == "sklearn":
        probabilities = reorder_sklearn_probabilities(
            model,
            probabilities,
        )

    # Keras không lưu sẵn classes_ trong output.
    # Phải đảm bảo thứ tự output 0..3 khớp pipeline train:
    # DoS, Malware, Normal, Port Scan.

    if not np.isfinite(probabilities).all():
        raise ValueError("Model output chứa NaN hoặc Inf.")

    if np.any(probabilities < -1e-6):
        raise ValueError("Model output có xác suất âm.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Tổng xác suất không gần 1. "
            "Kiểm tra model có trả xác suất softmax hay không."
        )

    return probabilities


# ============================================================
# 7. FIXED EXPLANATION AND BACKGROUND INDICES
# ============================================================

def stratified_indices(y, n_total, seed):
    y = np.asarray(y)
    n_total = min(int(n_total), len(y))

    if n_total <= 0:
        raise ValueError("Số mẫu cần lấy phải lớn hơn 0.")

    classes = np.unique(y)

    if n_total < len(classes):
        raise ValueError(
            "Số mẫu cần lấy nhỏ hơn số lớp xuất hiện."
        )

    rng = np.random.default_rng(seed)

    groups = {
        int(c): rng.permutation(np.flatnonzero(y == c))
        for c in classes
    }

    allocation = {int(c): 0 for c in classes}
    remaining = n_total

    # Phân bổ mẫu đều giữa các lớp, không vượt quá support.
    while remaining > 0:
        active = [
            int(c)
            for c in classes
            if allocation[int(c)] < len(groups[int(c)])
        ]

        if not active:
            break

        for c in rng.permutation(active):
            allocation[c] += 1
            remaining -= 1

            if remaining == 0:
                break

    selected = [
        groups[c][:allocation[c]]
        for c in allocation
        if allocation[c] > 0
    ]

    return np.sort(np.concatenate(selected).astype(int))


def make_or_load_fixed_indices(X_test, y_test):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fingerprint_data = dataset_fingerprint(X_test, y_test)

    config = (
        f"signal22-v1|global={N_GLOBAL}|"
        f"background={N_BACKGROUND}|seed={RANDOM_SEED}"
    )

    fingerprint = hashlib.sha256(
        f"{fingerprint_data}|{config}".encode("utf-8")
    ).hexdigest()

    tag = hashlib.sha256(
        config.encode("utf-8")
    ).hexdigest()[:12]

    explain_path = SAVE_DIR / f"explain_indices_{tag}.npy"
    background_path = SAVE_DIR / f"background_indices_{tag}.npy"
    fingerprint_path = SAVE_DIR / f"sampling_fingerprint_{tag}.txt"

    same_dataset = (
        fingerprint_path.is_file()
        and fingerprint_path.read_text(
            encoding="utf-8"
        ).strip() == fingerprint
    )

    def valid_indices(indices, expected_size):
        return (
            indices.ndim == 1
            and len(indices) == expected_size
            and len(np.unique(indices)) == len(indices)
            and np.all(indices >= 0)
            and np.all(indices < len(y_test))
        )

    explain_idx = None
    background_idx = None

    if same_dataset and explain_path.is_file():
        candidate = np.load(explain_path)

        if valid_indices(
            candidate,
            min(N_GLOBAL, len(y_test)),
        ):
            explain_idx = candidate.astype(int)

    if same_dataset and background_path.is_file():
        candidate = np.load(background_path)

        if valid_indices(
            candidate,
            min(N_BACKGROUND, len(y_test)),
        ):
            background_idx = candidate.astype(int)

    if explain_idx is None:
        explain_idx = stratified_indices(
            y_test,
            min(N_GLOBAL, len(y_test)),
            RANDOM_SEED,
        )

        np.save(explain_path, explain_idx)

        print(f"[INFO] Saved explain indices: {explain_path}")

    else:
        print(f"[INFO] Reusing explain indices: {explain_path}")

    if background_idx is None:
        background_idx = stratified_indices(
            y_test,
            min(N_BACKGROUND, len(y_test)),
            RANDOM_SEED + 1,
        )

        np.save(background_path, background_idx)

        print(f"[INFO] Saved background indices: {background_path}")

    else:
        print(f"[INFO] Reusing background indices: {background_path}")

    fingerprint_path.write_text(
        fingerprint,
        encoding="utf-8",
    )

    # LIME sử dụng các mẫu từ cùng explain_idx.
    lime_indices = {}

    for class_idx in range(N_CLASSES_EXPECTED):
        candidates = explain_idx[
            y_test[explain_idx] == class_idx
        ]

        lime_indices[class_idx] = candidates[
            :N_LIME_PER_CLASS
        ]

    manifest_rows = []

    for idx in explain_idx:
        manifest_rows.append([
            "explain",
            int(idx),
            int(y_test[idx]),
        ])

    for idx in background_idx:
        manifest_rows.append([
            "background",
            int(idx),
            int(y_test[idx]),
        ])

    for class_idx, indices in lime_indices.items():
        for idx in indices:
            manifest_rows.append([
                "lime",
                int(idx),
                int(class_idx),
            ])

    save_csv(
        SAVE_DIR / "SIGNAL2022_sample_manifest.csv",
        [
            "sample_role",
            "test_row_index",
            "true_class_index",
        ],
        manifest_rows,
    )

    return explain_idx, background_idx, lime_indices


# ============================================================
# 8. NORMALIZE SHAP OUTPUT
# ============================================================

def normalize_one_class(values, n_samples, n_features):
    arr = np.asarray(values)

    while arr.ndim > 2:
        singleton_axes = [
            axis
            for axis in range(1, arr.ndim)
            if arr.shape[axis] == 1
        ]

        if not singleton_axes:
            break

        arr = np.squeeze(
            arr,
            axis=singleton_axes[-1],
        )

    if arr.ndim != 2:
        raise ValueError(
            f"Không chuẩn hóa được SHAP class array: {arr.shape}"
        )

    if arr.shape == (n_samples, n_features):
        return arr

    if arr.shape == (n_features, n_samples):
        return arr.T

    raise ValueError(
        f"SHAP class shape={arr.shape}; "
        f"mong đợi {(n_samples, n_features)}."
    )


def normalize_shap_values(
    raw_values,
    n_samples,
    n_features,
    n_classes,
):
    if isinstance(raw_values, list):
        if len(raw_values) != n_classes:
            raise ValueError(
                f"SHAP trả {len(raw_values)} phần tử; "
                f"mong đợi {n_classes}."
            )

        arrays = [
            normalize_one_class(
                value,
                n_samples,
                n_features,
            )
            for value in raw_values
        ]

        result = np.stack(arrays, axis=-1)

    else:
        arr = np.asarray(raw_values)

        # Bỏ trục singleton của CNN/LSTM, ví dụ (N,F,1,C).
        while arr.ndim > 3:
            singleton_axes = [
                axis
                for axis in range(1, arr.ndim)
                if arr.shape[axis] == 1
            ]

            if not singleton_axes:
                raise ValueError(
                    f"SHAP output không hỗ trợ: {arr.shape}"
                )

            arr = np.squeeze(
                arr,
                axis=singleton_axes[-1],
            )

        if arr.ndim == 2:
            if arr.shape != (n_samples, n_features):
                raise ValueError(
                    f"SHAP 2D shape không khớp: {arr.shape}"
                )

            if n_classes != 1:
                raise ValueError(
                    "SHAP chỉ trả một đầu ra trong khi model "
                    f"có {n_classes} lớp."
                )

            result = arr[:, :, None]

        elif arr.ndim == 3:
            if arr.shape == (
                n_samples,
                n_features,
                n_classes,
            ):
                result = arr

            elif arr.shape == (
                n_samples,
                n_classes,
                n_features,
            ):
                result = np.transpose(arr, (0, 2, 1))

            elif arr.shape == (
                n_classes,
                n_samples,
                n_features,
            ):
                result = np.transpose(arr, (1, 2, 0))

            else:
                raise ValueError(
                    f"Không xác định được trục SHAP: {arr.shape}"
                )

        else:
            raise ValueError(
                f"SHAP output không hỗ trợ: {arr.shape}"
            )

    expected = (n_samples, n_features, n_classes)

    if result.shape != expected:
        raise ValueError(
            f"SHAP normalized shape={result.shape}; "
            f"mong đợi {expected}."
        )

    if not np.isfinite(result).all():
        raise ValueError("SHAP chứa NaN hoặc Inf.")

    return result.astype(float)


def reorder_tree_shap_classes(raw_values, model):
    if not hasattr(model, "classes_"):
        raise ValueError(
            "TreeSHAP không xác định được model.classes_."
        )

    class_ids = np.asarray(model.classes_).astype(int)

    if sorted(class_ids.tolist()) != list(range(N_CLASSES_EXPECTED)):
        raise ValueError(
            f"TreeSHAP classes_ không khớp 0..3: "
            f"{class_ids.tolist()}"
        )

    if isinstance(raw_values, list):
        if len(raw_values) != N_CLASSES_EXPECTED:
            raise ValueError(
                f"TreeSHAP trả {len(raw_values)} lớp."
            )

        reordered = [None] * N_CLASSES_EXPECTED

        for source_pos, class_id in enumerate(class_ids):
            reordered[class_id] = raw_values[source_pos]

        return reordered

    arr = np.asarray(raw_values)

    if arr.ndim == 3:
        # Các phiên bản SHAP thường đặt lớp ở trục cuối.
        if arr.shape[-1] == N_CLASSES_EXPECTED:
            return arr[:, :, np.argsort(class_ids)]

        # Một số định dạng đặt lớp ở trục giữa.
        if arr.shape[1] == N_CLASSES_EXPECTED:
            return arr[:, np.argsort(class_ids), :]

    return raw_values


# ============================================================
# 9. IMPORTANCE REPORTS
# ============================================================

def save_importance_reports(
    method,
    feature_names,
    class_importance,
    class_explain_counts,
    test_support,
    output_dir,
):
    per_class_rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for feature_idx, feature_name in enumerate(feature_names):
            value = class_importance[class_idx, feature_idx]

            per_class_rows.append([
                class_idx,
                class_name,
                int(test_support[class_idx]),
                int(class_explain_counts[class_idx]),
                feature_name,
                float(value) if np.isfinite(value) else "NaN",
            ])

    save_csv(
        output_dir / f"{method}_per_class_importance.csv",
        [
            "class_index",
            "class_name",
            "test_support",
            "n_explained_samples",
            "feature",
            "mean_abs_importance",
        ],
        per_class_rows,
    )

    supported_classes = np.flatnonzero(test_support > 0)

    valid_classes = np.asarray([
        c
        for c in range(N_CLASSES_EXPECTED)
        if (
            class_explain_counts[c] > 0
            and np.isfinite(class_importance[c]).all()
        )
    ], dtype=int)

    # Trung bình đều giữa các lớp có giải thích hợp lệ.
    if len(valid_classes) > 0:
        balanced = np.mean(
            class_importance[valid_classes],
            axis=0,
        )
    else:
        balanced = np.full(len(feature_names), np.nan)

    # Trọng số tự nhiên theo support thật của toàn bộ test set.
    all_supported_valid = all(
        c in valid_classes
        for c in supported_classes
    )

    if len(supported_classes) > 0 and all_supported_valid:
        weights = (
            test_support[supported_classes].astype(float)
            / test_support[supported_classes].sum()
        )

        natural = np.sum(
            class_importance[supported_classes]
            * weights[:, None],
            axis=0,
        )

    else:
        natural = np.full(len(feature_names), np.nan)

        print(
            f"[WARN] {method}: không tính được natural importance "
            "vì có lớp hiện diện trong test set nhưng thiếu "
            "importance hợp lệ."
        )

    aggregate_rows = []

    for idx, feature_name in enumerate(feature_names):
        aggregate_rows.append([
            feature_name,
            float(natural[idx]) if np.isfinite(natural[idx]) else "NaN",
            float(balanced[idx]) if np.isfinite(balanced[idx]) else "NaN",
        ])

    save_csv(
        output_dir / f"{method}_global_importance.csv",
        [
            "feature",
            "importance_natural_test_distribution",
            "importance_class_balanced",
        ],
        aggregate_rows,
    )


def save_importance_plot(
    importances,
    feature_names,
    title,
    output_path,
    top_n=20,
):
    importances = np.asarray(importances).reshape(-1)

    if not np.isfinite(importances).any():
        print(f"[WARN] Không có importance hợp lệ để vẽ: {title}")
        return

    clean_values = np.nan_to_num(
        importances,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    order = np.argsort(clean_values)[::-1][:top_n]
    order = order[::-1]

    try:
        plt.figure(figsize=(10, 7))

        plt.barh(
            [feature_names[i] for i in order],
            clean_values[order],
        )

        plt.xlabel("Mean Absolute Importance")
        plt.title(title)
        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )

    finally:
        close_plot()


def save_shap_summary_plot(
    shap_values,
    X_explain,
    feature_names,
    title,
    output_path,
):
    if len(X_explain) == 0:
        return

    try:
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            max_display=min(20, len(feature_names)),
            show=False,
        )

        plt.title(title)
        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )

    finally:
        close_plot()


# ============================================================
# 10. CALCULATE SHAP
# ============================================================

def calculate_shap(
    method,
    model,
    model_type,
    model_name,
    X_background,
    X_explain,
):
    n_samples = len(X_explain)
    n_features = X_explain.shape[1]

    if method == "TreeSHAP":
        explainer = shap.TreeExplainer(model)

        raw_values = explainer.shap_values(X_explain)

        raw_values = reorder_tree_shap_classes(
            raw_values,
            model,
        )

    elif method == "KernelSHAP":

        def predict_fn(data):
            return predict_probabilities(
                model,
                data,
                model_type,
                model_name,
            )

        explainer = shap.KernelExplainer(
            predict_fn,
            X_background,
        )

        raw_values = explainer.shap_values(
            X_explain,
            nsamples=KERNEL_NSAMPLES,
        )

    elif method == "GradientSHAP":
        background_input = prepare_model_input(
            X_background,
            model_type,
            model_name,
        )

        explain_input = prepare_model_input(
            X_explain,
            model_type,
            model_name,
        )

        explainer = shap.GradientExplainer(
            model,
            background_input,
        )

        raw_values = explainer.shap_values(
            explain_input,
        )

    else:
        raise ValueError(f"SHAP method không hợp lệ: {method}")

    return normalize_shap_values(
        raw_values,
        n_samples,
        n_features,
        N_CLASSES_EXPECTED,
    )


def run_shap_method(
    method,
    model,
    model_type,
    model_name,
    X_test,
    y_test,
    explain_idx,
    background_idx,
    feature_names,
    test_support,
    output_dir,
):
    print(f"[INFO] Running {method}...")

    X_explain = X_test[explain_idx]
    y_explain = y_test[explain_idx]
    X_background = X_test[background_idx]

    # Kiểm tra xác suất và thứ tự output trước khi giải thích.
    predict_probabilities(
        model,
        X_explain[:min(5, len(X_explain))],
        model_type,
        model_name,
    )

    shap_values = calculate_shap(
        method=method,
        model=model,
        model_type=model_type,
        model_name=model_name,
        X_background=X_background,
        X_explain=X_explain,
    )

    class_importance = np.full(
        (N_CLASSES_EXPECTED, len(feature_names)),
        np.nan,
    )

    class_counts = np.zeros(N_CLASSES_EXPECTED, dtype=int)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Theo cùng logic CIC17: giải thích riêng mẫu có nhãn thật
        # thuộc lớp đang phân tích.
        mask = y_explain == class_idx
        class_counts[class_idx] = int(mask.sum())

        if not np.any(mask):
            print(
                f"[WARN] {method}: không có mẫu thật thuộc "
                f"lớp {class_name} trong tập explain."
            )
            continue

        class_sv = shap_values[mask, :, class_idx]
        class_X = X_explain[mask]

        class_importance[class_idx] = np.mean(
            np.abs(class_sv),
            axis=0,
        )

        safe_class = safe_filename(class_name)

        save_shap_summary_plot(
            class_sv,
            class_X,
            feature_names,
            f"{method} - {model_name} - {class_name}",
            output_dir / f"{method}_{safe_class}_summary.png",
        )

        save_importance_plot(
            class_importance[class_idx],
            feature_names,
            f"{method} - {model_name} - {class_name}",
            output_dir / f"{method}_{safe_class}_bar.png",
        )

    # Importance global tổng hợp từ tất cả mẫu và cả 4 đầu ra.
    global_importance = np.mean(
        np.abs(shap_values),
        axis=(0, 2),
    )

    save_importance_plot(
        global_importance,
        feature_names,
        f"{method} - {model_name} - Global",
        output_dir / f"{method}_global_bar.png",
    )

    save_importance_reports(
        method=method,
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print(f"[OK] Completed {method}")


# ============================================================
# 11. LIME
# ============================================================

def run_lime(
    model,
    model_type,
    model_name,
    X_test,
    y_test,
    X_train,
    lime_indices,
    feature_names,
    test_support,
    output_dir,
):
    from lime.lime_tabular import LimeTabularExplainer

    print("[INFO] Running LIME...")

    # Ưu tiên X_train đã qua đúng preprocessing.
    # Nếu thiếu thì dùng X_test, đồng thời cảnh báo.
    X_reference = X_train if X_train is not None else X_test

    if X_train is None:
        print(
            "[WARN] LIME dùng X_test làm background vì không có X_train."
        )

    explainer = LimeTabularExplainer(
        training_data=X_reference,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )

    def predict_fn(data):
        return predict_probabilities(
            model,
            np.asarray(data, dtype=np.float32),
            model_type,
            model_name,
        )

    class_importance = np.full(
        (N_CLASSES_EXPECTED, len(feature_names)),
        np.nan,
    )

    class_counts = np.zeros(N_CLASSES_EXPECTED, dtype=int)
    local_rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Lấy các mẫu cố định từ tập explain, phân nhóm theo nhãn thật.
        sample_indices = lime_indices.get(
            class_idx,
            np.asarray([], dtype=int),
        )

        if len(sample_indices) == 0:
            print(
                f"[WARN] LIME: không có mẫu giải thích "
                f"cho lớp {class_name}."
            )
            continue

        per_sample_abs_weights = []

        for sample_number, sample_idx in enumerate(
            sample_indices,
            start=1,
        ):
            explanation = explainer.explain_instance(
                data_row=X_test[sample_idx],
                predict_fn=predict_fn,
                labels=(class_idx,),
                num_features=len(feature_names),
                num_samples=LIME_NUM_SAMPLES,
            )

            local_weights = np.zeros(
                len(feature_names),
                dtype=float,
            )

            feature_weights = explanation.as_map().get(
                class_idx,
                [],
            )

            for feature_idx, weight in feature_weights:
                if 0 <= feature_idx < len(feature_names):
                    local_weights[feature_idx] = float(weight)

            per_sample_abs_weights.append(
                np.abs(local_weights)
            )

            for feature_idx, feature_name in enumerate(feature_names):
                local_rows.append([
                    class_idx,
                    class_name,
                    int(sample_idx),
                    sample_number,
                    feature_name,
                    float(local_weights[feature_idx]),
                    float(abs(local_weights[feature_idx])),
                ])

        class_counts[class_idx] = len(sample_indices)

        class_importance[class_idx] = np.mean(
            np.vstack(per_sample_abs_weights),
            axis=0,
        )

        save_importance_plot(
            class_importance[class_idx],
            feature_names,
            f"LIME - {model_name} - {class_name}",
            output_dir / f"LIME_{safe_filename(class_name)}_bar.png",
        )

        print(
            f"[INFO] LIME {class_name}: "
            f"{len(sample_indices)} samples; "
            f"test support={int(test_support[class_idx])}"
        )

    save_csv(
        output_dir / "LIME_local_weights.csv",
        [
            "class_index",
            "class_name",
            "test_row_index",
            "sample_number_in_class",
            "feature",
            "lime_weight_signed",
            "lime_weight_absolute",
        ],
        local_rows,
    )

    # Global LIME trung bình đều giữa các lớp có importance hợp lệ.
    valid_classes = np.flatnonzero(
        (class_counts > 0)
        & np.all(np.isfinite(class_importance), axis=1)
    )

    if len(valid_classes) > 0:
        global_importance = np.mean(
            class_importance[valid_classes],
            axis=0,
        )

        save_importance_plot(
            global_importance,
            feature_names,
            f"LIME - {model_name} - Global Class Balanced",
            output_dir / "LIME_global_bar.png",
        )

    save_importance_reports(
        method="LIME",
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print("[OK] Completed LIME")


# ============================================================
# 12. MAIN
# ============================================================

def run_global_signal22():
    print("=" * 80)
    print("SIGNAL2022 MULTICLASS GLOBAL XAI")
    print("=" * 80)

    print(f"Data directory : {PREPROCESSED_DIR.resolve()}")
    print(f"Model directory: {MODEL_DIR.resolve()}")
    print(f"Output directory: {SAVE_DIR.resolve()}")

    X_test, y_test, feature_names, test_support = load_dataset()

    X_train = load_X_train_if_available(
        X_test.shape[1]
    )

    explain_idx, background_idx, lime_indices = (
        make_or_load_fixed_indices(X_test, y_test)
    )

    # Lưu mapping lớp và support thật.
    save_csv(
        SAVE_DIR / "class_mapping.csv",
        [
            "class_index",
            "class_name",
            "test_support",
        ],
        [
            [
                class_idx,
                CLASS_NAMES[class_idx],
                int(test_support[class_idx]),
            ]
            for class_idx in range(N_CLASSES_EXPECTED)
        ],
    )

    print("\n" + "=" * 80)
    print("RUN CONFIGURATION")
    print(f"Test samples       : {len(X_test)}")
    print(f"Features           : {len(feature_names)}")
    print(f"Classes            : {N_CLASSES_EXPECTED}")
    print(f"Explain samples    : {len(explain_idx)}")
    print(f"Background samples : {len(background_idx)}")
    print(f"LIME max/class     : {N_LIME_PER_CLASS}")
    print(f"KernelSHAP nsamples: {KERNEL_NSAMPLES}")
    print("Model metrics      : NOT RUN")
    print("=" * 80)

    for model_name in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            model, model_type = load_model(model_name)

            output_dir = SAVE_DIR / model_name
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Kiểm tra model trả đúng số lớp và xác suất hợp lệ.
            check_idx = explain_idx[:min(5, len(explain_idx))]

            predict_probabilities(
                model,
                X_test[check_idx],
                model_type,
                model_name,
            )

            # Lưu dự đoán trên các mẫu được chọn để giải thích.
            probabilities = predict_probabilities(
                model,
                X_test[explain_idx],
                model_type,
                model_name,
            )

            predicted_classes = np.argmax(
                probabilities,
                axis=1,
            )

            prediction_rows = []

            for pos, sample_idx in enumerate(explain_idx):
                row = [
                    int(sample_idx),
                    int(y_test[sample_idx]),
                    CLASS_NAMES[int(y_test[sample_idx])],
                    int(predicted_classes[pos]),
                    CLASS_NAMES[int(predicted_classes[pos])],
                ]

                row.extend(
                    float(p) for p in probabilities[pos]
                )

                prediction_rows.append(row)

            save_csv(
                output_dir / "explain_sample_predictions.csv",
                [
                    "test_row_index",
                    "true_class_index",
                    "true_class_name",
                    "predicted_class_index",
                    "predicted_class_name",
                ] + [
                    f"probability_{name}"
                    for name in CLASS_NAMES
                ],
                prediction_rows,
            )

            explainers = []

            if model_name in TREE_MODELS:
                explainers.append("TreeSHAP")

            # KernelSHAP cho model sklearn không phải cây.
            # CNN/LSTM bỏ qua KernelSHAP để hạn chế thời gian.
            if model_type == "sklearn" and model_name not in TREE_MODELS:
                explainers.append("KernelSHAP")

            if model_type == "keras":
                explainers.append("GradientSHAP")

            explainers.append("LIME")

            for method in explainers:
                print(f"\n--- {model_name}: {method} ---")

                method_output_dir = output_dir / method
                method_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                try:
                    if method == "LIME":
                        run_lime(
                            model=model,
                            model_type=model_type,
                            model_name=model_name,
                            X_test=X_test,
                            y_test=y_test,
                            X_train=X_train,
                            lime_indices=lime_indices,
                            feature_names=feature_names,
                            test_support=test_support,
                            output_dir=method_output_dir,
                        )

                    else:
                        run_shap_method(
                            method=method,
                            model=model,
                            model_type=model_type,
                            model_name=model_name,
                            X_test=X_test,
                            y_test=y_test,
                            explain_idx=explain_idx,
                            background_idx=background_idx,
                            feature_names=feature_names,
                            test_support=test_support,
                            output_dir=method_output_dir,
                        )

                except Exception as exc:
                    print(
                        f"[ERROR] {model_name}/{method}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        except Exception as exc:
            print(
                f"[MODEL ERROR] {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 80)
    print("HOÀN THÀNH SIGNAL2022 MULTICLASS GLOBAL XAI")
    print(f"Output: {SAVE_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    # Tránh dùng GPU và hạn chế lỗi tài nguyên khi chạy TensorFlow.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    run_global_signal22()
