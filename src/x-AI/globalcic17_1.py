import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Phải chọn backend trước khi import pyplot để tránh Tkinter GUI.
import matplotlib
matplotlib.use("Agg")

import csv
import joblib
import hashlib
import warnings
from pathlib import Path

import numpy as np
import shap
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
SAVE_DIR = Path("XAI_Results3") / "global"

DATASET_NAME = "CIC17"
N_CLASSES_EXPECTED = 8

# Tăng theo khả năng tính toán.
N_GLOBAL = 400
N_BACKGROUND = 30

# LIME lấy tối đa 50 mẫu cho mỗi lớp, từ tập explain cố định.
N_LIME_PER_CLASS = 50
LIME_NUM_SAMPLES = 1000

# KernelSHAP tốn thời gian. Có thể thử 100, 300, 500...
KERNEL_NSAMPLES = 300

RANDOM_SEED = 42

CLASS_NAMES_FALLBACK = [
    "Botnet/Infiltration",
    "Brute Force",
    "DDoS",
    "DoS",
    "Normal",
    "Other",
    "Port Scan",
    "Web Attack",
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

# Chỉ sử dụng fallback nếu đã xác minh rằng mã nhãn 0..7
# đúng với CLASS_NAMES_FALLBACK trong pipeline huấn luyện.
ALLOW_VERIFIED_FALLBACK_CLASS_ORDER = True

# Chỉ chấp nhận các file train đã tiền xử lý có tên cụ thể.
X_TRAIN_CANDIDATES = [
    "preprocessed_multi_cic17_X_train.npy",
    "preprocessed_cic17_X_train.npy",
]

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
    # Preprocessing lưu các artifact bằng joblib.dump; joblib.load cũng đọc
    # được các pickle thông thường nếu metadata cũ dùng pickle trực tiếp.
    return joblib.load(path)

def save_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def safe_filename(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )

def close_plot():
    # Giải phóng figure sau khi lưu, không mở GUI.
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
# 3. FEATURE NAMES AND LABEL ENCODER
# ============================================================

def load_feature_names(n_features):
    # Không tìm metadata chung của dataset khác.
    candidates = [
        "preprocessed_multi_cic17_feature_names.pkl",
        "preprocessed_multi_cic17_feature_names.npy",
    ]

    path = find_existing_file(PREPROCESSED_DIR, candidates)

    if path is None:
        raise FileNotFoundError(
            "Không tìm thấy metadata feature names dành riêng cho CIC17.\n"
            f"Thư mục: {PREPROCESSED_DIR}\n"
            f"Tên được chấp nhận: {candidates}\n"
            "Không tự dựng tên feature hoặc dùng metadata dataset khác."
        )

    try:
        if path.suffix.lower() == ".npy":
            names = np.load(path, allow_pickle=True)
        else:
            names = load_pickle(path)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được feature metadata {path.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(names, dict):
        found = False
        for key in (
            "feature_names",
            "selected_features",
            "features",
            "columns",
        ):
            if key in names:
                names = names[key]
                found = True
                break

        if not found:
            raise ValueError(
                f"{path.name} là dictionary nhưng không có "
                "key feature names đã biết."
            )

    names = list(names)

    if len(names) != n_features:
        raise ValueError(
            f"X_test có {n_features} feature nhưng "
            f"{path.name} có {len(names)} tên."
        )

    names = [str(name) for name in names]

    if len(set(names)) != len(names):
        print(
            "[WARN] Feature names có tên trùng nhau. "
            "Kiểm tra metadata preprocessing."
        )

    print(f"[INFO] Feature metadata: {path}")
    return names

def load_label_encoder():
    # Không dùng glob rộng để tránh nạp nhầm encoder.
    candidates = [
        "preprocessed_multi_cic17_label_encoder.pkl",
        "preprocessed_multi_cic17_le.pkl",
    ]

    path = find_existing_file(PREPROCESSED_DIR, candidates)

    if path is None:
        print(
            "[WARN] Không tìm thấy encoder CIC17 theo tên chuẩn. "
            "Chỉ có thể dùng fallback nếu thứ tự nhãn số đã được "
            "xác minh độc lập từ pipeline huấn luyện."
        )
        return None

    try:
        encoder = load_pickle(path)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được encoder {path.name}: "
            f"{type(exc).__name__}: {exc}\n"
            "Kiểm tra đúng định dạng file và artifact sinh ra "
            "từ pipeline preprocessing."
        ) from exc

    if not hasattr(encoder, "classes_"):
        raise ValueError(
            f"{path.name} không có thuộc tính classes_."
        )

    print(f"[INFO] Label encoder: {path}")
    return encoder

def load_dataset():
    x_path = (
        PREPROCESSED_DIR /
        "preprocessed_multi_cic17_X_test.npy"
    )
    y_path = (
        PREPROCESSED_DIR /
        "preprocessed_multi_cic17_y_test.npy"
    )

    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            "Thiếu file test CIC17:\n"
            f"{x_path}\n{y_path}"
        )

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test phải là ma trận 2D, nhận {X_test.shape}."
        )

    if y_test.ndim > 1:
        if y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.reshape(-1)

    y_test = np.asarray(y_test)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Số dòng X_test và y_test không bằng nhau."
        )

    if len(y_test) == 0:
        raise ValueError("Test set rỗng.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test chứa NaN hoặc Inf.")

    encoder = load_label_encoder()

    # Nhãn chuỗi cần encoder để ánh xạ sang chỉ số.
    if y_test.dtype.kind in "OUS":
        if encoder is None:
            raise ValueError(
                "y_test là nhãn chuỗi nhưng không có encoder hợp lệ."
            )

        mapping = {
            value: i
            for i, value in enumerate(encoder.classes_)
        }

        try:
            y_test = np.asarray(
                [mapping[value] for value in y_test],
                dtype=int,
            )
        except KeyError as exc:
            raise ValueError(
                f"Nhãn {exc} không tồn tại trong encoder.classes_."
            ) from exc
    else:
        y_test = y_test.astype(int)

    if encoder is not None:
        class_names = [str(x) for x in encoder.classes_]

        if len(class_names) != N_CLASSES_EXPECTED:
            raise ValueError(
                f"Encoder có {len(class_names)} lớp, "
                f"mong đợi {N_CLASSES_EXPECTED}."
            )
    else:
        if not ALLOW_VERIFIED_FALLBACK_CLASS_ORDER:
            raise ValueError(
                "Không có encoder hợp lệ và fallback đã bị tắt."
            )

        class_names = CLASS_NAMES_FALLBACK.copy()

        print(
            "[WARN] Sử dụng CLASS_NAMES_FALLBACK. "
            "Phải xác minh thứ tự lớp này khớp pipeline huấn luyện."
        )

    if np.any(y_test < 0) or np.any(y_test >= N_CLASSES_EXPECTED):
        raise ValueError("y_test chứa chỉ số lớp ngoài 0..7.")

    feature_names = load_feature_names(X_test.shape[1])

    # Support thật của toàn bộ test set.
    test_support = np.bincount(
        y_test,
        minlength=N_CLASSES_EXPECTED,
    )

    if len(test_support) != N_CLASSES_EXPECTED:
        raise ValueError("Không tính được support đủ 8 lớp.")

    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")
    print(f"[INFO] Feature count: {len(feature_names)}")
    print(f"[INFO] Class order: {class_names}")

    for i, name in enumerate(class_names):
        print(
            f"  Class {i}: {name}; "
            f"test support = {test_support[i]}"
        )

    return (
        X_test,
        y_test,
        feature_names,
        class_names,
        test_support,
    )

def load_X_train_if_available(n_features):
    path = find_existing_file(
        PREPROCESSED_DIR,
        X_TRAIN_CANDIDATES,
    )

    if path is None:
        print(
            "[WARN] Không tìm thấy X_train CIC17 đã tiền xử lý. "
            "LIME sẽ dùng X_test làm dữ liệu nền."
        )
        return None

    X_train = np.load(path)

    if X_train.ndim != 2:
        raise ValueError(
            f"{path.name} phải là ma trận 2D."
        )

    if X_train.shape[1] != n_features:
        raise ValueError(
            f"{path.name} có {X_train.shape[1]} feature nhưng "
            f"X_test có {n_features}."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(f"{path.name} chứa NaN hoặc Inf.")

    print(f"[INFO] LIME training background: {path}")
    return X_train

# ============================================================
# 4. LOAD MODELS
# ============================================================

def load_model(model_name):
    if model_name in DEEP_MODELS:
        candidates = [
            f"BEST_{model_name}_{DATASET_NAME}.keras",
            f"{DATASET_NAME}_{model_name}.keras",
            f"CIC17_{model_name}.keras",
            f"CIC17_{model_name}.h5",
            f"BEST_{model_name}_{DATASET_NAME}.h5",
        ]

        path = find_existing_file(MODEL_DIR, candidates)

        if path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model Keras {model_name}.\n"
                f"Thư mục: {MODEL_DIR}\n"
                f"Đã thử: {candidates}"
            )

        model = tf.keras.models.load_model(
            path,
            compile=False,
        )

    else:
        candidates = [
            f"{DATASET_NAME}_{model_name}.pkl",
            f"BEST_{model_name}_{DATASET_NAME}.pkl",
        ]

        path = find_existing_file(MODEL_DIR, candidates)

        if path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model {model_name}.\n"
                f"Thư mục: {MODEL_DIR}\n"
                f"Đã thử: {candidates}"
            )

        try:
            model = load_pickle(path)
        except Exception as exc:
            raise RuntimeError(
                f"Không đọc được model {path.name}: {exc}"
            ) from exc

    print(f"[INFO] Loaded {model_name}: {path}")
    return model

def prepare_model_input(X, model_name):
    if model_name == "CNN":
        return X.reshape(X.shape[0], X.shape[1], 1)

    if model_name == "LSTM":
        return X.reshape(X.shape[0], 1, X.shape[1])

    return X

# ============================================================
# 5. PROBABILITY OUTPUT AND CLASS ORDER
# ============================================================

def reorder_sklearn_probabilities(model, probabilities):
   
    if not hasattr(model, "classes_"):
        raise ValueError(
            f"{type(model).__name__} không có classes_. "
            "Không thể xác minh thứ tự predict_proba."
        )

    model_classes = np.asarray(model.classes_)

    if model_classes.ndim != 1:
        raise ValueError(
            f"model.classes_ không phải vector: {model_classes.shape}"
        )

    try:
        class_ids = model_classes.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"model.classes_={model_classes.tolist()} không phải "
            "mã số lớp nguyên. Cần mapping tường minh từ pipeline."
        ) from exc

    if sorted(class_ids.tolist()) != list(range(N_CLASSES_EXPECTED)):
        raise ValueError(
            f"model.classes_={model_classes.tolist()} không phải "
            "đủ mã lớp 0..7. Dừng để tránh giải thích sai lớp."
        )

    reordered = np.zeros(
        (probabilities.shape[0], N_CLASSES_EXPECTED),
        dtype=float,
    )

    for source_col, class_id in enumerate(class_ids):
        reordered[:, class_id] = probabilities[:, source_col]

    return reordered

def predict_probabilities(model, model_name, X):

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"predict_probabilities cần X 2D, nhận {X.shape}."
        )

    X_input = prepare_model_input(X, model_name)

    if model_name in DEEP_MODELS:
        probabilities = model.predict(X_input, verbose=0)
    else:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"{model_name} không có predict_proba."
            )
        probabilities = model.predict_proba(X_input)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.ndim != 2:
        raise ValueError(
            f"{model_name} output shape không hợp lệ: "
            f"{probabilities.shape}"
        )
    if probabilities.shape != (len(X), N_CLASSES_EXPECTED):
        raise ValueError(
            f"{model_name} trả shape {probabilities.shape}; "
            f"mong đợi {(len(X), N_CLASSES_EXPECTED)}."
        )
    if model_name not in DEEP_MODELS:
        probabilities = reorder_sklearn_probabilities(
            model,
            probabilities,
        )
    else:
        # Với Keras, phải xác minh thứ tự softmax khớp mã nhãn
        # trong pipeline train. Không thể suy ra từ output shape.
        pass

    if not np.isfinite(probabilities).all():
        raise ValueError("Xác suất chứa NaN hoặc Inf.")

    if np.any(probabilities < -1e-6):
        raise ValueError("Xác suất có giá trị âm.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Tổng xác suất mỗi dòng không gần 1. "
            "Kiểm tra đầu ra model có phải xác suất softmax không."
        )

    return probabilities

# ============================================================
# 6. FIXED EXPLANATION/BACKGROUND INDICES
# ============================================================

def stratified_indices(y, n_total, seed):
    """
    Tạo tập chỉ số phân tầng tái lập.
    Ưu tiên lấy ít nhất một mẫu cho mỗi lớp có mặt.
    """
    y = np.asarray(y)
    n_total = min(int(n_total), len(y))

    if n_total <= 0:
        raise ValueError("Số mẫu cần lấy phải lớn hơn 0.")

    if n_total == len(y):
        return np.arange(len(y), dtype=int)

    classes = np.unique(y)

    if n_total < len(classes):
        raise ValueError(
            f"Yêu cầu {n_total} mẫu nhưng có "
            f"{len(classes)} lớp hiện diện."
        )

    rng = np.random.default_rng(seed)
    groups = {c: rng.permutation(np.flatnonzero(y == c)) for c in classes}
    allocation = {c: 0 for c in classes}
    remaining = n_total
    # Rotate remainder slots so they do not systematically favor low class IDs.
    while remaining:
        active = [c for c in classes if allocation[c] < len(groups[c])]
        if not active:
            break
        for c in rng.permutation(active):
            allocation[c] += 1
            remaining -= 1
            if remaining == 0:
                break
    selected = [groups[c][:allocation[c]] for c in classes if allocation[c]]
    return np.sort(np.concatenate(selected).astype(int))

def stratified_class_indices(y, max_per_class, seed):
    """Select up to max_per_class fixed examples independently for each class."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    return {
        int(c): np.sort(rng.permutation(np.flatnonzero(y == c))[:max_per_class]).astype(int)
        for c in np.unique(y)
    }

def make_or_load_fixed_indices(X_test, y_test):
    """
    Tái sử dụng indices nếu test set có cùng fingerprint.
    Nếu test set đổi, tạo lại indices để tránh dùng nhầm file cũ.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    data_fingerprint = dataset_fingerprint(X_test, y_test)
    config = (
        f"sampling-v2|balanced|global={N_GLOBAL}|background={N_BACKGROUND}|"
        f"lime_per_class={N_LIME_PER_CLASS}|seed={RANDOM_SEED}"
    )
    fingerprint = hashlib.sha256(
        f"{data_fingerprint}|{config}".encode("utf-8")
    ).hexdigest()
    cache_tag = hashlib.sha256(config.encode("utf-8")).hexdigest()[:12]
    explain_path = SAVE_DIR / f"{DATASET_NAME}_explain_indices_{cache_tag}.npy"
    background_path = SAVE_DIR / f"{DATASET_NAME}_background_indices_{cache_tag}.npy"
    lime_path = SAVE_DIR / f"{DATASET_NAME}_lime_indices_{cache_tag}.npz"
    fingerprint_path = SAVE_DIR / f"{DATASET_NAME}_sampling_fingerprint_{cache_tag}.txt"

    previous_fingerprint = None
    if fingerprint_path.is_file():
        previous_fingerprint = (
            fingerprint_path.read_text(encoding="utf-8").strip()
        )

    same_dataset = previous_fingerprint == fingerprint

    def valid_indices(indices):
        return (
            indices.ndim == 1
            and len(indices) > 0
            and np.all(indices >= 0)
            and np.all(indices < len(y_test))
            and len(np.unique(indices)) == len(indices)
        )

    explain_idx = None
    background_idx = None

    if same_dataset and explain_path.is_file():
        candidate = np.load(explain_path)
        if valid_indices(candidate) and len(candidate) == min(N_GLOBAL, len(y_test)):
            explain_idx = candidate.astype(int)

    if same_dataset and background_path.is_file():
        candidate = np.load(background_path)
        if valid_indices(candidate) and len(candidate) == min(N_BACKGROUND, len(y_test)):
            background_idx = candidate.astype(int)

    lime_indices = None
    if same_dataset and lime_path.is_file():
        with np.load(lime_path) as cached:
            candidate_map = {int(k): cached[k].astype(int) for k in cached.files}
        expected_classes = set(map(int, np.unique(y_test)))
        if set(candidate_map) == expected_classes and all(
            valid_indices(v)
            and len(v) == min(N_LIME_PER_CLASS, int(np.sum(y_test == k)))
            and np.all(y_test[v] == k)
            for k, v in candidate_map.items()
        ):
            lime_indices = candidate_map

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
        print(
            f"[INFO] Reusing background indices: {background_path}"
        )

    if lime_indices is None:
        lime_indices = stratified_class_indices(y_test, N_LIME_PER_CLASS, RANDOM_SEED + 2)
        np.savez(lime_path, **{str(k): v for k, v in lime_indices.items()})
        print(f"[INFO] Saved per-class LIME indices: {lime_path}")
    else:
        print(f"[INFO] Reusing per-class LIME indices: {lime_path}")

    fingerprint_path.write_text(
        fingerprint,
        encoding="utf-8",
    )

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
            manifest_rows.append(["lime", int(idx), int(class_idx)])

    save_csv(
        SAVE_DIR / f"{DATASET_NAME}_sample_manifest.csv",
        [
            "sample_role",
            "test_row_index",
            "true_class_index",
        ],
        manifest_rows,
    )

    # Số mẫu giải thích thực tế của từng lớp.
    count_rows = []

    for class_idx in range(N_CLASSES_EXPECTED):
        count_rows.append([
            class_idx,
            int(np.sum(y_test == class_idx)),
            int(np.sum(y_test[explain_idx] == class_idx)),
            int(np.sum(y_test[background_idx] == class_idx)),
        ])

    save_csv(
        SAVE_DIR / f"{DATASET_NAME}_sample_counts.csv",
        [
            "class_index",
            "test_support",
            "n_explain_samples",
            "n_background_samples",
            "n_lime_samples",
        ],
        [
            row + [len(lime_indices.get(int(row[0]), []))]
            for row in count_rows
        ],
    )

    return explain_idx, background_idx, lime_indices

# ============================================================
# 7. SHAP OUTPUT NORMALIZATION
# ============================================================

def normalize_one_shap_class(arr, n_samples, n_features):
    arr = np.asarray(arr)

    # Bỏ trục singleton ngoài trục sample.
    while arr.ndim > 2:
        singleton_axes = [
            axis for axis in range(1, arr.ndim)
            if arr.shape[axis] == 1
        ]

        if not singleton_axes:
            break

        arr = np.squeeze(arr, axis=singleton_axes[-1])

    if arr.ndim == 1 and n_features == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(
            f"Không chuẩn hóa được SHAP class array: {arr.shape}"
        )

    if arr.shape == (n_samples, n_features):
        return arr

    if arr.shape == (n_features, n_samples):
        return arr.T

    raise ValueError(
        f"SHAP class shape {arr.shape}; mong đợi "
        f"({n_samples}, {n_features})."
    )

def normalize_shap_values(
    raw_values,
    n_samples,
    n_features,
    n_classes,
):
    """
    Chuẩn hóa SHAP về (samples, features, classes).
    """
    if isinstance(raw_values, list):
        if len(raw_values) != n_classes:
            raise ValueError(
                f"SHAP list có {len(raw_values)} phần tử, "
                f"mong đợi {n_classes}."
            )

        class_arrays = [
            normalize_one_shap_class(
                value,
                n_samples,
                n_features,
            )
            for value in raw_values
        ]

        result = np.stack(class_arrays, axis=-1)

    else:
        arr = np.asarray(raw_values)

        # CNN/LSTM có thể thêm trục channel/timestep singleton.
        while arr.ndim > 3:
            singleton_axes = [
                axis for axis in range(1, arr.ndim)
                if arr.shape[axis] == 1
            ]

            if not singleton_axes:
                raise ValueError(
                    f"SHAP có số chiều không hỗ trợ: {arr.shape}"
                )

            arr = np.squeeze(arr, axis=singleton_axes[-1])

        if arr.ndim == 2:
            if arr.shape != (n_samples, n_features):
                raise ValueError(
                    f"SHAP 2D không khớp: {arr.shape}"
                )
            result = arr[:, :, None]

        elif arr.ndim == 3:
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"Số mẫu SHAP không khớp: {arr.shape}"
                )

            # (N, F, C)
            if arr.shape[1] == n_features:
                result = arr

            # (N, C, F)
            elif arr.shape[2] == n_features:
                result = np.transpose(arr, (0, 2, 1))

            else:
                raise ValueError(
                    f"Không xác định được trục feature: {arr.shape}"
                )

        else:
            raise ValueError(
                f"SHAP output không hỗ trợ: {arr.shape}"
            )

    expected = (n_samples, n_features, n_classes)

    if result.shape != expected:
        raise ValueError(
            f"SHAP sau chuẩn hóa có shape {result.shape}; "
            f"mong đợi {expected}."
        )

    if not np.isfinite(result).all():
        raise ValueError("SHAP values chứa NaN hoặc Inf.")

    return result.astype(float)

def reorder_tree_shap_classes(raw_values, model):
    """
    Đưa TreeSHAP về thứ tự lớp 0..7.
    Chỉ hỗ trợ khi classes_ của model là mã lớp 0..7.
    """
    if not hasattr(model, "classes_"):
        raise ValueError(
            "TreeSHAP: model không có classes_ để xác minh thứ tự."
        )

    class_ids = np.asarray(model.classes_).astype(int)

    if sorted(class_ids.tolist()) != list(range(N_CLASSES_EXPECTED)):
        raise ValueError(
            f"TreeSHAP classes_ không phải 0..7: {class_ids.tolist()}"
        )

    # raw_values list: thứ tự phần tử tương ứng model.classes_.
    if isinstance(raw_values, list):
        if len(raw_values) != N_CLASSES_EXPECTED:
            raise ValueError(
                f"TreeSHAP trả {len(raw_values)} lớp."
            )

        reordered = [None] * N_CLASSES_EXPECTED

        for source_pos, class_id in enumerate(class_ids):
            reordered[class_id] = raw_values[source_pos]

        return reordered

    # Với ndarray, thứ tự lớp thường theo classes_.
    arr = np.asarray(raw_values)

    if arr.ndim == 3:
        if arr.shape[-1] == N_CLASSES_EXPECTED:
            return arr[:, :, np.argsort(class_ids)]

        if arr.shape[1] == N_CLASSES_EXPECTED:
            return arr[:, np.argsort(class_ids), :]

    return raw_values

# ============================================================
# 8. IMPORTANCE REPORTS AND PLOTS
# ============================================================

def save_importance_reports(
    method,
    model_name,
    class_names,
    feature_names,
    class_importance,
    class_explain_counts,
    test_support,
    output_dir,
):
    """
    class_importance[c, j] là mean absolute importance.

    Natural:
        trọng số theo support thật của toàn bộ test set.

    Balanced:
        trung bình đều giữa các lớp có importance hợp lệ.
    """
    n_classes = len(class_names)
    n_features = len(feature_names)

    per_class_rows = []

    for c in range(n_classes):
        for j in range(n_features):
            value = class_importance[c, j]

            per_class_rows.append([
                class_names[c],
                int(class_explain_counts[c]),
                int(test_support[c]),
                feature_names[j],
                (
                    float(value)
                    if np.isfinite(value)
                    else "NaN"
                ),
            ])

    save_csv(
        output_dir / f"{method}_per_class_importance.csv",
        [
            "class_name",
            "n_explained_samples",
            "test_support",
            "feature",
            "mean_abs_importance",
        ],
        per_class_rows,
    )

    valid_classes = np.asarray([
        c for c in range(n_classes)
        if class_explain_counts[c] > 0
        and np.isfinite(class_importance[c]).all()
    ], dtype=int)

    # Cân bằng lớp: mỗi lớp hợp lệ trọng số bằng nhau.
    if len(valid_classes) > 0:
        balanced = np.mean(
            class_importance[valid_classes],
            axis=0,
        )
    else:
        balanced = np.full(n_features, np.nan)

    # Natural: không âm thầm bỏ qua lớp có support nhưng thiếu
    # importance. Nếu thiếu lớp, ghi NaN để tránh kết luận sai.
    supported_classes = np.flatnonzero(test_support > 0)

    all_supported_valid = (
        len(supported_classes) > 0
        and all(
            c in valid_classes
            for c in supported_classes
        )
    )

    if all_supported_valid:
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
        natural = np.full(n_features, np.nan)

        print(
            f"[WARN] {method}: không thể tính natural importance "
            "vì ít nhất một lớp có support trong test nhưng "
            "không có importance hợp lệ."
        )

    aggregate_rows = []

    for j, feature in enumerate(feature_names):
        aggregate_rows.append([
            feature,
            (
                float(natural[j])
                if np.isfinite(natural[j])
                else "NaN"
            ),
            (
                float(balanced[j])
                if np.isfinite(balanced[j])
                else "NaN"
            ),
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

    print(
        f"[OK] Saved {method} reports for {model_name}"
    )

def save_class_plot(
    values,
    X_plot,
    feature_names,
    class_name,
    method,
    model_name,
    output_dir,
):
    if values.size == 0 or X_plot.shape[0] == 0:
        return

    try:
        shap.summary_plot(
            values,
            X_plot,
            feature_names=feature_names,
            show=False,
            max_display=min(20, len(feature_names)),
        )

        plt.title(
            f"{method} - {model_name} - {class_name}",
            fontsize=11,
        )
        plt.tight_layout()

        filename = (
            f"{method}_{safe_filename(class_name)}.png"
        )

        plt.savefig(
            output_dir / filename,
            dpi=160,
            bbox_inches="tight",
        )

        print(f"[OK] Saved plot: {filename}")

    finally:
        close_plot()

# ============================================================
# 9. SHAP METHODS
# ============================================================

def calculate_shap(
    method,
    model,
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

        return normalize_shap_values(
            raw_values,
            n_samples,
            n_features,
            N_CLASSES_EXPECTED,
        )

    if method == "KernelSHAP":
        def predict_fn(data):
            return predict_probabilities(
                model,
                model_name,
                data,
            )

        explainer = shap.KernelExplainer(
            predict_fn,
            X_background,
        )

        raw_values = explainer.shap_values(
            X_explain,
            nsamples=KERNEL_NSAMPLES,
        )

        return normalize_shap_values(
            raw_values,
            n_samples,
            n_features,
            N_CLASSES_EXPECTED,
        )

    if method == "GradientSHAP":
        background_input = prepare_model_input(
            X_background,
            model_name,
        )

        explain_input = prepare_model_input(
            X_explain,
            model_name,
        )

        explainer = shap.GradientExplainer(
            model,
            background_input,
        )

        raw_values = explainer.shap_values(
            explain_input
        )

        return normalize_shap_values(
            raw_values,
            n_samples,
            n_features,
            N_CLASSES_EXPECTED,
        )

    raise ValueError(
        f"SHAP method không hỗ trợ: {method}"
    )

def run_shap_method(
    method,
    model,
    model_name,
    X_test,
    y_test,
    explain_idx,
    background_idx,
    feature_names,
    class_names,
    test_support,
    output_dir,
):
    print(f"[INFO] Running {method}")

    # Tập mẫu và background cố định dùng chung giữa các explainer.
    X_explain = X_test[explain_idx]
    y_explain = y_test[explain_idx]
    X_background = X_test[background_idx]

    # Kiểm tra xác suất trước khi tính SHAP.
    probs = predict_probabilities(
        model,
        model_name,
        X_explain,
    )

    if probs.shape != (len(X_explain), N_CLASSES_EXPECTED):
        raise ValueError(
            f"Probability shape sai: {probs.shape}"
        )

    shap_values = calculate_shap(
        method,
        model,
        model_name,
        X_background,
        X_explain,
    )

    n_classes = len(class_names)
    n_features = len(feature_names)

    class_importance = np.full(
        (n_classes, n_features),
        np.nan,
    )

    class_counts = np.zeros(n_classes, dtype=int)

    for class_idx, class_name in enumerate(class_names):
        # Phân tích riêng các mẫu có nhãn thật thuộc lớp này.
        mask = y_explain == class_idx
        n = int(mask.sum())
        class_counts[class_idx] = n

        if n == 0:
            print(
                f"[WARN] {method}: không có mẫu cho "
                f"lớp {class_name}."
            )
            continue

        class_sv = shap_values[mask, :, class_idx]
        class_X = X_explain[mask]

        if class_sv.shape != (n, n_features):
            raise ValueError(
                f"{method}/{class_name} shape sai: "
                f"{class_sv.shape}"
            )

        # Global importance theo độ lớn tuyệt đối.
        class_importance[class_idx] = np.mean(
            np.abs(class_sv),
            axis=0,
        )

        save_class_plot(
            class_sv,
            class_X,
            feature_names,
            class_name,
            method,
            model_name,
            output_dir,
        )

    save_importance_reports(
        method=method,
        model_name=model_name,
        class_names=class_names,
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print(f"[OK] Completed {method}")

# ============================================================
# 10. LIME
# ============================================================

def run_lime(
    model,
    model_name,
    X_test,
    y_test,
    X_train,
    lime_indices,
    feature_names,
    class_names,
    test_support,
    output_dir,
):
    from lime.lime_tabular import LimeTabularExplainer

    print("[INFO] Running LIME")

    # Ưu tiên train data đã qua cùng preprocessing.
    # Nếu không có thì dùng X_test và thông báo rõ.
    lime_background = (
        X_train if X_train is not None else X_test
    )

    lime_exp = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_SEED,
    )

    def predict_fn(data):
        return predict_probabilities(
            model,
            model_name,
            data,
        )

    n_classes = len(class_names)
    n_features = len(feature_names)

    class_importance = np.full(
        (n_classes, n_features),
        np.nan,
    )

    class_counts = np.zeros(n_classes, dtype=int)
    all_rows = []

    # Dùng chung explain_idx, không lấy mẫu mới ngoài tập đó.
    for class_idx, class_name in enumerate(class_names):
        candidates = lime_indices.get(class_idx, np.asarray([], dtype=int))

        if len(candidates) == 0:
            print(
                f"[WARN] LIME: không có mẫu giải thích "
                f"cho lớp {class_name}."
            )
            continue

        # explain_idx đã được cố định và sắp xếp.
        class_sample_indices = candidates[:N_LIME_PER_CLASS]
        class_counts[class_idx] = len(class_sample_indices)

        per_sample_importance = []

        for sample_number, sample_idx in enumerate(
            class_sample_indices,
            start=1,
        ):
            explanation = lime_exp.explain_instance(
                data_row=X_test[sample_idx],
                predict_fn=predict_fn,
                labels=(class_idx,),
                num_features=n_features,
                num_samples=LIME_NUM_SAMPLES,
            )

            # Lấy weight theo index feature, không dùng regex.
            mapped = explanation.as_map().get(class_idx, [])
            signed_weights = np.zeros(
                n_features,
                dtype=float,
            )

            for feature_idx, weight in mapped:
                if 0 <= feature_idx < n_features:
                    signed_weights[feature_idx] = float(weight)

            per_sample_importance.append(
                np.abs(signed_weights)
            )

            for j, feature_name in enumerate(feature_names):
                all_rows.append([
                    class_name,
                    int(sample_idx),
                    sample_number,
                    feature_name,
                    float(signed_weights[j]),
                    float(abs(signed_weights[j])),
                ])

        if per_sample_importance:
            class_importance[class_idx] = np.mean(
                np.vstack(per_sample_importance),
                axis=0,
            )

        print(
            f"[INFO] LIME {class_name}: "
            f"{len(class_sample_indices)} mẫu giải thích; "
            f"test support={int(test_support[class_idx])}."
        )

        importance = class_importance[class_idx]

        if np.isfinite(importance).all():
            order = np.argsort(importance)[::-1][
                :min(20, n_features)
            ]

            try:
                plt.figure(figsize=(10, 7))

                plt.barh(
                    [feature_names[j] for j in order[::-1]],
                    importance[order[::-1]],
                )

                plt.xlabel("Mean absolute LIME weight")
                plt.title(
                    f"LIME - {model_name} - {class_name}"
                )
                plt.tight_layout()

                filename = (
                    f"LIME_{safe_filename(class_name)}.png"
                )

                plt.savefig(
                    output_dir / filename,
                    dpi=160,
                    bbox_inches="tight",
                )

                print(f"[OK] Saved plot: {filename}")

            finally:
                close_plot()

    save_csv(
        output_dir / "LIME_local_weights.csv",
        [
            "class_name",
            "test_row_index",
            "sample_number_in_class",
            "feature",
            "lime_weight_signed",
            "lime_weight_absolute",
        ],
        all_rows,
    )

    save_importance_reports(
        method="LIME",
        model_name=model_name,
        class_names=class_names,
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print("[OK] Completed LIME")

# ============================================================
# 11. MAIN
# ============================================================

def run_all_models_cic17():
    (
        X_test,
        y_test,
        feature_names,
        class_names,
        test_support,
    ) = load_dataset()

    X_train = load_X_train_if_available(
        n_features=X_test.shape[1]
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Tạo hoặc nạp chỉ số cố định chỉ một lần.
    explain_idx, background_idx, lime_indices = make_or_load_fixed_indices(
        X_test,
        y_test,
    )

    print("=" * 90)
    print("CIC17 MULTICLASS XAI")
    print(f"X_test: {X_test.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Class count: {len(class_names)}")
    print(f"Balanced explain samples: {len(explain_idx)}")
    print(f"Fixed background samples: {len(background_idx)}")
    print(f"Fixed LIME samples/class (max): {N_LIME_PER_CLASS}")
    print(f"KernelSHAP nsamples: {KERNEL_NSAMPLES}")
    print("Model performance metrics: NOT RUN")
    print("=" * 90)

    for model_name in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            model = load_model(model_name)

            output_dir = (
                SAVE_DIR / DATASET_NAME / model_name
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Lưu mapping và support test thật.
            save_csv(
                output_dir / "class_mapping.csv",
                [
                    "class_index",
                    "class_name",
                    "test_support",
                ],
                [
                    [
                        i,
                        class_names[i],
                        int(test_support[i]),
                    ]
                    for i in range(len(class_names))
                ],
            )

            # Kiểm tra output trước khi giải thích.
            check_idx = explain_idx[
                :min(5, len(explain_idx))
            ]

            check_probs = predict_probabilities(
                model,
                model_name,
                X_test[check_idx],
            )

            if check_probs.shape != (
                len(check_idx),
                N_CLASSES_EXPECTED,
            ):
                raise ValueError(
                    f"Model output shape sai: {check_probs.shape}"
                )

            explainers = []

            if model_name in {"RandomForest", "LightGBM"}:
                explainers.append("TreeSHAP")

            # KernelSHAP giữ theo cấu hình ban đầu:
            # bỏ qua CNN/LSTM vì chi phí tính toán và đầu vào.
            if model_name not in {"CNN", "LSTM"}:
                explainers.append("KernelSHAP")

            if model_name in DEEP_MODELS:
                explainers.append("GradientSHAP")

            explainers.append("LIME")

            for method in explainers:
                try:
                    method_output_dir = output_dir / method
                    method_output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    if method == "LIME":
                        run_lime(
                            model=model,
                            model_name=model_name,
                            X_test=X_test,
                            y_test=y_test,
                            X_train=X_train,
                            lime_indices=lime_indices,
                            feature_names=feature_names,
                            class_names=class_names,
                            test_support=test_support,
                            output_dir=method_output_dir,
                        )
                    else:
                        run_shap_method(
                            method=method,
                            model=model,
                            model_name=model_name,
                            X_test=X_test,
                            y_test=y_test,
                            explain_idx=explain_idx,
                            background_idx=background_idx,
                            feature_names=feature_names,
                            class_names=class_names,
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
                f"[ERROR] Không xử lý được model {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 90)
    print("HOÀN THÀNH CIC17 MULTICLASS XAI")
    print(f"Output directory: {SAVE_DIR.resolve()}")
    print("=" * 90)

if __name__ == "__main__":
    run_all_models_cic17()
