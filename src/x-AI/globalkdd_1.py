import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Chọn backend không cần GUI để tránh lỗi Tkinter.
import matplotlib
matplotlib.use("Agg")

import csv
import hashlib
import warnings
import pickle
import joblib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import shap
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

# Không chia thư mục SHAP/LIME.
SAVE_DIR = Path("XAI_Results3") / "global" / "NSL-KDD"

DATASET_NAME = "NSL-KDD"
N_CLASSES_EXPECTED = 6

CLASS_NAMES_FALLBACK = [
    "DoS",
    "Normal",
    "Other",
    "Probe",
    "R2L",
    "U2R",
]

# Giữ đúng thứ tự lớp đã dùng khi huấn luyện.
# Chỉ sử dụng fallback khi đã xác minh thứ tự này.
ALLOW_VERIFIED_FALLBACK_CLASS_ORDER = True

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
TREE_MODELS = {"RandomForest", "LightGBM", "AdaBoost"}

# Dùng cùng tập mẫu cho các model và explainer.
N_GLOBAL = 400
N_BACKGROUND = 30

# LIME lấy tối đa 50 mẫu cho mỗi lớp trong tập explain cố định.
N_LIME_PER_CLASS = 50
LIME_NUM_SAMPLES = 1000

# KernelSHAP khá tốn thời gian.
KERNEL_NSAMPLES = 300

RANDOM_SEED = 42

X_TRAIN_CANDIDATES = [
    "preprocessed_multi_nsl-kdd_X_train.npy",
]

FEATURE_NAME_CANDIDATES = [
    "preprocessed_multi_nsl-kdd_feature_names.pkl",
    "preprocessed_multi_nsl-kdd_feature_names.npy",
    "preprocessed_multi_nsl-kdd_selected_features.pkl",
]

LABEL_ENCODER_CANDIDATES = [
    "preprocessed_multi_nsl-kdd_label_encoder.pkl",
    "preprocessed_multi_nsl-kdd_le.pkl",
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
    # joblib.load hỗ trợ các artifact pickle tương thích.
    return joblib.load(path)

def save_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def safe_filename(value):
    result = str(value)
    for char in '<>:"/\\|?*':
        result = result.replace(char, "_")
    return result.replace(" ", "_")

def close_plot():
    plt.close("all")

def dataset_fingerprint(X, y):
    h = hashlib.sha256()

    h.update(str(X.shape).encode("utf-8"))
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(X).view(np.uint8))

    h.update(str(y.shape).encode("utf-8"))
    h.update(str(y.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(y).view(np.uint8))

    return h.hexdigest()

# ============================================================
# 3. FEATURE NAMES AND LABEL ENCODER
# ============================================================

def load_feature_names(n_features):
    path = find_existing_file(
        PREPROCESSED_DIR,
        FEATURE_NAME_CANDIDATES,
    )

    if path is None:
        raise FileNotFoundError(
            "Không tìm thấy metadata feature names dành riêng "
            "cho NSL-KDD.\n"
            f"Thư mục: {PREPROCESSED_DIR}\n"
            f"Các file đã kiểm tra: {FEATURE_NAME_CANDIDATES}"
        )

    try:
        if path.suffix.lower() == ".npy":
            names = np.load(path, allow_pickle=True)
        else:
            names = load_pickle(path)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được {path.name}: "
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
        print("[WARN] Metadata có tên feature trùng nhau.")

    print(f"[INFO] Feature metadata: {path}")
    return names

def load_label_encoder():
    path = find_existing_file(
        PREPROCESSED_DIR,
        LABEL_ENCODER_CANDIDATES,
    )

    if path is None:
        print("[WARN] Không tìm thấy label encoder NSL-KDD.")
        return None

    try:
        encoder = load_pickle(path)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được encoder {path.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(encoder, "classes_"):
        raise ValueError(
            f"{path.name} không có thuộc tính classes_."
        )

    if len(encoder.classes_) != N_CLASSES_EXPECTED:
        raise ValueError(
            f"Encoder có {len(encoder.classes_)} lớp, "
            f"mong đợi {N_CLASSES_EXPECTED}."
        )

    print(f"[INFO] Label encoder: {path}")
    return encoder

# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset():
    x_path = (
        PREPROCESSED_DIR
        / "preprocessed_multi_nsl-kdd_X_test.npy"
    )

    y_path = (
        PREPROCESSED_DIR
        / "preprocessed_multi_nsl-kdd_y_test.npy"
    )

    if not x_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {x_path}")

    if not y_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {y_path}")

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    if X_test.ndim != 2:
        raise ValueError(
            f"X_test phải có shape (N, F), nhận {X_test.shape}. "
            "Không tự reshape dữ liệu vì có thể làm sai feature."
        )

    if y_test.ndim > 1:
        if y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        else:
            y_test = y_test.reshape(-1)

    y_test = np.asarray(y_test)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Số dòng X_test và y_test không khớp."
        )

    if len(y_test) == 0:
        raise ValueError("Test set rỗng.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test chứa NaN hoặc Inf.")

    encoder = load_label_encoder()

    if y_test.dtype.kind in "OUS":
        if encoder is None:
            raise ValueError(
                "y_test chứa nhãn dạng chuỗi nhưng không có "
                "label encoder hợp lệ."
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
                f"Nhãn {exc} không có trong encoder.classes_."
            ) from exc
    else:
        y_test = y_test.astype(int)

    if encoder is not None:
        class_names = [str(x) for x in encoder.classes_]

        if len(class_names) != N_CLASSES_EXPECTED:
            raise ValueError("Encoder không có đủ 6 lớp.")

        if class_names != CLASS_NAMES_FALLBACK:
            raise ValueError(
                "Thứ tự encoder.classes_ không khớp với "
                "CLASS_NAMES_FALLBACK.\n"
                f"Encoder: {class_names}\n"
                f"Code: {CLASS_NAMES_FALLBACK}\n"
                "Hãy đồng bộ thứ tự lớp với pipeline huấn luyện "
                "trước khi chạy XAI."
            )
    else:
        if not ALLOW_VERIFIED_FALLBACK_CLASS_ORDER:
            raise ValueError(
                "Không có encoder hợp lệ và fallback đã bị tắt."
            )

        class_names = CLASS_NAMES_FALLBACK.copy()

        print(
            "[WARN] Dùng CLASS_NAMES_FALLBACK. "
            "Cần xác minh độc lập rằng mã 0..5 khớp thứ tự này."
        )

    if np.any(y_test < 0) or np.any(
        y_test >= N_CLASSES_EXPECTED
    ):
        raise ValueError(
            "y_test có chỉ số ngoài phạm vi lớp 0..5."
        )

    feature_names = load_feature_names(X_test.shape[1])

    test_support = np.bincount(
        y_test,
        minlength=N_CLASSES_EXPECTED,
    )

    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")
    print(f"[INFO] Feature count: {len(feature_names)}")

    for i, name in enumerate(class_names):
        print(
            f"  Class {i}: {name}; "
            f"test support={test_support[i]}"
        )

    return (
        X_test.astype(np.float32),
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
            "[WARN] Không tìm thấy X_train NSL-KDD. "
            "LIME sẽ sử dụng X_test làm dữ liệu tham chiếu."
        )
        return None

    X_train = np.load(path)

    if X_train.ndim != 2:
        raise ValueError(
            f"{path.name} phải có shape (N, F)."
        )

    if X_train.shape[1] != n_features:
        raise ValueError(
            f"{path.name} có {X_train.shape[1]} feature, "
            f"nhưng X_test có {n_features}."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(f"{path.name} chứa NaN hoặc Inf.")

    print(f"[INFO] LIME training reference: {path}")
    return X_train.astype(np.float32)

# ============================================================
# 5. LOAD MODELS
# ============================================================

def load_model(model_name):
    keras_candidates = [
        f"{DATASET_NAME}_{model_name}.keras",
        f"BEST_{model_name}_{DATASET_NAME}.keras",
        f"BEST_{DATASET_NAME}_{model_name}.keras",
    ]

    pkl_candidates = [
        f"{DATASET_NAME}_{model_name}.pkl",
    ]

    keras_path = find_existing_file(
        MODEL_DIR,
        keras_candidates,
    )

    pkl_path = find_existing_file(
        MODEL_DIR,
        pkl_candidates,
    )

    if model_name in DEEP_MODELS:
        if keras_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model Keras {model_name}. "
                f"Đã thử: {keras_candidates}"
            )

        model = tf.keras.models.load_model(
            keras_path,
            compile=False,
        )

        print(f"[INFO] Loaded {model_name}: {keras_path}")
        return model, "keras"

    if pkl_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy model sklearn {model_name}. "
            f"Đã thử: {pkl_candidates}"
        )

    try:
        model = load_pickle(pkl_path)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được model {pkl_path.name}: {exc}"
        ) from exc

    print(f"[INFO] Loaded {model_name}: {pkl_path}")
    return model, "sklearn"

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
            "Model sklearn không có classes_; "
            "không thể xác minh thứ tự xác suất."
        )

    model_classes = np.asarray(model.classes_)

    try:
        class_ids = model_classes.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"model.classes_={model_classes.tolist()} không phải "
            "mã lớp nguyên 0..5. Cần mapping đúng từ pipeline."
        ) from exc

    if sorted(class_ids.tolist()) != list(
        range(N_CLASSES_EXPECTED)
    ):
        raise ValueError(
            f"model.classes_={model_classes.tolist()} không phải "
            "đủ mã lớp 0..5. Dừng để tránh giải thích sai."
        )

    reordered = np.zeros_like(probabilities, dtype=float)

    for source_col, class_id in enumerate(class_ids):
        reordered[:, class_id] = probabilities[:, source_col]

    return reordered

def predict_probabilities(
    model,
    X,
    model_type,
    model_name,
):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"predict_probabilities cần X 2D, nhận {X.shape}."
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

    # Keras không lưu classes_ để kiểm tra thứ tự output.
    # Phải bảo đảm lớp softmax theo đúng thứ tự y_train khi train.
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
            "Tổng xác suất không gần 1. "
            "Kiểm tra đầu ra model có phải softmax/probability không."
        )

    return probabilities

# ============================================================
# 7. FIXED EXPLANATION AND BACKGROUND INDICES
# ============================================================

def stratified_indices(y, n_total, seed):
    y = np.asarray(y)
    n_total = min(int(n_total), len(y))

    if n_total <= 0:
        raise ValueError("n_total phải lớn hơn 0.")

    if n_total == len(y):
        return np.arange(len(y), dtype=int)

    classes = np.unique(y)

    if n_total < len(classes):
        raise ValueError(
            "Số mẫu nhỏ hơn số lớp hiện diện."
        )

    rng = np.random.default_rng(seed)

    groups = {
        int(c): rng.permutation(
            np.flatnonzero(y == c)
        )
        for c in classes
    }

    allocation = {int(c): 0 for c in classes}
    remaining = n_total

    # Phân bổ luân phiên để cân bằng lớp.
    while remaining > 0:
        active = [
            int(c)
            for c in classes
            if allocation[int(c)] < len(groups[int(c)])
        ]

        if not active:
            break

        for c in rng.permutation(active):
            allocation[int(c)] += 1
            remaining -= 1

            if remaining == 0:
                break

    selected = [
        groups[c][:allocation[c]]
        for c in allocation
        if allocation[c] > 0
    ]

    return np.sort(
        np.concatenate(selected).astype(int)
    )

def make_lime_indices(y, max_per_class, seed):
    rng = np.random.default_rng(seed)
    result = {}

    for class_id in range(N_CLASSES_EXPECTED):
        candidates = np.flatnonzero(y == class_id)
        n = min(max_per_class, len(candidates))

        if n == 0:
            result[class_id] = np.asarray([], dtype=int)
        else:
            result[class_id] = np.sort(
                rng.choice(
                    candidates,
                    size=n,
                    replace=False,
                )
            ).astype(int)

    return result

def make_or_load_fixed_indices(X_test, y_test):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    data_hash = dataset_fingerprint(X_test, y_test)

    config = (
        f"nslkdd-sampling-v2|global={N_GLOBAL}|"
        f"background={N_BACKGROUND}|"
        f"lime={N_LIME_PER_CLASS}|seed={RANDOM_SEED}"
    )

    fingerprint = hashlib.sha256(
        f"{data_hash}|{config}".encode("utf-8")
    ).hexdigest()

    tag = hashlib.sha256(
        config.encode("utf-8")
    ).hexdigest()[:12]

    explain_path = (
        SAVE_DIR / f"NSLKDD_explain_indices_{tag}.npy"
    )
    background_path = (
        SAVE_DIR / f"NSLKDD_background_indices_{tag}.npy"
    )
    lime_path = (
        SAVE_DIR / f"NSLKDD_lime_indices_{tag}.npz"
    )
    fingerprint_path = (
        SAVE_DIR / f"NSLKDD_sampling_fingerprint_{tag}.txt"
    )

    old_fingerprint = (
        fingerprint_path.read_text(encoding="utf-8").strip()
        if fingerprint_path.exists()
        else None
    )

    reuse = old_fingerprint == fingerprint

    def valid_indices(indices):
        return (
            indices.ndim == 1
            and len(indices) > 0
            and len(np.unique(indices)) == len(indices)
            and np.all(indices >= 0)
            and np.all(indices < len(y_test))
        )

    def load_cached_indices(path, expected_n, name):
        if not (reuse and path.exists()):
            return None
        try:
            candidate = np.load(path, allow_pickle=False)
            if (
                valid_indices(candidate)
                and len(candidate) == expected_n
            ):
                print(f"[INFO] Reusing {name} indices.")
                return candidate.astype(int)
            print(f"[WARN] Rebuilding invalid {name} index cache.")
        except Exception as exc:
            print(
                f"[WARN] Rebuilding unreadable {name} index cache: "
                f"{type(exc).__name__}: {exc}"
            )
        return None

    explain_idx = load_cached_indices(
        explain_path,
        min(N_GLOBAL, len(y_test)),
        "explain",
    )
    background_idx = load_cached_indices(
        background_path,
        min(N_BACKGROUND, len(y_test)),
        "background",
    )
    lime_indices = None

    if reuse and lime_path.exists():
        try:
            with np.load(lime_path) as cached:
                candidate_map = {
                    int(key): cached[key].astype(int)
                    for key in cached.files
                }

            valid_map = set(candidate_map) == set(
                range(N_CLASSES_EXPECTED)
            )

            for c in range(N_CLASSES_EXPECTED):
                arr = candidate_map.get(c)
                expected_n = min(
                    N_LIME_PER_CLASS,
                    int(np.sum(y_test == c)),
                )

                if arr is None or arr.ndim != 1 or len(arr) != expected_n:
                    valid_map = False
                    break

                if len(arr) > 0:
                    # Validate bounds and uniqueness before indexing y_test.
                    if not valid_indices(arr):
                        valid_map = False
                        break

                    if not np.all(y_test[arr] == c):
                        valid_map = False
                        break

            if valid_map:
                lime_indices = candidate_map
        except Exception as exc:
            print(
                f"[WARN] Bỏ qua LIME cache không hợp lệ: "
                f"{type(exc).__name__}: {exc}"
            )

    if explain_idx is None:
        explain_idx = stratified_indices(
            y_test,
            min(N_GLOBAL, len(y_test)),
            RANDOM_SEED,
        )
        np.save(explain_path, explain_idx)
    else:
        print("[INFO] Reusing explain indices.")

    if background_idx is None:
        background_idx = stratified_indices(
            y_test,
            min(N_BACKGROUND, len(y_test)),
            RANDOM_SEED + 1,
        )
        np.save(background_path, background_idx)
    else:
        print("[INFO] Reusing background indices.")

    if lime_indices is None:
        lime_indices = make_lime_indices(
            y_test,
            N_LIME_PER_CLASS,
            RANDOM_SEED + 2,
        )

        np.savez(
            lime_path,
            **{
                str(c): arr
                for c, arr in lime_indices.items()
            },
        )
    else:
        print("[INFO] Reusing LIME indices.")

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

    for class_id, indices in lime_indices.items():
        for idx in indices:
            manifest_rows.append([
                "lime",
                int(idx),
                int(class_id),
            ])

    save_csv(
        SAVE_DIR / "NSLKDD_sample_manifest.csv",
        [
            "sample_role",
            "test_row_index",
            "true_class_index",
        ],
        manifest_rows,
    )

    count_rows = []

    for class_id in range(N_CLASSES_EXPECTED):
        count_rows.append([
            class_id,
            CLASS_NAMES_FALLBACK[class_id],
            int(np.sum(y_test == class_id)),
            int(np.sum(y_test[explain_idx] == class_id)),
            int(np.sum(y_test[background_idx] == class_id)),
            len(lime_indices.get(class_id, [])),
        ])

    save_csv(
        SAVE_DIR / "NSLKDD_sample_counts.csv",
        [
            "class_index",
            "class_name",
            "test_support",
            "n_explain_samples",
            "n_background_samples",
            "n_lime_samples",
        ],
        count_rows,
    )

    print(f"[INFO] Explain samples: {len(explain_idx)}")
    print(f"[INFO] Background samples: {len(background_idx)}")

    return explain_idx, background_idx, lime_indices

# ============================================================
# 8. SHAP NORMALIZATION
# ============================================================

def normalize_one_class(values, n_samples, n_features):
    arr = np.asarray(values)

    # Bỏ các chiều singleton ngoài trục mẫu.
    while arr.ndim > 2:
        axes = [
            axis
            for axis in range(1, arr.ndim)
            if arr.shape[axis] == 1
        ]

        if not axes:
            break

        arr = np.squeeze(arr, axis=axes[-1])

    if arr.ndim == 1 and n_features == 1:
        arr = arr.reshape(-1, 1)

    if arr.shape == (n_samples, n_features):
        return arr

    if arr.shape == (n_features, n_samples):
        return arr.T

    raise ValueError(
        f"Không chuẩn hóa được SHAP class array: {arr.shape}"
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
                f"SHAP list có {len(raw_values)} lớp, "
                f"mong đợi {n_classes}."
            )

        result = np.stack(
            [
                normalize_one_class(
                    values,
                    n_samples,
                    n_features,
                )
                for values in raw_values
            ],
            axis=-1,
        )

    else:
        arr = np.asarray(raw_values)

        while arr.ndim > 3:
            axes = [
                axis
                for axis in range(1, arr.ndim)
                if arr.shape[axis] == 1
            ]

            if not axes:
                raise ValueError(
                    f"SHAP có shape không hỗ trợ: {arr.shape}"
                )

            arr = np.squeeze(arr, axis=axes[-1])

        if arr.ndim == 2:
            if arr.shape != (n_samples, n_features):
                raise ValueError(
                    f"SHAP 2D không khớp: {arr.shape}"
                )

            if n_classes != 1:
                raise ValueError(
                    "SHAP trả một đầu ra nhưng model có nhiều lớp."
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
                n_classes,
                n_samples,
                n_features,
            ):
                result = np.transpose(arr, (1, 2, 0))

            elif arr.shape == (
                n_samples,
                n_classes,
                n_features,
            ):
                result = np.transpose(arr, (0, 2, 1))

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
            f"SHAP shape={result.shape}; mong đợi {expected}."
        )

    if not np.isfinite(result).all():
        raise ValueError("SHAP values chứa NaN hoặc Inf.")

    return result.astype(float)

def reorder_tree_shap_classes(raw_values, model):
    if not hasattr(model, "classes_"):
        raise ValueError(
            "TreeSHAP không xác minh được thứ tự lớp."
        )

    try:
        class_ids = np.asarray(model.classes_).astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "TreeSHAP classes_ không phải mã lớp nguyên."
        ) from exc

    if sorted(class_ids.tolist()) != list(
        range(N_CLASSES_EXPECTED)
    ):
        raise ValueError(
            f"TreeSHAP classes_ không phải 0..5: "
            f"{class_ids.tolist()}"
        )

    if isinstance(raw_values, list):
        if len(raw_values) != N_CLASSES_EXPECTED:
            raise ValueError(
                "TreeSHAP không trả đủ 6 lớp."
            )

        reordered = [None] * N_CLASSES_EXPECTED

        for source_pos, class_id in enumerate(class_ids):
            reordered[class_id] = raw_values[source_pos]

        return reordered

    arr = np.asarray(raw_values)

    # Dạng (N, F, C)
    if arr.ndim == 3 and arr.shape[-1] == N_CLASSES_EXPECTED:
        return arr[:, :, np.argsort(class_ids)]

    # Dạng (N, C, F)
    if arr.ndim == 3 and arr.shape[1] == N_CLASSES_EXPECTED:
        return arr[:, np.argsort(class_ids), :]

    return raw_values

# ============================================================
# 9. IMPORTANCE REPORTS AND PLOTS
# ============================================================

def save_importance_reports(
    method,
    class_names,
    feature_names,
    class_importance,
    class_explain_counts,
    test_support,
    output_dir,
):
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
                float(value) if np.isfinite(value) else "NaN",
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

    valid_classes = [
        c
        for c in range(n_classes)
        if class_explain_counts[c] > 0
        and np.isfinite(class_importance[c]).all()
    ]

    if valid_classes:
        balanced = np.mean(
            class_importance[valid_classes],
            axis=0,
        )
    else:
        balanced = np.full(n_features, np.nan)

    supported_classes = [
        c for c in range(n_classes)
        if test_support[c] > 0
    ]

    # Không gán importance bằng 0 cho lớp chưa được giải thích.
    all_supported_valid = (
        bool(supported_classes)
        and all(c in valid_classes for c in supported_classes)
    )

    if all_supported_valid:
        support_weights = test_support[supported_classes].astype(float)
        support_weights /= support_weights.sum()

        natural = np.sum(
            class_importance[supported_classes]
            * support_weights[:, None],
            axis=0,
        )
    else:
        natural = np.full(n_features, np.nan)
        print(
            f"[WARN] {method}: không tính được natural importance "
            "vì thiếu importance hợp lệ ở một lớp có mặt trong test."
        )

    aggregate_rows = []

    for j, feature in enumerate(feature_names):
        aggregate_rows.append([
            feature,
            float(natural[j]) if np.isfinite(natural[j]) else "NaN",
            float(balanced[j]) if np.isfinite(balanced[j]) else "NaN",
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

def save_class_plot(
    values,
    X_plot,
    feature_names,
    class_name,
    method,
    output_dir,
):
    if len(values) == 0:
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
            f"{method} - {class_name}",
            fontsize=11,
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / f"{method}_{safe_filename(class_name)}.png",
            dpi=160,
            bbox_inches="tight",
        )

    finally:
        close_plot()

def save_bar_plot(
    importances,
    feature_names,
    output_path,
    title,
    top_n=20,
):
    importances = np.asarray(importances, dtype=float)

    if not np.isfinite(importances).all():
        return

    order = np.argsort(importances)[-top_n:]

    try:
        plt.figure(figsize=(10, 7))

        plt.barh(
            [feature_names[i] for i in order],
            importances[order],
        )

        plt.xlabel("Mean absolute importance")
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
# 10. COMPUTE SHAP
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
        if model_type != "sklearn":
            raise ValueError("TreeSHAP chỉ dùng cho model sklearn.")

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
                np.asarray(data, dtype=np.float32),
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
        if model_type != "keras":
            raise ValueError(
                "GradientSHAP chỉ dùng cho model Keras."
            )

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

        raw_values = explainer.shap_values(explain_input)

    else:
        raise ValueError(f"SHAP method không hỗ trợ: {method}")

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
    class_names,
    test_support,
    output_dir,
):
    print(f"[SHAP] Đang chạy {method}...")

    X_explain = X_test[explain_idx]
    y_explain = y_test[explain_idx]
    X_background = X_test[background_idx]

    # Xác minh output xác suất trước khi giải thích.
    predict_probabilities(
        model,
        X_explain[:min(5, len(X_explain))],
        model_type,
        model_name,
    )

    shap_values = calculate_shap(
        method,
        model,
        model_type,
        model_name,
        X_background,
        X_explain,
    )

    n_features = len(feature_names)

    class_importance = np.full(
        (N_CLASSES_EXPECTED, n_features),
        np.nan,
    )

    class_counts = np.zeros(N_CLASSES_EXPECTED, dtype=int)

    for class_idx, class_name in enumerate(class_names):
        mask = y_explain == class_idx
        n = int(mask.sum())
        class_counts[class_idx] = n

        if n == 0:
            print(
                f"[WARN] {method}: không có mẫu thật thuộc "
                f"lớp {class_name} trong tập explain."
            )
            continue

        # Giải thích đầu ra của lớp c trên mẫu có nhãn thật c.
        class_sv = shap_values[mask, :, class_idx]
        class_X = X_explain[mask]

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
            output_dir,
        )

    save_importance_reports(
        method=method,
        class_names=class_names,
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print(f"[SHAP] Hoàn tất {method}.")

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
    class_names,
    test_support,
    output_dir,
):
    from lime.lime_tabular import LimeTabularExplainer

    print("[LIME] Đang chạy...")

    lime_background = (
        X_train if X_train is not None else X_test
    )

    explainer = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=class_names,
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
    all_rows = []

    for class_idx, class_name in enumerate(class_names):
        indices = lime_indices.get(
            class_idx,
            np.asarray([], dtype=int),
        )

        if len(indices) == 0:
            print(f"[WARN] LIME: lớp {class_name} không có mẫu.")
            continue

        # Dùng mẫu từ tập chỉ số cố định, tối đa 50 mẫu/lớp.
        indices = indices[:N_LIME_PER_CLASS]
        sample_weights = []

        for sample_no, sample_idx in enumerate(indices, start=1):
            # LIME tạo perturbation để ước lượng hành vi cục bộ.
            # Đây không phải bằng chứng về quan hệ nhân quả.
            explanation = explainer.explain_instance(
                data_row=X_test[sample_idx],
                predict_fn=predict_fn,
                labels=(class_idx,),
                num_features=len(feature_names),
                num_samples=LIME_NUM_SAMPLES,
            )

            local_map = explanation.as_map().get(class_idx, [])

            signed_weights = np.zeros(
                len(feature_names),
                dtype=float,
            )

            for feature_idx, weight in local_map:
                if 0 <= feature_idx < len(feature_names):
                    signed_weights[feature_idx] = float(weight)

            sample_weights.append(np.abs(signed_weights))

            for feature_idx, feature_name in enumerate(feature_names):
                all_rows.append([
                    class_name,
                    int(sample_idx),
                    sample_no,
                    int(y_test[sample_idx]),
                    feature_name,
                    float(signed_weights[feature_idx]),
                    float(abs(signed_weights[feature_idx])),
                ])

        if sample_weights:
            class_importance[class_idx] = np.mean(
                np.vstack(sample_weights),
                axis=0,
            )

        class_counts[class_idx] = len(indices)

        save_bar_plot(
            class_importance[class_idx],
            feature_names,
            output_dir / f"LIME_{safe_filename(class_name)}.png",
            f"LIME - {model_name} - {class_name}",
        )

        print(
            f"[LIME] {class_name}: {len(indices)} mẫu; "
            f"test support={int(test_support[class_idx])}"
        )

    save_csv(
        output_dir / "LIME_local_weights.csv",
        [
            "class_name",
            "test_row_index",
            "sample_number_in_class",
            "true_class_index",
            "feature",
            "lime_weight_signed",
            "lime_weight_absolute",
        ],
        all_rows,
    )

    save_importance_reports(
        method="LIME",
        class_names=class_names,
        feature_names=feature_names,
        class_importance=class_importance,
        class_explain_counts=class_counts,
        test_support=test_support,
        output_dir=output_dir,
    )

    print("[LIME] Hoàn tất.")

# ============================================================
# 12. MAIN
# ============================================================

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NSL-KDD MULTICLASS GLOBAL XAI")
    print("=" * 80)
    print(f"Data directory : {PREPROCESSED_DIR.resolve()}")
    print(f"Model directory: {MODEL_DIR.resolve()}")
    print(f"Output directory: {SAVE_DIR.resolve()}")

    (
        X_test,
        y_test,
        feature_names,
        class_names,
        test_support,
    ) = load_dataset()

    X_train = load_X_train_if_available(
        n_features=X_test.shape[1],
    )

    explain_idx, background_idx, lime_indices = (
        make_or_load_fixed_indices(
            X_test,
            y_test,
        )
    )

    # Ghi mapping lớp cùng support thật của test set.
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
                class_names[class_idx],
                int(test_support[class_idx]),
            ]
            for class_idx in range(N_CLASSES_EXPECTED)
        ],
    )

    print("\nConfiguration:")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Explain samples: {len(explain_idx)}")
    print(f"  Background samples: {len(background_idx)}")
    print(f"  LIME max samples/class: {N_LIME_PER_CLASS}")
    print(f"  KernelSHAP nsamples: {KERNEL_NSAMPLES}")
    print("  Model performance metrics: NOT RUN")

    for model_name in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        try:
            model, model_type = load_model(model_name)

            # Keep artifacts separate so later models/methods cannot
            # overwrite reports generated earlier in this run.
            model_output_dir = SAVE_DIR / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Kiểm tra model có trả đúng 6 xác suất hay không.
            check_idx = explain_idx[:min(5, len(explain_idx))]

            predict_probabilities(
                model,
                X_test[check_idx],
                model_type,
                model_name,
            )

            # TreeSHAP cho model cây.
            # KernelSHAP cho sklearn models để có thể áp dụng
            # cho xác suất đầu ra nhiều lớp.
            # GradientSHAP cho mạng neural.
            methods = []

            if model_type == "sklearn" and model_name in TREE_MODELS:
                methods.append("TreeSHAP")

            if model_type == "sklearn":
                methods.append("KernelSHAP")

            if model_type == "keras":
                methods.append("GradientSHAP")

            methods.append("LIME")

            for method in methods:
                try:
                    method_output_dir = model_output_dir / method
                    method_output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

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
                            class_names=class_names,
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
                f"[MODEL ERROR] {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 80)
    print("HOÀN TẤT NSL-KDD GLOBAL XAI")
    print(f"Kết quả: {SAVE_DIR.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
