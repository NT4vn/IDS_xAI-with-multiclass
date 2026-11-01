
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import warnings
from pathlib import Path

import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ================= PATH =================
PREPROCESSED_DIR = Path(
    r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped"
)
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("XAI_Results3") / "local" / "CIC17"

DATASET_NAME = "CIC17"
DATASET_PREFIX = "cic17"

SAMPLES_PER_CLASS = 2
BACKGROUND_SIZE = 50
LIME_BACKGROUND_SIZE = 2000
LIME_NUM_SAMPLES = 1000
KERNEL_NSAMPLES = 100

# Thứ tự phải khớp mapping class trong epoch.py.
CLASS_NAMES = [
    "Botnet / Infiltration",
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
TREE_MODELS = {"RandomForest", "LightGBM"}

# ================= LOAD FEATURE NAMES =================
def load_feature_names():
    path = (
        PREPROCESSED_DIR
        / f"preprocessed_multi_{DATASET_PREFIX}_feature_names.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy feature names: {path}\n"
            "Kiểm tra metadata được tạo lúc preprocessing."
        )

    with open(path, "rb") as f:
        names = pickle.load(f)

    if isinstance(names, dict):
        for key in ("feature_names", "selected_features", "columns"):
            if key in names:
                names = names[key]
                break
        else:
            raise ValueError(
                "Không xác định được danh sách feature trong metadata."
            )

    names = [str(name) for name in list(names)]

    if not names:
        raise ValueError("Feature names đang rỗng.")

    return names

# ================= LOAD DATA =================
def load_data():
    x_path = (
        PREPROCESSED_DIR
        / f"preprocessed_multi_{DATASET_PREFIX}_X_test.npy"
    )
    y_path = (
        PREPROCESSED_DIR
        / f"preprocessed_multi_{DATASET_PREFIX}_y_test.npy"
    )

    if not x_path.exists():
        raise FileNotFoundError(f"Không tìm thấy X_test: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Không tìm thấy y_test: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(
            f"X_test cần có dạng 2D, hiện tại: {X.shape}"
        )

    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=1)

    y = np.asarray(y).reshape(-1).astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"Số mẫu không khớp: X={len(X)}, y={len(y)}"
        )

    if len(X) == 0:
        raise ValueError("X_test đang rỗng.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X_test chứa NaN hoặc Inf.")

    if np.any(y < 0) or np.any(y >= len(CLASS_NAMES)):
        raise ValueError(
            f"Nhãn y phải nằm trong khoảng "
            f"0 đến {len(CLASS_NAMES) - 1}. "
            "Kiểm tra mapping class trong epoch.py."
        )

    feature_names = load_feature_names()

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X_test có {X.shape[1]} feature nhưng metadata có "
            f"{len(feature_names)} tên. Không tự cắt danh sách."
        )

    print(f"Data loaded: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_names)}")

    return X, y, feature_names

def select_samples_by_class(y, samples_per_class=SAMPLES_PER_CLASS, seed=42):
    """Lấy tối đa số mẫu cấu hình cho mỗi lớp từ toàn bộ tập test."""
    rng = np.random.default_rng(seed)
    selected = {}
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(np.asarray(y) == class_id)
        if len(candidates) == 0:
            continue
        count = min(samples_per_class, len(candidates))
        selected[class_id] = np.sort(
            rng.choice(candidates, size=count, replace=False)
        ).astype(int).tolist()
    return selected

def load_training_background():
    """Lấy nền LIME/SHAP từ train đã tiền xử lý, theo đúng không gian feature."""
    x_path = PREPROCESSED_DIR / f"preprocessed_multi_{DATASET_PREFIX}_X_train.npy"
    y_path = PREPROCESSED_DIR / f"preprocessed_multi_{DATASET_PREFIX}_y_train.npy"
    X_train = np.load(x_path, mmap_mode="r")
    y_train = np.asarray(np.load(y_path)).reshape(-1)
    rng = np.random.default_rng(42)
    count = min(LIME_BACKGROUND_SIZE, len(y_train))
    indices = np.sort(rng.choice(len(y_train), size=count, replace=False))
    return np.asarray(X_train[indices], dtype=np.float32)

def write_prediction_report(path, selected_by_class, sample_indices, y_true, probabilities):
    positions = {int(sample_id): pos for pos, sample_id in enumerate(sample_indices)}
    predicted = np.argmax(probabilities, axis=1)
    with open(path, "w", encoding="utf-8") as f:
        for true_class, indices in selected_by_class.items():
            f.write(f"\n=== Lớp thật: {CLASS_NAMES[true_class]} ===\n")
            for sample_id in indices:
                pos = positions[int(sample_id)]
                pred = int(predicted[pos])
                f.write(
                    f"Sample {sample_id}: true={true_class} ({CLASS_NAMES[true_class]}), "
                    f"predicted={pred} ({CLASS_NAMES[pred]}), "
                    f"correct={pred == true_class}, "
                    f"probability={probabilities[pos, pred]:.6f}\n"
                )
                f.write(
                    "  Probabilities: "
                    + ", ".join(
                        f"{CLASS_NAMES[c]}={probabilities[pos, c]:.6f}"
                        for c in range(len(CLASS_NAMES))
                    )
                    + "\n"
                )

# ================= MODEL INPUT =================
def prepare_model_input(X, model_name):
    X = np.asarray(X)

    if model_name == "CNN":
        return X.reshape(X.shape[0], X.shape[1], 1)

    if model_name == "LSTM":
        return X.reshape(X.shape[0], 1, X.shape[1])

    return X

# ================= LOAD MODEL =================
def load_model(model_name):
    if model_name in DEEP_MODELS:
        model_path = MODEL_DIR / f"{DATASET_NAME}_{model_name}.keras"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}"
            )

        return tf.keras.models.load_model(model_path)

    model_path = MODEL_DIR / f"{DATASET_NAME}_{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}"
        )

    with open(model_path, "rb") as f:
        return pickle.load(f)

# ================= PREDICTION =================
def predict_probabilities(data, model, model_name):
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(
            f"Đầu vào dự đoán phải là 2D, nhận được {data.shape}"
        )

    if model_name in DEEP_MODELS:
        model_input = prepare_model_input(data, model_name)
        output = model.predict(model_input, verbose=0)

    elif hasattr(model, "predict_proba"):
        output = model.predict_proba(data)

        # Đảm bảo thứ tự cột khớp class ID 0..7.
        # Áp dụng khi sklearn lưu model.classes_.
        if hasattr(model, "classes_"):
            classes = np.asarray(model.classes_).reshape(-1)

            if (
                len(classes) != len(output[0])
                or not np.issubdtype(classes.dtype, np.integer)
                or not np.array_equal(
                    np.sort(classes),
                    np.arange(len(CLASS_NAMES))
                )
            ):
                raise ValueError(
                    f"{model_name}: model.classes_={classes} không "
                    "khớp class ID 0..7. Cần kiểm tra mapping nhãn."
                )

            aligned = np.zeros(
                (len(output), len(CLASS_NAMES)),
                dtype=float
            )
            aligned[:, classes.astype(int)] = output
            output = aligned

    else:
        raise TypeError(
            f"{model_name} không hỗ trợ predict_proba(). "
            "Không dùng nhãn dự đoán one-hot thay xác suất cho XAI."
        )

    output = np.asarray(output)

    if output.ndim != 2 or output.shape != (
        len(data), len(CLASS_NAMES)
    ):
        raise ValueError(
            f"{model_name} trả về shape {output.shape}; "
            f"mong đợi ({len(data)}, {len(CLASS_NAMES)})."
        )

    if not np.all(np.isfinite(output)):
        raise ValueError(
            f"{model_name} trả về xác suất chứa NaN hoặc Inf."
        )

    return output

# ================= NORMALIZE SHAP =================
def normalize_shap_values(
    raw_values,
    n_samples,
    n_features,
    n_classes
):
    """
    Chuẩn hóa SHAP đa lớp thành:
        result[class_id] -> (n_samples, n_features)

    Hỗ trợ:
    - list gồm SHAP array theo từng class
    - ndarray (samples, features, classes)
    - ndarray (samples, classes, features)
    - đầu ra CNN/LSTM có chiều singleton
    """

    if isinstance(raw_values, shap.Explanation):
        raw_values = raw_values.values

    def normalize_single_class(item):
        arr = np.asarray(item)

        if arr.shape[0] != n_samples:
            raise ValueError(
                f"SHAP sample shape không hợp lệ: {arr.shape}"
            )

        # Bỏ các chiều singleton của CNN/LSTM.
        arr = np.squeeze(arr)

        if arr.shape != (n_samples, n_features):
            raise ValueError(
                f"SHAP theo class có shape {arr.shape}; "
                f"mong đợi {(n_samples, n_features)}."
            )

        return arr

    if isinstance(raw_values, list):
        class_values = [
            normalize_single_class(item)
            for item in raw_values
        ]

        if len(class_values) != n_classes:
            raise ValueError(
                f"SHAP trả về {len(class_values)} class, "
                f"mong đợi {n_classes}."
            )

        return class_values

    arr = np.asarray(raw_values)

    if arr.shape[0] != n_samples:
        raise ValueError(
            f"Số mẫu SHAP không khớp: {arr.shape}"
        )

    # Loại các chiều singleton không phải trục samples.
    # Ví dụ CNN: (N, F, 1, C) -> (N, F, C).
    # Ví dụ LSTM: (N, 1, F, C) -> (N, F, C).
    while arr.ndim > 3:
        singleton_axes = [
            axis for axis in range(1, arr.ndim)
            if arr.shape[axis] == 1
        ]

        if not singleton_axes:
            break

        arr = np.squeeze(arr, axis=singleton_axes[0])

    if arr.ndim == 3:
        if arr.shape == (
            n_samples, n_features, n_classes
        ):
            return [
                arr[:, :, c] for c in range(n_classes)
            ]

        if arr.shape == (
            n_samples, n_classes, n_features
        ):
            return [
                arr[:, c, :] for c in range(n_classes)
            ]

    if arr.ndim == 2 and n_classes == 1:
        if arr.shape == (n_samples, n_features):
            return [arr]

    raise ValueError(
        f"Không chuẩn hóa được SHAP output: {arr.shape}. "
        f"Samples={n_samples}, features={n_features}, "
        f"classes={n_classes}."
    )

# ================= BASE VALUES =================
def get_expected_values(
    explainer,
    background_probs,
    n_classes
):
    """
    Lấy baseline tương ứng từng class.
    Với GradientExplainer, dùng mean probability của background.
    """

    expected = getattr(explainer, "expected_value", None)

    if expected is not None:
        expected = np.asarray(expected).reshape(-1)

        if len(expected) == n_classes:
            return expected.astype(float)

        if len(expected) == 1 and n_classes == 1:
            return expected.astype(float)

    # Fallback gần đúng; không phải base value SHAP chính xác
    # cho mọi loại explainer.
    return np.mean(background_probs, axis=0).astype(float)

# ================= SHAP EXPLANATION =================
def calculate_shap(model, model_name, X, X_background):
    n_samples, n_features = X.shape
    n_classes = len(CLASS_NAMES)

    X_background = X_background[:min(BACKGROUND_SIZE, len(X_background))]

    if model_name == "LightGBM":
        # Explain predicted probabilities so SHAP values use a readable
        # 0..1 scale instead of LightGBM's potentially huge raw margins.
        def predict_fn(data):
            return predict_probabilities(
                np.asarray(data), model, model_name
            )

        explainer = shap.KernelExplainer(predict_fn, X_background)
        raw_values = explainer.shap_values(
            X, nsamples=KERNEL_NSAMPLES
        )

    elif model_name in TREE_MODELS:
        # RandomForest / LightGBM
        explainer = shap.TreeExplainer(
            model,
            data=X_background,
            # Raw output is supported for both sklearn forests and LightGBM.
            # Some LightGBM wrappers do not expose objective metadata needed
            # by TreeExplainer to transform contributions to probabilities.
            model_output="raw",
            feature_perturbation="interventional",
        )
        # LightGBM multiclass can fail SHAP's numerical additivity assertion
        # despite correctly aligned inputs (large raw margins amplify rounding).
        raw_values = explainer.shap_values(
            X, check_additivity=(model_name != "LightGBM")
        )

    elif model_name in DEEP_MODELS:
        # DNN / CNN / LSTM
        background_input = prepare_model_input(
            X_background, model_name
        )
        X_input = prepare_model_input(X, model_name)

        explainer = shap.GradientExplainer(
            model,
            background_input
        )
        raw_values = explainer.shap_values(X_input)

    else:
        # AdaBoost / MLP:
        # KernelExplainer hỗ trợ model bất kỳ có hàm xác suất.
        def predict_fn(data):
            return predict_probabilities(
                np.asarray(data), model, model_name
            )

        explainer = shap.KernelExplainer(
            predict_fn,
            X_background
        )

        raw_values = explainer.shap_values(
            X,
            nsamples=KERNEL_NSAMPLES
        )

    values_by_class = normalize_shap_values(
        raw_values,
        n_samples,
        n_features,
        n_classes
    )

    background_probs = predict_probabilities(
        X_background, model, model_name
    )

    expected_by_class = get_expected_values(
        explainer,
        background_probs,
        n_classes
    )

    return values_by_class, expected_by_class

# ================= SAVE SHAP PLOTS =================
def save_shap_plots(
    idx,
    predicted_class,
    X,
    feature_names,
    shap_values,
    expected_value,
    model_name,
    save_path,
    sample_id=None,
):
    class_name = CLASS_NAMES[predicted_class]

    sample_values = np.asarray(
        shap_values[idx]
    ).reshape(-1)

    sample_data = X[idx]

    if len(sample_values) != len(feature_names):
        raise ValueError(
            f"SHAP có {len(sample_values)} giá trị nhưng "
            f"metadata có {len(feature_names)} feature."
        )

    exp = shap.Explanation(
        values=sample_values,
        base_values=float(expected_value),
        data=sample_data,
        feature_names=feature_names
    )

    display_idx = idx if sample_id is None else int(sample_id)
    suffix = f"sample_{display_idx}_class_{predicted_class}"

    # ---------- WATERFALL ----------
    shap.plots.waterfall(
        exp,
        max_display=20,
        show=False
    )

    plt.gcf().set_size_inches(10, 7)
    plt.title(
        f"SHAP Waterfall - {model_name} - "
        f"Sample {idx} - {class_name}"
    )
    plt.savefig(
        save_path / f"SHAP_Waterfall_{suffix}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close("all")

    # ---------- FORCE PLOT HTML ----------
    force = shap.force_plot(
        float(expected_value),
        sample_values,
        sample_data,
        feature_names=feature_names,
        matplotlib=False
    )

    shap.save_html(
        str(save_path / f"SHAP_Force_{suffix}.html"),
        force
    )

    # ---------- DECISION PLOT ----------
    shap.decision_plot(
        float(expected_value),
        sample_values,
        sample_data,
        feature_names=feature_names,
        show=False
    )

    plt.gcf().set_size_inches(10, 6)
    plt.title(
        f"SHAP Decision - {model_name} - "
        f"Sample {idx} - {class_name}"
    )
    plt.savefig(
        save_path / f"SHAP_Decision_{suffix}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close("all")

# ================= LIME EXPLANATION =================
def run_lime(
    model,
    model_name,
    X,
    feature_names,
    sample_indices,
    sample_ids,
    predicted_classes,
    lime_background,
    save_path
):
    from lime.lime_tabular import LimeTabularExplainer

    lime_explainer = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=42
    )

    def predict_fn(data):
        return predict_probabilities(
            np.asarray(data), model, model_name
        )

    for pos, idx in enumerate(sample_indices):
        class_idx = int(predicted_classes[pos])

        exp = lime_explainer.explain_instance(
            data_row=X[pos],
            predict_fn=predict_fn,
            labels=[class_idx],
            num_features=min(20, X.shape[1]),
            num_samples=LIME_NUM_SAMPLES
        )

        suffix = f"sample_{int(sample_ids[pos])}_class_{class_idx}"

        # HTML
        exp.save_to_file(
            str(save_path / f"LIME_{suffix}.html")
        )

        # PNG
        fig = exp.as_pyplot_figure(label=class_idx)
        fig.set_size_inches(10, 6)
        fig.suptitle(
            f"LIME - {model_name} - Sample {idx} - "
            f"{CLASS_NAMES[class_idx]}"
        )
        fig.tight_layout()

        fig.savefig(
            save_path / f"LIME_{suffix}.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)

        print(
            f"      ✓ LIME sample {idx}: "
            f"{CLASS_NAMES[class_idx]}"
        )

# ================= MAIN =================
def run_local_explanation():
    X_all, y_all, feature_names = load_data()
    selected_by_class = select_samples_by_class(y_all)
    sample_indices = [
        idx for indices in selected_by_class.values() for idx in indices
    ]
    if not sample_indices:
        raise ValueError("Không tìm thấy mẫu test thuộc các lớp cần giải thích.")
    positions = np.asarray(sample_indices, dtype=int)
    X = np.asarray(X_all[positions], dtype=np.float32)
    y = np.asarray(y_all[positions], dtype=int)
    lime_background = load_training_background()
    shap_background = lime_background[:min(BACKGROUND_SIZE, len(lime_background))]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("LOCAL XAI - CIC17 MULTICLASS")
    print("SHAP + LIME | HTML + PNG")
    print(f"Samples selected: {len(sample_indices)} / {len(X_all)}")
    print(f"Features: {X_all.shape[1]}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print(f"Samples per true class: {SAMPLES_PER_CLASS}")
    print(f"Sample indices grouped by true class: {selected_by_class}")
    print("=" * 90)

    for model_name in MODELS:
        print(f"\n🔹 Model: {model_name}")

        try:
            model = load_model(model_name)
        except Exception as e:
            print(
                f"   ❌ Load model lỗi: "
                f"{type(e).__name__}: {e}"
            )
            continue

        save_path = SAVE_DIR / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # ================= PREDICTION =================
        try:
            probabilities = predict_probabilities(
                X, model, model_name
            )

            predicted_classes = np.argmax(
                probabilities, axis=1
            )

            write_prediction_report(
                save_path / "predictions.txt",
                selected_by_class,
                sample_indices,
                y,
                probabilities,
            )
            write_prediction_report(
                save_path / "local_predictions.txt",
                selected_by_class,
                sample_indices,
                y,
                probabilities,
            )

            print(
                f"   ✓ Prediction shape: "
                f"{probabilities.shape}"
            )

        except Exception as e:
            print(
                f"   ❌ Prediction lỗi: "
                f"{type(e).__name__}: {e}"
            )
            continue

        # ================= SHAP =================
        print("   → SHAP...")

        try:
            values_by_class, expected_by_class = calculate_shap(
                model,
                model_name,
                X,
                shap_background,
            )

            for pos, sample_id in enumerate(sample_indices):
                class_idx = int(predicted_classes[pos])

                save_shap_plots(
                    idx=pos,
                    predicted_class=class_idx,
                    X=X,
                    feature_names=feature_names,
                    shap_values=values_by_class[class_idx],
                    expected_value=expected_by_class[class_idx],
                    model_name=model_name,
                    save_path=save_path,
                    sample_id=sample_id,
                )

                print(
                    f"      ✓ SHAP sample {sample_id}: "
                    f"true={CLASS_NAMES[int(y[pos])]}, "
                    f"predicted={CLASS_NAMES[class_idx]}"
                )

        except Exception as e:
            plt.close("all")
            print(
                f"   ❌ SHAP lỗi: "
                f"{type(e).__name__}: {e}"
            )

        # ================= LIME =================
        print("   → LIME...")

        try:
            run_lime(
                model=model,
                model_name=model_name,
                X=X,
                feature_names=feature_names,
                sample_indices=list(range(len(sample_indices))),
                sample_ids=sample_indices,
                predicted_classes=predicted_classes,
                lime_background=lime_background,
                save_path=save_path
            )

        except Exception as e:
            plt.close("all")
            print(
                f"   ❌ LIME lỗi: "
                f"{type(e).__name__}: {e}"
            )

    print("\n🎯 DONE LOCAL XAI - CIC17 MULTICLASS!")
    print(f"📂 Output: {SAVE_DIR.resolve()}")

# ================= RUN =================
if __name__ == "__main__":
    run_local_explanation()
