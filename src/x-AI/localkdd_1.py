
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
PREPROCESSED_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("XAI_Results3") / "local" / "NSL-KDD"

DATASET_NAME = "NSL-KDD"
DATASET_PREFIX = "nsl-kdd"

SAMPLES_PER_CLASS = 2
BACKGROUND_SIZE = 50
LIME_BACKGROUND_SIZE = 2000
LIME_NUM_SAMPLES = 1000
KERNEL_NSAMPLES = 100

# Thứ tự phải khớp mapping class trong epoch.py.
CLASS_NAMES = [
    "DoS",
    "Normal",
    "Other",
    "Probe",
    "R2L",
    "U2R",
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
            "Hãy kiểm tra file feature names được lưu cùng pipeline "
            "tiền xử lý đa lớp."
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
                "Không xác định được danh sách feature trong file pickle."
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

    y = y.astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"Số mẫu không khớp: X={len(X)}, y={len(y)}"
        )

    feature_names = load_feature_names()

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Số feature không khớp: X có {X.shape[1]} cột, "
            f"feature names có {len(feature_names)} tên.\n"
            "Không tự cắt danh sách; cần xác minh đúng file metadata."
        )

    if np.any(y < 0) or np.any(y >= len(CLASS_NAMES)):
        raise ValueError(
            f"Nhãn y phải nằm trong khoảng 0 đến "
            f"{len(CLASS_NAMES) - 1}. Kiểm tra mapping class."
        )

    return X, y, feature_names

def select_samples_by_class(y, samples_per_class=SAMPLES_PER_CLASS, seed=42):
    rng = np.random.default_rng(seed)
    selected = {}
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(np.asarray(y) == class_id)
        if len(candidates):
            count = min(samples_per_class, len(candidates))
            selected[class_id] = np.sort(
                rng.choice(candidates, size=count, replace=False)
            ).astype(int).tolist()
    return selected

def load_training_background():
    x_path = PREPROCESSED_DIR / f"preprocessed_multi_{DATASET_PREFIX}_X_train.npy"
    y_path = PREPROCESSED_DIR / f"preprocessed_multi_{DATASET_PREFIX}_y_train.npy"
    X_train = np.load(x_path, mmap_mode="r")
    y_train = np.asarray(np.load(y_path)).reshape(-1)
    if len(X_train) != len(y_train):
        raise ValueError("X_train và y_train không cùng số mẫu.")
    rng = np.random.default_rng(42)
    count = min(LIME_BACKGROUND_SIZE, len(y_train))
    indices = np.sort(rng.choice(len(y_train), size=count, replace=False))
    return np.asarray(X_train[indices], dtype=np.float32)

def write_prediction_report(path, selected_by_class, sample_indices, probabilities):
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
                    f"correct={pred == true_class}, probability={probabilities[pos, pred]:.6f}\n"
                )
                f.write("  Probabilities: " + ", ".join(
                    f"{CLASS_NAMES[c]}={probabilities[pos, c]:.6f}"
                    for c in range(len(CLASS_NAMES))
                ) + "\n")

# ================= INPUT SHAPE =================
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

    if model_name in DEEP_MODELS:
        model_input = prepare_model_input(data, model_name)
        output = model.predict(model_input, verbose=0)
    elif hasattr(model, "predict_proba"):
        output = model.predict_proba(data)
    else:
        raise TypeError(
            f"{model_name} không hỗ trợ predict_proba(). "
            "Cần bổ sung bộ chuyển đổi đầu ra phù hợp."
        )

    output = np.asarray(output)

    if output.ndim == 1:
        output = output.reshape(-1, 1)

    if output.ndim != 2 or output.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Đầu ra model có shape {output.shape}; "
            f"mong đợi (samples, {len(CLASS_NAMES)}). "
            "Kiểm tra model và thứ tự class."
        )

    return output

# ================= NORMALIZE SHAP =================
def normalize_shap_values(
    raw_values, n_samples, n_features, n_classes
):
    """
    Chuyển SHAP output thành list:
        result[class_id] -> array(n_samples, n_features)
    """

    if isinstance(raw_values, list):
        result = []

        for item in raw_values:
            arr = np.asarray(item)

            # CNN/LSTM có thể thêm chiều singleton.
            if arr.ndim == 3 and arr.shape == (
                n_samples, n_features, 1
            ):
                arr = arr[:, :, 0]

            elif arr.ndim == 3 and arr.shape == (
                n_samples, 1, n_features
            ):
                arr = arr[:, 0, :]

            elif arr.ndim == 4 and arr.shape[0] == n_samples:
                arr = arr.reshape(n_samples, -1)

            if arr.shape == (n_samples, n_features):
                result.append(arr)

        if len(result) == n_classes:
            return result

        if len(result) == 1 and n_classes == 1:
            return result

    arr = np.asarray(raw_values)

    # Output có dạng (samples, features, classes).
    if arr.ndim == 3 and arr.shape == (
        n_samples, n_features, n_classes
    ):
        return [arr[:, :, c] for c in range(n_classes)]

    # Output có dạng (samples, classes, features).
    if arr.ndim == 3 and arr.shape == (
        n_samples, n_classes, n_features
    ):
        return [arr[:, c, :] for c in range(n_classes)]

    # Output CNN/LSTM có chiều thời gian singleton.
    if arr.ndim == 4 and arr.shape[0] == n_samples:
        if arr.shape[1] == 1 and arr.shape[2] == n_features:
            if arr.shape[3] == n_classes:
                arr = arr[:, 0, :, :]
                return [arr[:, :, c] for c in range(n_classes)]

            if arr.shape[3] == 1:
                arr = arr[:, 0, :, 0]
                if n_classes == 1:
                    return [arr]

        if arr.shape[1] == n_features and arr.shape[2] == 1:
            if arr.shape[3] == n_classes:
                arr = arr[:, :, 0, :]
                return [arr[:, :, c] for c in range(n_classes)]

    if arr.ndim == 2 and arr.shape == (n_samples, n_features):
        if n_classes == 1:
            return [arr]

    raise ValueError(
        f"Không chuẩn hóa được SHAP output: {arr.shape}. "
        f"Samples={n_samples}, features={n_features}, "
        f"classes={n_classes}."
    )

# ================= SHAP EXPLANATION =================
def calculate_shap(model, model_name, X, X_background):
    n_samples, n_features = X.shape
    n_classes = len(CLASS_NAMES)

    if model_name == "LightGBM":
        # Explain probabilities with KernelSHAP to avoid huge raw margins.
        def predict_fn(data):
            return predict_probabilities(
                np.asarray(data), model, model_name
            )

        explainer = shap.KernelExplainer(predict_fn, X_background)
        raw_values = explainer.shap_values(X, nsamples=KERNEL_NSAMPLES)
        values_by_class = normalize_shap_values(
            raw_values, n_samples, n_features, n_classes
        )
        background_probs = predict_probabilities(
            X_background, model, model_name
        )
        expected_by_class = np.mean(background_probs, axis=0).tolist()

    elif model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(
            # Raw output avoids requiring objective metadata from wrappers.
            model, data=X_background, model_output="raw",
            feature_perturbation="interventional",
        )
        # LightGBM multiclass may fail the additivity check from numerical
        # precision in large raw margins, even when feature dimensions match.
        raw_values = explainer.shap_values(
            X, check_additivity=(model_name != "LightGBM")
        )

        values_by_class = normalize_shap_values(
            raw_values, n_samples, n_features, n_classes
        )

        expected_raw = np.asarray(explainer.expected_value)

        if expected_raw.ndim == 0:
            expected_by_class = [float(expected_raw)] * n_classes
        else:
            expected_raw = expected_raw.reshape(-1)
            if len(expected_raw) == n_classes:
                expected_by_class = expected_raw.tolist()
            else:
                raise ValueError(
                    f"expected_value không khớp số lớp: "
                    f"{expected_raw.shape}"
                )

    elif model_name in DEEP_MODELS:
        X_input = prepare_model_input(X, model_name)
        background = prepare_model_input(X_background, model_name)

        explainer = shap.GradientExplainer(model, background)
        raw_values = explainer.shap_values(X_input)

        values_by_class = normalize_shap_values(
            raw_values, n_samples, n_features, n_classes
        )

        # GradientExplainer không cung cấp expected_value theo cách
        # TreeExplainer cung cấp. Dùng trung bình dự đoán trên background
        # làm baseline gần đúng cho biểu đồ local.
        background_probs = predict_probabilities(
            X_background[:len(background)], model, model_name
        )
        expected_by_class = np.mean(
            background_probs, axis=0
        ).tolist()

    else:
        # MLPClassifier and AdaBoostClassifier are sklearn estimators, not
        # differentiable neural nets or supported TreeExplainer types.
        def predict_fn(data):
            return predict_probabilities(
                np.asarray(data), model, model_name
            )

        explainer = shap.KernelExplainer(predict_fn, X_background)
        raw_values = explainer.shap_values(X, nsamples=KERNEL_NSAMPLES)
        values_by_class = normalize_shap_values(
            raw_values, n_samples, n_features, n_classes
        )
        background_probs = predict_probabilities(
            X_background, model, model_name
        )
        expected_by_class = np.mean(background_probs, axis=0).tolist()

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

    sample_values = np.asarray(shap_values[idx]).reshape(-1)
    sample_data = X[idx]

    if len(sample_values) != len(feature_names):
        raise ValueError(
            f"SHAP sample có {len(sample_values)} giá trị, "
            f"nhưng feature names có {len(feature_names)} tên."
        )

    exp = shap.Explanation(
        values=sample_values,
        base_values=float(expected_value),
        data=sample_data,
        feature_names=feature_names,
    )

    sample_id = idx if sample_id is None else sample_id
    suffix = f"sample_{sample_id}_class_{predicted_class}"

    # Waterfall
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=20, show=False)
    plt.title(
        f"SHAP Waterfall - {model_name} - "
        f"Sample {idx} - {class_name}",
        fontsize=11,
    )
    plt.savefig(
        save_path / f"SHAP_Waterfall_{suffix}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Force plot
    force = shap.force_plot(
        float(expected_value),
        sample_values,
        sample_data,
        feature_names=feature_names,
        matplotlib=False,
    )
    shap.save_html(
        str(save_path / f"SHAP_Force_{suffix}.html"),
        force,
    )

    # Decision plot
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        float(expected_value),
        sample_values,
        sample_data,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"SHAP Decision - {model_name} - "
        f"Sample {idx} - {class_name}",
        fontsize=11,
    )
    plt.savefig(
        save_path / f"SHAP_Decision_{suffix}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

# ================= LIME EXPLANATION =================
def run_lime(
    model,
    model_name,
    X,
    feature_names,
    sample_indices,
    predicted_classes,
    save_path,
    sample_ids,
    lime_background,
):
    from lime.lime_tabular import LimeTabularExplainer

    lime_explainer = LimeTabularExplainer(
        training_data=lime_background,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=42,
    )

    def predict_fn(data):
        return predict_probabilities(data, model, model_name)

    for pos, sample_id in enumerate(sample_ids):
        class_idx = int(predicted_classes[pos])

        exp = lime_explainer.explain_instance(
            X[pos],
            predict_fn,
            labels=[class_idx],
            num_features=min(20, X.shape[1]),
            num_samples=LIME_NUM_SAMPLES,
        )

        suffix = f"sample_{sample_id}_class_{class_idx}"
        html_file = save_path / f"LIME_{suffix}.html"
        exp.save_to_file(str(html_file))

        # Lưu thêm biểu đồ LIME PNG để tiện đưa vào báo cáo.
        fig = exp.as_pyplot_figure(label=class_idx)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        fig.savefig(
            save_path / f"LIME_{suffix}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(
            f"      ✓ LIME sample {sample_id}: "
            f"{CLASS_NAMES[class_idx]}"
        )

# ================= MAIN =================
def run_local_explanation():
    X_all, y_all, feature_names = load_data()
    selected_by_class = select_samples_by_class(y_all)
    sample_indices = [idx for ids in selected_by_class.values() for idx in ids]
    if not sample_indices:
        raise ValueError("Không có mẫu test thuộc các class cần giải thích.")
    X = np.asarray(X_all[np.asarray(sample_indices, dtype=int)], dtype=np.float32)
    y = np.asarray(y_all[sample_indices], dtype=int)
    lime_background = load_training_background()
    shap_background = lime_background[:min(BACKGROUND_SIZE, len(lime_background))]

    print("=" * 90)
    print("LOCAL XAI - NSL-KDD MULTICLASS")
    print(f"Số mẫu test: {len(X_all)}; chọn giải thích: {len(X)}")
    print(f"Số feature: {X.shape[1]}")
    print(f"Số class: {len(CLASS_NAMES)}")
    print(f"Samples per true class: {SAMPLES_PER_CLASS}")
    print(f"Sample indices grouped by true class: {selected_by_class}")
    print(f"Feature names: {len(feature_names)}")
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

        try:
            probabilities = predict_probabilities(
                X, model, model_name
            )
            predicted_classes = np.argmax(
                probabilities, axis=1
            )

            print(
                f"   ✓ Prediction shape: {probabilities.shape}"
            )

            save_path = SAVE_DIR / model_name
            save_path.mkdir(parents=True, exist_ok=True)

            write_prediction_report(save_path / "predictions.txt", selected_by_class, sample_indices, probabilities)
            write_prediction_report(save_path / "local_predictions.txt", selected_by_class, sample_indices, probabilities)

            # ================= SHAP =================
            print("   → SHAP...")

            try:
                values_by_class, expected_by_class = calculate_shap(
                    model, model_name, X, shap_background
                )

                for pos, idx in enumerate(sample_indices):
                    # Giải thích lớp mà model dự đoán cho mẫu đó.
                    class_idx = int(predicted_classes[pos])

                    save_shap_plots(
                        # SHAP arrays contain only the selected subset; use
                        # its position here and keep the test row id for naming.
                        idx=pos,
                        predicted_class=class_idx,
                        X=X,
                        feature_names=feature_names,
                        shap_values=values_by_class[class_idx],
                        expected_value=expected_by_class[class_idx],
                        model_name=model_name,
                        save_path=save_path,
                        sample_id=idx,
                    )

                    print(
                        f"      ✓ SHAP sample {idx}: "
                        f"true={CLASS_NAMES[int(y[pos])]}, "
                        f"predicted={CLASS_NAMES[class_idx]}"
                    )

            except Exception as e:
                print(
                    f"      ❌ SHAP lỗi: "
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
                    predicted_classes=predicted_classes,
                    save_path=save_path,
                    sample_ids=sample_indices,
                    lime_background=lime_background,
                )
            except Exception as e:
                print(
                    f"      ❌ LIME lỗi: "
                    f"{type(e).__name__}: {e}"
                )

        except Exception as e:
            print(
                f"   ❌ Lỗi xử lý {model_name}: "
                f"{type(e).__name__}: {e}"
            )

    print("\n🎯 DONE LOCAL XAI - NSL-KDD MULTICLASS!")
    print(f"📂 Output: {SAVE_DIR}")

# ================= RUN =================
if __name__ == "__main__":
    run_local_explanation()
