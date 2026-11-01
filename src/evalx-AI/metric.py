# =====================================================
# XAI METRICS FULL - FINAL (ADD LSTM + FIX ALL BUGS)
# =====================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pickle
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import tensorflow as tf
import warnings
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from joblib import parallel_backend

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
tf.get_logger().setLevel('ERROR')

# ================= PATH =================
DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
MODEL_DIR = Path("saved_models")
SAVE_DIR = Path("metric1")
SAVE_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "CIC17", "X_test": "preprocessed_multi_cic17_X_test.npy", "Y_test": "preprocessed_multi_cic17_y_test.npy"},
    {"name": "NSL-KDD", "X_test": "preprocessed_multi_nsl-kdd_X_test.npy","Y_test": "preprocessed_multi_nsl-kdd_y_test.npy"},
    {"name": "SIGNAL2022", "X_test": "preprocessed_multi_signal22_X_test.npy","Y_test": "preprocessed_multi_signal22_y_test.npy"}
]

MODELS = ["RandomForest", "LightGBM", "AdaBoost", "MLP", "DNN", "CNN", "LSTM"]

# ================= CONFIG =================
N_SAMPLES = 15
NOISE = 0.03
BACKGROUND_SAMPLES = 30

# =====================================================
# ================= METRICS =================
# =====================================================

def deletion_metric(model_fn, x, w, y_true, baseline):
    x = x.flatten()
    order = np.argsort(-np.abs(w))
    scores = []
    for i in range(len(w) + 1):
        x_mod = x.copy()
        x_mod[order[:i]] = baseline[order[:i]]
        pred = model_fn(x_mod.reshape(1, -1))[0]
        scores.append(pred[y_true])
    return np.mean(scores)

def insertion_metric(model_fn, x, w, y_true, baseline):
    x = x.flatten()
    order = np.argsort(-np.abs(w))
    scores = []
    for i in range(len(w) + 1):
        x_mod = baseline.copy()
        x_mod[order[:i]] = x[order[:i]]
        pred = model_fn(x_mod.reshape(1, -1))[0]
        scores.append(pred[y_true])
    return np.mean(scores)

def infidelity(model_fn, x, w, target_class, noise_std=0.02, n_samples=15):
    x = x.flatten()
    scores = []
    for _ in range(n_samples):
        noise = np.random.normal(0, noise_std, x.shape)
        x_noisy = x + noise
        pred_diff = model_fn(x.reshape(1, -1))[0, target_class] - model_fn(x_noisy.reshape(1, -1))[0, target_class]
        approx = np.dot(w, noise)
        scores.append((pred_diff - approx) ** 2)
    return np.mean(scores)

def safe_spearman(w1, w2):
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    if w1.size == 0 or w2.size == 0:
        return 0.0
    if np.allclose(w1, w1[0]) or np.allclose(w2, w2[0]):
        return 0.0
    corr, _ = spearmanr(w1, w2)
    return float(corr) if not np.isnan(corr) else 0.0

def importance_corr(model_fn, x, w, y_true):
    x = x.flatten()
    changes = []
    for i in range(len(w)):
        x_mod = x.copy()
        x_mod[i] = 0
        pred_orig = model_fn(x.reshape(1, -1))[0]
        pred_mod = model_fn(x_mod.reshape(1, -1))[0]
        changes.append(pred_orig[y_true] - pred_mod[y_true])
    return safe_spearman(w, changes)

def robustness(w1, w2):
    return np.linalg.norm(w1 - w2)

def sparsity(w):
    return np.count_nonzero(np.abs(w) > 1e-8) / len(w)

def entropy(w):
    w = np.abs(w)
    w = w / (np.sum(w) + 1e-12)
    return -np.sum(w * np.log(w + 1e-12))

def topk_ratio(w, k=5):
    abs_w = np.abs(w)
    return np.sum(np.sort(abs_w)[-k:]) / (np.sum(abs_w) + 1e-12)

def consistency(w1, w2):
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    sp = safe_spearman(w1, w2)
    if np.allclose(w1, 0) or np.allclose(w2, 0):
        cos = 0.0
    else:
        cos = 1 - cosine(w1, w2)
    return sp, cos

def normalize_shap_vector(sv, feature_count, class_index):
    if sv is None:
        return np.zeros(feature_count, dtype=float)

    if isinstance(sv, (list, tuple)):
        if len(sv) == 0:
            return np.zeros(feature_count, dtype=float)
        sv = sv[min(class_index, len(sv) - 1)]

    if hasattr(sv, "values"):
        sv = sv.values

    arr = np.asarray(sv, dtype=float)
    if arr.size == 0:
        return np.zeros(feature_count, dtype=float)

    # SHAP multiclass layouts can be (samples, features, classes) or
    # (samples, classes, features); reduce explicitly to the requested class.
    if arr.ndim == 3:
        if arr.shape[1] == feature_count:
            arr = arr[0, :, min(class_index, arr.shape[2] - 1)]
        elif arr.shape[2] == feature_count:
            arr = arr[0, min(class_index, arr.shape[1] - 1), :]
    elif arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == feature_count:
            arr = arr[0]
        elif arr.shape[0] == feature_count:
            arr = arr[:, min(class_index, arr.shape[1] - 1)]
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            if arr.shape[0] == feature_count:
                arr = arr[:, min(class_index, arr.shape[1] - 1)]
            elif arr.shape[1] == feature_count:
                arr = arr[0]

    arr = np.asarray(arr).reshape(-1)

    if arr.size > feature_count:
        arr = arr[:feature_count]
    elif arr.size < feature_count:
        pad = np.zeros(feature_count - arr.size, dtype=float)
        arr = np.concatenate([arr, pad])

    return arr.astype(float)

# =====================================================
# ================= MAIN =================
# =====================================================
print("=== XAI FULL METRICS FINAL ===")

for ds in DATASETS:
    dataset_name = ds["name"]

    X_test = np.load(DATA_DIR / ds["X_test"])
    y_test = np.load(DATA_DIR / ds["Y_test"])
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    baseline = np.mean(X_test, axis=0)
    feature_names = [f"F{i}" for i in range(X_test.shape[1])]

    for model_name in MODELS:
        print("\n-> " + model_name)

        try:
            # ================= LOAD MODEL =================
            if model_name in ["DNN", "CNN", "LSTM"]:
                model = tf.keras.models.load_model(MODEL_DIR / f"{dataset_name}_{model_name}.keras")
                is_keras = True
                needs_3d = (model_name in ["CNN", "LSTM"])
            else:
                with open(MODEL_DIR / f"{dataset_name}_{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                is_keras = False
                needs_3d = False

            # ================= PREDICT FN =================
            def predict_fn(X):
                with parallel_backend('sequential'):
                    if needs_3d:
                        X = X.reshape((X.shape[0], 1, X.shape[1]))

                    if is_keras:
                        preds = model.predict(X, verbose=0)
                    else:
                        preds = model.predict_proba(X)

                    if preds.ndim == 1:
                        preds = np.vstack([1 - preds, preds]).T

                    if preds.shape[1] == 1:
                        preds = np.hstack([1 - preds, preds])

                    return preds

            # ================= EXPLAINERS =================
            kernel_explainer = shap.KernelExplainer(predict_fn, X_test[:BACKGROUND_SAMPLES])
            tree_explainer = None
            deep_explainer = None

            if model_name in ["RandomForest", "LightGBM"]:
                tree_explainer = shap.TreeExplainer(model)

            # ❌ KHÔNG dùng DeepSHAP cho LSTM
            if model_name in ["DNN", "CNN"]:
                try:
                    bg = X_test[:BACKGROUND_SAMPLES]
                    if model_name == "CNN":
                        bg = bg.reshape((bg.shape[0], 1, bg.shape[1]))
                    deep_explainer = shap.DeepExplainer(model, bg)
                except:
                    deep_explainer = None

            lime_exp = LimeTabularExplainer(
                X_test[:100],
                feature_names=feature_names,
                mode="classification"
            )

            rows = []
            if len(X_test) == 0:
                raise ValueError("Test dataset is empty")
            indices = np.random.choice(len(X_test), min(N_SAMPLES, len(X_test)), replace=False)

            for idx in indices:
                x = X_test[idx:idx+1]
                # Use the ground-truth class consistently as the explanation target.
                target_class = int(y_test[idx])

                shap_dict = {}

                # ================= KernelSHAP =================
                sv = kernel_explainer.shap_values(x)
                shap_dict["KernelSHAP"] = normalize_shap_vector(sv, X_test.shape[1], target_class)

                # ================= TreeSHAP =================
                if tree_explainer:
                    sv = tree_explainer.shap_values(x)
                    shap_dict["TreeSHAP"] = normalize_shap_vector(sv, X_test.shape[1], target_class)

                # ================= DeepSHAP =================
                if deep_explainer:
                    x_input = x
                    if model_name == "CNN":
                        x_input = x.reshape((1, 1, x.shape[1]))

                    sv = deep_explainer.shap_values(x_input)
                    shap_dict["DeepSHAP"] = normalize_shap_vector(sv, X_test.shape[1], target_class)

                # ================= LIME =================
                exp = lime_exp.explain_instance(
                    x[0], predict_fn, labels=(target_class,), num_features=len(feature_names)
                )
                lime_w = np.zeros(len(feature_names))

                class_id = target_class
                for fid, weight in exp.local_exp[class_id]:
                    lime_w[fid] = weight

                shap_dict["LIME"] = lime_w

                row = {}

                # ================= METRICS =================
                for name, w in shap_dict.items():
                    p = name.lower().replace("shap", "")

                    row[f"del_{p}"] = deletion_metric(predict_fn, x[0], w, target_class, baseline)
                    row[f"ins_{p}"] = insertion_metric(predict_fn, x[0], w, target_class, baseline)
                    row[f"infid_{p}"] = infidelity(predict_fn, x[0], w, target_class)
                    row[f"corr_{p}"] = importance_corr(predict_fn, x[0], w, target_class)
                    row[f"sparsity_{p}"] = sparsity(w)
                    row[f"entropy_{p}"] = entropy(w)
                    row[f"topk_{p}"] = topk_ratio(w)

                # ================= ROBUSTNESS =================
                noise = np.random.normal(0, NOISE, x.shape)
                sv_noisy = kernel_explainer.shap_values(x + noise)
                noisy_vec = normalize_shap_vector(sv_noisy, X_test.shape[1], target_class)
                row["robust"] = robustness(shap_dict["KernelSHAP"], noisy_vec)

                # ================= CONSISTENCY =================
                if "KernelSHAP" in shap_dict and "LIME" in shap_dict:
                    sp, cos = consistency(shap_dict["KernelSHAP"], shap_dict["LIME"])
                    row["spearman"] = sp
                    row["cosine"] = cos

                rows.append(row)

            df = pd.DataFrame(rows)
            m = df.mean(numeric_only=True)

            metrics_names = [
                "Deletion", "Insertion", "Infidelity", "Importance",
                "Robustness", "Sparsity", "Entropy", "TopK_Ratio",
                "Spearman", "Cosine"
            ]

            def safe_metric_value(series, *keys):
                for key in keys:
                    value = series.get(key, np.nan)
                    if pd.notna(value):
                        return float(value)
                return np.nan

            result_rows = []

            for metric in metrics_names:
                row = {"Model": model_name, "Metric": metric}

                mapping = {
                    "Deletion": "del_",
                    "Insertion": "ins_",
                    "Infidelity": "infid_",
                    "Importance": "corr_",
                    "Sparsity": "sparsity_",
                    "Entropy": "entropy_",
                    "TopK_Ratio": "topk_"
                }

                if metric in mapping:
                    key = mapping[metric]
                    kernel_val = safe_metric_value(m, key + "kernel")
                    tree_val = safe_metric_value(m, key + "tree")
                    deep_val = safe_metric_value(m, key + "deep")
                    lime_val = safe_metric_value(m, key + "lime")

                    row["KernelSHAP"] = kernel_val
                    row["TreeSHAP"] = tree_val
                    row["DeepSHAP"] = deep_val
                    row["LIME"] = lime_val

                elif metric == "Robustness":
                    val = safe_metric_value(m, "robust")
                    if pd.isna(val):
                        val = 0.0
                    row.update({"KernelSHAP": val, "TreeSHAP": np.nan, "DeepSHAP": np.nan, "LIME": np.nan})

                elif metric == "Spearman":
                    val = safe_metric_value(m, "spearman")
                    if pd.isna(val):
                        val = 0.0
                    row.update({"KernelSHAP": val, "TreeSHAP": np.nan, "DeepSHAP": np.nan, "LIME": val})

                elif metric == "Cosine":
                    val = safe_metric_value(m, "cosine")
                    if pd.isna(val):
                        val = 0.0
                    row.update({"KernelSHAP": val, "TreeSHAP": np.nan, "DeepSHAP": np.nan, "LIME": val})

                result_rows.append(row)

            result_df = pd.DataFrame(result_rows)

            save_file = SAVE_DIR / f"{dataset_name}_{model_name}_FINAL.xlsx"
            result_df.to_excel(save_file, index=False)

            print("OK Saved: " + str(save_file))

        except Exception as e:
            print("ERROR " + model_name + ": " + str(e))

print("\n=== DONE FULL ===")
