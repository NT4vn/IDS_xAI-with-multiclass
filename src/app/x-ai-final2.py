# ============================================================
# XAI GLOBAL GUI - SHAP & LIME
# Hiển thị tên Feature thật từ file feature_names.pkl
# Dataset: CIC17 / NSL-KDD / SIGNAL2022
# Model: RandomForest / LightGBM / AdaBoost / MLP / DNN / CNN / LSTM
# ============================================================

import os
from pathlib import Path
import pickle
import traceback

import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import shap
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk

# Tắt GPU để tránh lỗi TensorFlow/CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ============================================================
# PATH & CONFIG
# ============================================================

DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed_multiclass_grouped")
MODEL_DIR = Path("saved_models")

DATASETS = [
    {
        "name": "CIC17",
        "X_test": "preprocessed_multi_cic17_X_test.npy",
        "Y_test": "preprocessed_multi_cic17_y_test.npy",
        "feature_names": "preprocessed_multi_cic17_feature_names.pkl",
        "label_encoder": "preprocessed_multi_cic17_label_encoder.pkl",
    },
    {
        "name": "NSL-KDD",
        "X_test": "preprocessed_multi_nsl-kdd_X_test.npy",
        "Y_test": "preprocessed_multi_nsl-kdd_y_test.npy",
        "feature_names": "preprocessed_multi_nsl-kdd_feature_names.pkl",
        "label_encoder": "preprocessed_multi_nsl-kdd_label_encoder.pkl",
    },
    {
        "name": "SIGNAL2022",
        "X_test": "preprocessed_multi_signal22_X_test.npy",
        "Y_test": "preprocessed_multi_signal22_y_test.npy",
        "feature_names": "preprocessed_multi_signal22_feature_names.pkl",
        "label_encoder": "preprocessed_multi_signal22_label_encoder.pkl",
    },
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

# Số lượng mẫu dùng để giải thích SHAP/LIME
N_GLOBAL = 80

# ============================================================
# TKINTER GUI SETUP
# ============================================================

root = tk.Tk()
root.title("XAI GLOBAL VIEWER PRO")
root.geometry("1200x800")

# CONTROL & PLOT FRAMES
frame_control = tk.Frame(root, height=150)
frame_control.pack(fill="x", padx=10, pady=10)

frame_plot = tk.Frame(root)
frame_plot.pack(fill="both", expand=True, padx=10, pady=10)

# VARIABLES
dataset_var = tk.StringVar(value="NSL-KDD")
model_var = tk.StringVar(value="RandomForest")
explainer_var = tk.StringVar(value="TreeSHAP")
class_var = tk.StringVar(value="")

# COMBOBOXES
ttk.Label(frame_control, text="Dataset").grid(
    row=0, column=0, padx=5, pady=5
)
dataset_combo = ttk.Combobox(
    frame_control,
    textvariable=dataset_var,
    values=[d["name"] for d in DATASETS],
    width=20,
    state="readonly",
)
dataset_combo.grid(row=0, column=1, padx=5)

ttk.Label(frame_control, text="Model").grid(row=0, column=2, padx=5, pady=5)
model_combo = ttk.Combobox(
    frame_control,
    textvariable=model_var,
    values=MODELS,
    width=20,
    state="readonly",
)
model_combo.grid(row=0, column=3, padx=5)

ttk.Label(frame_control, text="Explainer").grid(
    row=0, column=4, padx=5, pady=5
)
explainer_combo = ttk.Combobox(
    frame_control,
    textvariable=explainer_var,
    values=["TreeSHAP", "KernelSHAP", "DeepSHAP", "LIME"],
    width=20,
    state="readonly",
)
explainer_combo.grid(row=0, column=5, padx=5)

ttk.Label(frame_control, text="Class to explain").grid(row=0, column=6, padx=5, pady=5)
class_combo = ttk.Combobox(frame_control, textvariable=class_var, values=[], width=22, state="readonly")
class_combo.grid(row=0, column=7, padx=5)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(name):
  print("\n" + "=" * 70 + "\nLOADING DATASET\n" + "=" * 70)
  ds = next(d for d in DATASETS if d["name"] == name)

  X_path = DATA_DIR / ds["X_test"]
  Y_path = DATA_DIR / ds["Y_test"]
  feature_path = DATA_DIR / ds["feature_names"]

  print(f"Dataset: {name}\nX path: {X_path}\nY path: {Y_path}")
  print(f"Feature path: {feature_path}")

  if not X_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file X_test:\n{X_path}")
  if not Y_path.exists():
    raise FileNotFoundError(f"Không tìm thấy file Y_test:\n{Y_path}")

  X = np.load(X_path)
  y = np.load(Y_path)
  encoder_path = DATA_DIR / ds["label_encoder"]
  if not encoder_path.exists():
    raise FileNotFoundError(f"Không tìm thấy label encoder:\n{encoder_path}")
  label_encoder = joblib.load(encoder_path)
  class_names = [str(name) for name in label_encoder.classes_]

  print(f"X shape: {X.shape}\nY shape: {y.shape}")

  if y.ndim > 1:
    y = np.argmax(y, axis=1)

  if feature_path.exists():
    feature_names = joblib.load(feature_path)
    print("\n✅ Đã load Feature Names")
  else:
    print(
        f"\n⚠️ Không tìm thấy Feature Names tại {feature_path}. Dùng tên"
        " mặc định."
    )
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

  feature_names = list(feature_names)

  print(f"\nSố feature trong X: {X.shape[1]}")
  print(f"Số feature names: {len(feature_names)}")

  if len(feature_names) != X.shape[1]:
    raise ValueError(
        f"\n❌ FEATURE MISMATCH!\nX có {X.shape[1]} feature\nFeature names"
        f" có {len(feature_names)} feature\n\nNguyên nhân có thể là file"
        " feature_names.pkl không cùng phiên bản preprocessing với X_test.npy."
    )

  print("\n===== FEATURE NAMES =====")
  for i, feature in enumerate(feature_names):
    print(f"{i:02d}: {feature}")
  print("=" * 70)

  return X, y, feature_names, class_names

def load_model(dataset, model_name):
  print(f"\nLoading model... Dataset: {dataset} | Model: {model_name}")

  if model_name in ["DNN", "CNN", "LSTM"]:
    model_path = MODEL_DIR / f"{dataset}_{model_name}.keras"
    if not model_path.exists():
      raise FileNotFoundError(f"Không tìm thấy model:\n{model_path}")
    model = tf.keras.models.load_model(model_path)
  else:
    model_path = MODEL_DIR / f"{dataset}_{model_name}.pkl"
    if not model_path.exists():
      raise FileNotFoundError(f"Không tìm thấy model:\n{model_path}")
    with open(model_path, "rb") as f:
      model = pickle.load(f)

  print("✅ Model loaded!")
  return model

def reset_plot():
  for widget in frame_plot.winfo_children():
    widget.destroy()

def create_predict_fn(model, model_name, needs_3d):

  def predict_fn(data):
    data = np.asarray(data)
    if needs_3d and data.ndim == 2:
      data = data.reshape((data.shape[0], data.shape[1], 1)) if model_name == "CNN" else data.reshape((data.shape[0], 1, data.shape[1]))

    if model_name in ["DNN", "CNN", "LSTM"]:
      result = np.asarray(model.predict(data, verbose=0))
      if result.ndim == 1:
        result = result.reshape(-1, 1)
      if result.ndim == 2 and result.shape[1] == 1:
        result = np.hstack([1 - result, result])
      return result

    if hasattr(model, "predict_proba"):
      return model.predict_proba(data)

    prediction = np.asarray(model.predict(data))
    return prediction.reshape(-1, 1)

  return predict_fn

def normalize_shap_values_legacy(sv, X):
  print("\nOriginal SHAP shape:", np.array(sv).shape)

  if isinstance(sv, list):
    print(f"SHAP output là LIST (Số output: {len(sv)})")
    sv = sv[-1] if len(sv) > 1 else sv[0]

  sv = np.asarray(sv)
  print("After list processing:", sv.shape)

  # Binary-class SHAP output often has shape: (n_samples, n_features, 2)
  # We keep the positive class contribution (index 1) for the attack class.
  if sv.ndim == 3:
    if sv.shape[-1] == 2 and sv.shape[1] == X.shape[1]:
      sv = sv[:, :, 1]
    elif sv.shape[1] == 1:
      sv = sv[:, 0, :]
    elif sv.shape[-1] == 1:
      sv = sv[:, :, 0]

  elif sv.ndim == 4:
    if sv.shape[-1] == 2:
      sv = sv[:, :, :, 1]
    elif sv.shape[1] == 1:
      sv = sv[:, 0, :, :]
    elif sv.shape[2] == 1:
      sv = sv[:, :, 0, :]

  sv = np.squeeze(sv)
  print("Normalized SHAP shape:", sv.shape)

  if sv.shape == X.shape:
    return sv

  if sv.shape[0] == X.shape[0] and sv.shape[1] == X.shape[1]:
    return sv

  if sv.size == X.size:
    sv = sv.reshape(X.shape)
    print("✅ Đã reshape SHAP theo X shape")
    return sv

  # If SHAP is still in the form (samples, features, 2) but the 2-class dim wasn't removed,
  # take the positive class directly.
  if sv.ndim == 3 and sv.shape[-1] == 2 and sv.shape[1] == X.shape[1]:
    sv = sv[:, :, 1]
    return sv

  raise ValueError(
      f"Không thể khớp SHAP shape với X shape. SHAP={sv.shape}, X={X.shape}"
  )

# ============================================================
# MULTICLASS SHAP OUTPUT NORMALIZATION
# ============================================================

def normalize_shap_values(sv, X, class_idx, n_classes):
  """Select one class from common SHAP multi-output layouts."""
  n_samples, n_features = X.shape
  if isinstance(sv, list):
    if class_idx >= len(sv):
      raise ValueError(f"SHAP returned {len(sv)} outputs; class {class_idx} requested")
    sv = np.asarray(sv[class_idx])
  else:
    sv = np.asarray(sv)

  if sv.ndim == 4:
    if sv.shape[0] == n_samples and sv.shape[1] == n_features and sv.shape[-1] == n_classes:
      sv = sv[:, :, 0, class_idx]
    elif sv.shape[0] == n_samples and sv.shape[2] == n_features and sv.shape[-1] == n_classes:
      sv = sv[:, 0, :, class_idx]
    else:
      raise ValueError(f"Cannot select class from SHAP shape {sv.shape}")
  elif sv.ndim == 3:
    if sv.shape == (n_samples, n_features, n_classes):
      sv = sv[:, :, class_idx]
    elif sv.shape == (n_samples, n_classes, n_features):
      sv = sv[:, class_idx, :]
    elif sv.shape == (n_classes, n_samples, n_features):
      sv = sv[class_idx]
    elif sv.shape == (n_samples, n_features, 1):
      sv = sv[:, :, 0]
    elif sv.shape == (n_samples, 1, n_features):
      sv = sv[:, 0, :]
    else:
      raise ValueError(f"Cannot select class from SHAP shape {sv.shape}")
  elif sv.ndim == 2 and sv.shape == (n_features, n_classes):
    sv = sv[:, class_idx].reshape(1, -1)
  elif sv.ndim == 2 and sv.shape == (n_classes, n_features):
    sv = sv[class_idx].reshape(1, -1)
  elif sv.ndim == 1 and sv.size == n_features:
    sv = sv.reshape(1, -1)

  if sv.shape != (n_samples, n_features):
    if sv.size == n_samples * n_features:
      sv = sv.reshape(n_samples, n_features)
    else:
      raise ValueError(f"SHAP shape {sv.shape} does not match {(n_samples, n_features)}")
  return sv

# ============================================================
# RUN XAI PIPELINE
# ============================================================

def run_xai():
  try:
    reset_plot()

    dataset_name = dataset_var.get()
    model_name = model_var.get()
    explainer_type = explainer_var.get()

    print("\n" + "=" * 70 + "\nXAI RUN\n" + "=" * 70)
    print(
        f"Dataset: {dataset_name} | Model: {model_name} | Explainer:"
        f" {explainer_type}"
    )

    X_test, y_test, feature_names, class_names = load_dataset(dataset_name)
    class_combo.configure(values=class_names)
    if class_var.get() not in class_names:
      class_var.set(class_names[0])
    class_idx = class_names.index(class_var.get())
    n_samples = min(N_GLOBAL, len(X_test))
    X = X_test[:n_samples]

    print(f"\nSố mẫu XAI: {X.shape[0]} | Số feature: {X.shape[1]}")

    model = load_model(dataset_name, model_name)
    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]
    predict_fn = create_predict_fn(model, model_name, needs_3d)
    model_classes = np.asarray(getattr(model, "classes_", []))
    if model_classes.size and not np.array_equal(model_classes, np.arange(len(class_names))):
      raise ValueError(f"Model class order {model_classes.tolist()} does not match encoded labels 0..{len(class_names)-1}")
    output_check = np.asarray(predict_fn(X[:1]))
    if output_check.ndim != 2 or output_check.shape[1] != len(class_names):
      raise ValueError(f"Model outputs {output_check.shape}; expected one score per {len(class_names)} classes")

    X_3d = (
        X.reshape((X.shape[0], X.shape[1], 1)) if model_name == "CNN"
        else X.reshape((X.shape[0], 1, X.shape[1])) if needs_3d
        else X
    )
    sv = None

    # TreeSHAP
    if explainer_type == "TreeSHAP":
      if model_name in ["AdaBoost", "MLP"] or is_deep:
        print(
            f"⚠️ {model_name} không dùng TreeExplainer. → Chuyển sang"
            " KernelSHAP."
        )
        predict_fn = create_predict_fn(model, model_name, needs_3d)
        explainer = shap.KernelExplainer(predict_fn, X[:20])
        sv = explainer.shap_values(X)
      else:
        try:
          explainer = shap.TreeExplainer(model)
          sv = explainer.shap_values(X)
        except Exception:
          print(
              f"⚠️ TreeExplainer không hỗ trợ cho {model_name}. → Chuyển"
              " sang KernelSHAP."
          )
          predict_fn = create_predict_fn(model, model_name, needs_3d)
          explainer = shap.KernelExplainer(predict_fn, X[:20])
          sv = explainer.shap_values(X)

    # KernelSHAP
    elif explainer_type == "KernelSHAP":
      print("\nRunning KernelSHAP...")
      predict_fn = create_predict_fn(model, model_name, needs_3d)
      explainer = shap.KernelExplainer(predict_fn, X[:20])
      sv = explainer.shap_values(X)

    # DeepSHAP
    elif explainer_type == "DeepSHAP":
      if not is_deep:
        print(
            f"⚠️ {model_name} không hỗ trợ DeepSHAP. → Chuyển sang"
            " TreeExplainer."
        )
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
      else:
        if model_name == "LSTM":
          messagebox.showwarning(
              "DeepSHAP", "DeepSHAP cho LSTM đang được bỏ qua."
          )
          return
        print("\nRunning DeepSHAP...")
        explainer = shap.DeepExplainer(model, X_3d[:30])
        sv = explainer.shap_values(X_3d)

    # LIME
    elif explainer_type == "LIME":
      print("\nRunning LIME...")
      lime_exp = LimeTabularExplainer(
          X,
          feature_names=feature_names,
          class_names=class_names,
          mode="classification",
          discretize_continuous=True,
      )
      predict_fn = create_predict_fn(model, model_name, needs_3d)
      lime_values = np.zeros((len(X), X.shape[1]))

      for i in range(len(X)):
        print(f"LIME: {i + 1}/{len(X)}", end="\r")
        exp = lime_exp.explain_instance(
            X[i], predict_fn, labels=(class_idx,), num_features=X.shape[1]
        )
        for feature_idx, weight in exp.as_map().get(class_idx, []):
          if feature_idx < X.shape[1]:
            lime_values[i, feature_idx] = weight

      print("\n✅ LIME hoàn tất!")
      sv = lime_values

    # Normalize & Plot
    sv = normalize_shap_values(sv, X, class_idx, len(class_names))
    print("\nĐang vẽ biểu đồ...")

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv,
        X,
        feature_names=feature_names,
        show=False,
        max_display=min(20, len(feature_names)),
    )
    plt.title(
        f"{explainer_type} - {model_name} - {dataset_name} - {class_names[class_idx]}", fontsize=12
    )
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    print("\n✅ XAI hoàn tất!")

  except Exception as e:
    print("\n❌ XAI ERROR:")
    traceback.print_exc()
    messagebox.showerror("XAI Error", str(e))

# ============================================================
# BUTTONS & MAINLOOP
# ============================================================

run_button = ttk.Button(
    frame_control, text="Run", command=run_xai
)
run_button.grid(row=1, column=2, pady=10, padx=5)

reset_button = ttk.Button(
    frame_control, text="Reset", command=reset_plot
)
reset_button.grid(row=1, column=3, pady=10, padx=5)

root.mainloop()
