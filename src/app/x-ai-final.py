# =====================================================
# XAI VIEWER PRO VERSION (FULL FEATURES)
# FIX: LOAD ĐÚNG TÊN FEATURE TỪ *_feature_names.pkl
# =====================================================

from pathlib import Path
import pickle
import warnings

import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

warnings.filterwarnings("ignore")

# =====================================================
# PATH
# =====================================================

DATA_DIR = Path(r"D:\luanvsn\.venv\final\data\preprocessed")
MODEL_DIR = Path("saved_models")
XAI_DIR = Path("xai_comparison_final4")

XAI_DIR.mkdir(exist_ok=True)
(XAI_DIR / "global").mkdir(exist_ok=True)
(XAI_DIR / "local").mkdir(exist_ok=True)
(XAI_DIR / "comparison").mkdir(exist_ok=True)

# =====================================================
# DATASETS
# =====================================================

DATASETS = [
    {
        "name": "CIC17",
        "key": "cic17",
        "X_test": "preprocessed_cic17_X_test.npy",
        "Y_test": "preprocessed_cic17_y_test.npy",
        "feature_names": "preprocessed_cic17_feature_names.pkl",
    },
    {
        "name": "NSL-KDD",
        "key": "nsl-kdd",
        "X_test": "preprocessed_nsl-kdd_X_test.npy",
        "Y_test": "preprocessed_nsl-kdd_y_test.npy",
        "feature_names": "preprocessed_nsl-kdd_feature_names.pkl",
    },
    {
        "name": "SIGNAL2022",
        "key": "signal22",
        "X_test": "signal22_X_test.npy",
        "Y_test": "signal22_y_test.npy",
        "feature_names": "signal22_feature_names.pkl",
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

# =====================================================
# GLOBAL VARIABLES
# =====================================================

model = None
predict_fn = None
explainer = None
lime_explainer = None
X_test = None
y_test = None
feature_names = None
needs_3d = False
current_data = {}
canvas = None

# =====================================================
# LOAD FEATURE NAMES
# =====================================================

def load_feature_names(dataset_info, n_features):
  feature_file = DATA_DIR / dataset_info["feature_names"]

  print("\n====================================")
  print("LOAD FEATURE NAMES")
  print("====================================")
  print("Feature file:", feature_file)

  if not feature_file.exists():
    print("Không tìm thấy file feature names! Sử dụng tên mặc định F_0, F_1,...")
    return [f"F_{i}" for i in range(n_features)]

  try:
    loaded_features = list(joblib.load(feature_file))
    print("Số feature trong file:", len(loaded_features))
    print("Số feature trong X_test:", n_features)

    if len(loaded_features) != n_features:
      print("⚠️ Số lượng feature không khớp! Fallback về F_0, F_1,...")
      return [f"F_{i}" for i in range(n_features)]

    print("\n✓ Feature names loaded:")
    for i, name in enumerate(loaded_features):
      print(f"{i:02d}: {name}")
    print("\n✓ Đã load đúng tên feature!")

    return loaded_features
  except Exception as e:
    print("❌ Lỗi load feature names:", e)
    return [f"F_{i}" for i in range(n_features)]

# =====================================================
# LOAD PIPELINE
# =====================================================

def load_pipeline(dataset_name: str, model_name: str):
  global model, predict_fn, explainer, lime_explainer
  global X_test, y_test, feature_names, needs_3d

  try:
    # LOAD DATASET INFO
    ds_info = next(ds for ds in DATASETS if ds["name"] == dataset_name)

    # LOAD X_TEST & Y_TEST
    X_test = np.load(DATA_DIR / ds_info["X_test"])
    y_test = np.load(DATA_DIR / ds_info["Y_test"])

    if y_test.ndim > 1:
      y_test = np.argmax(y_test, axis=1)

    # LOAD TÊN FEATURE
    feature_names = load_feature_names(ds_info, X_test.shape[1])

    print("\nDataset:", dataset_name)
    print("Model:", model_name)
    print("X_test shape:", X_test.shape)
    print("Feature count:", len(feature_names))

    is_deep = model_name in ["DNN", "CNN", "LSTM"]
    needs_3d = model_name in ["CNN", "LSTM"]

    # LOAD MODEL
    if is_deep:
      model_path = MODEL_DIR / f"{dataset_name}_{model_name}.h5"
      model = tf.keras.models.load_model(model_path)

      def predict_fn_local(X):
        X = np.asarray(X)
        if needs_3d:
          if X.ndim == 1:
            X = X.reshape((1, 1, -1))
          elif X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        p = model.predict(X, verbose=0)
        if p.ndim == 1:
          p = p.reshape((-1, 1))
        if p.shape[1] == 1:
          p = np.hstack([1 - p, p])
        return p

      predict_fn = predict_fn_local
    else:
      model_path = MODEL_DIR / f"{dataset_name}_{model_name}.pkl"
      with open(model_path, "rb") as f:
        model = pickle.load(f)
      predict_fn = model.predict_proba

    # SHAP EXPLAINER
    if is_deep:
      background = X_test[:50]
      if model_name in ["LSTM", "CNN"]:
        explainer = shap.KernelExplainer(predict_fn, background[:20])
      else:
        explainer = shap.DeepExplainer(model, background)
    elif model_name in ["AdaBoost", "MLP"]:
      background = X_test[:50]
      explainer = shap.KernelExplainer(predict_fn, background)
    else:
      explainer = shap.TreeExplainer(model)

    # LIME EXPLAINER
    lime_explainer = LimeTabularExplainer(
        X_test[:100],
        feature_names=feature_names,
        class_names=["Benign", "Attack"],
        mode="classification",
    )

    print("\n✓ Loaded successfully!")
    print(f"✓ Dataset: {dataset_name}")
    print(f"✓ Model: {model_name}")
    print(f"✓ Features: {len(feature_names)}")

  except Exception as e:
    print(f"✗ Error loading pipeline: {e}")
    raise

# =====================================================
# GUI SETUP
# =====================================================

root = tk.Tk()
root.title("XAI Viewer PRO - Full Version")
root.geometry("1300x900")
root.minsize(980, 650)
root.resizable(True, True)
root.columnconfigure(0, weight=1)
root.rowconfigure(2, weight=1)

# =====================================================
# TOP FRAME
# =====================================================

frame_top = tk.Frame(root)
frame_top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
frame_top.grid_columnconfigure(0, weight=1)

# Dataset
tk.Label(frame_top, text="Dataset:").pack(side=tk.LEFT, padx=5)
dataset_box = ttk.Combobox(
    frame_top, values=[ds["name"] for ds in DATASETS], width=12
)
dataset_box.set("CIC17")
dataset_box.pack(side=tk.LEFT, padx=5)

# Model
tk.Label(frame_top, text="Model:").pack(side=tk.LEFT, padx=5)
model_box = ttk.Combobox(frame_top, values=MODELS, width=15)
model_box.set("RandomForest")
model_box.pack(side=tk.LEFT, padx=5)

# Filter
tk.Label(frame_top, text="Filter:").pack(side=tk.LEFT, padx=5)
filter_box = ttk.Combobox(
    frame_top,
    values=["All", "Only Benign", "Only Attack", "Wrong Only"],
    width=15,
)
filter_box.set("All")
filter_box.pack(side=tk.LEFT, padx=5)

# Sample Index
tk.Label(frame_top, text="Sample Index:").pack(side=tk.LEFT, padx=5)
entry_idx = tk.Entry(frame_top, width=12)
entry_idx.pack(side=tk.LEFT, padx=5)

# =====================================================
# RESULT TEXT
# =====================================================

result_text = scrolledtext.ScrolledText(root, height=14, font=("Consolas", 10))
result_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

result_text.tag_config(
    "wrong", foreground="red", font=("Consolas", 10, "bold")
)
result_text.tag_config("highlight", background="yellow")
result_text.tag_config("positive", foreground="red")
result_text.tag_config("negative", foreground="blue")

# =====================================================
# PLOT FRAME
# =====================================================

frame_plot = tk.Frame(root)
frame_plot.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 5))
frame_plot.grid_rowconfigure(0, weight=1)
frame_plot.grid_columnconfigure(0, weight=1)

# =====================================================
# FILTER LOGIC
# =====================================================

def get_filtered_indices(filter_type: str):
  preds = predict_fn(X_test)
  preds = np.argmax(preds, axis=1)

  if filter_type == "Only Benign":
    return np.where(y_test == 0)[0]
  elif filter_type == "Only Attack":
    return np.where(y_test == 1)[0]
  elif filter_type == "Wrong Only":
    return np.where(preds != y_test)[0]
  else:
    return np.arange(len(X_test))

# =====================================================
# EXPLAIN SAMPLE
# =====================================================

def explain_sample():
  global canvas, current_data

  result_text.delete(1.0, tk.END)

  DATASET = dataset_box.get()
  MODEL = model_box.get()
  FILTER = filter_box.get()

  try:
    load_pipeline(DATASET, MODEL)
  except Exception as e:
    result_text.insert(
        tk.END, f"Lỗi load model/dataset: {e}\n", "wrong"
    )
    return

  indices = get_filtered_indices(FILTER)

  try:
    idx = int(entry_idx.get())
    if idx < 0 or idx >= len(indices):
      raise ValueError
    real_idx = indices[idx]
  except Exception:
    result_text.insert(
        tk.END, "Index không hợp lệ hoặc ngoài phạm vi!\n", "wrong"
    )
    return

  x = X_test[real_idx]
  x_predict = x.reshape(1, 1, -1) if needs_3d else x.reshape(1, -1)

  pred_prob = predict_fn(x_predict)
  pred = int(np.argmax(pred_prob[0]))
  true = int(y_test[real_idx])

  result_text.insert(
      tk.END, f"Sample Index (filtered): {idx} | Real Index: {real_idx}\n"
  )
  result_text.insert(
      tk.END,
      f"True Label   : {true} ({'Benign' if true == 0 else 'Attack'})\n",
  )
  result_text.insert(
      tk.END,
      f"Predicted    : {pred} ({'Benign' if pred == 0 else 'Attack'})\n",
  )

  if pred_prob.shape[1] >= 2:
    benign_prob = float(pred_prob[0][0]) * 100
    attack_prob = float(pred_prob[0][1]) * 100
    result_text.insert(
        tk.END, f"Benign Probability: {benign_prob:.2f}%\n"
    )
    result_text.insert(
        tk.END, f"Attack Probability: {attack_prob:.2f}%\n"
    )

  if pred != true:
    result_text.insert(
        tk.END, "⚠️ MISCLASSIFIED SAMPLE ⚠️\n", "wrong"
    )

  # SHAP
  shap_list = []
  top_idx = []
  top_values = []

  try:
    x_input = x.reshape(1, -1)
    sv = explainer.shap_values(x_input)

    if isinstance(sv, list):
      shap_vals = sv[1] if len(sv) > 1 else sv[0]
    else:
      shap_vals = sv

    shap_vals = np.asarray(shap_vals)

    if shap_vals.ndim == 3:
      if shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[0, :, 1]
      else:
        shap_vals = shap_vals[0, :, 0]
    elif shap_vals.ndim == 2:
      shap_vals = shap_vals[0]

    shap_vals = np.asarray(shap_vals).flatten()

    if len(shap_vals) != len(feature_names):
      raise ValueError(
          f"SHAP shape {len(shap_vals)} != feature names"
          f" {len(feature_names)}"
      )

    num_top = min(10, len(shap_vals))
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:num_top].astype(int)

    result_text.insert(
        tk.END, f"\n--- SHAP Top {num_top} Features ---\n"
    )
    for i in top_idx:
      val = float(shap_vals[i])
      fname = feature_names[i]
      shap_list.append((fname, val))
      tag = "positive" if val > 0 else "negative"
      result_text.insert(tk.END, f"{fname:>35}: {val:>8.4f}\n", tag)

    top_values = shap_vals[top_idx]

  except Exception as e:
    result_text.insert(tk.END, f"Lỗi SHAP: {e}\n", "wrong")
    print(f"SHAP debug error: {e}")

  # LIME
  lime_list = []
  try:
    exp = lime_explainer.explain_instance(
        x, predict_fn, num_features=10
    )
    lime_list = exp.as_list()

    result_text.insert(tk.END, "\n--- LIME Top 10 Features ---\n")
    for f, v in lime_list:
      result_text.insert(tk.END, f"{f}: {v:.4f}\n")
  except Exception as e:
    result_text.insert(tk.END, f"Lỗi LIME: {e}\n", "wrong")

  # SAVE CURRENT DATA
  current_data = {
      "true": true,
      "pred": pred,
      "shap": shap_list,
      "lime": lime_list,
      "sample_idx": real_idx,
  }

  # CLEAR OLD PLOT
  for widget in frame_plot.winfo_children():
    widget.destroy()

  # DRAW SHAP PLOT
  fig, ax = plt.subplots(figsize=(10, 6))

  if len(top_idx) > 0:
    top_features = [feature_names[i] for i in top_idx]
    ax.barh(top_features[::-1], top_values[::-1])
    ax.axvline(0, linewidth=1)
    ax.set_title(
        f"SHAP Local Explanation - {DATASET} | {MODEL} | Sample {real_idx}"
    )
    ax.set_xlabel(
        "SHAP Value (Positive → Attack, Negative → Benign)"
    )
    fig.tight_layout()

  canvas = FigureCanvasTkAgg(fig, master=frame_plot)
  canvas.draw()
  canvas.get_tk_widget().pack(fill="both", expand=True)

# =====================================================
# EXPORT TO EXCEL
# =====================================================

def export_excel():
  if not current_data:
    messagebox.showwarning("Warning", "Chưa có dữ liệu để export!")
    return

  file = filedialog.asksaveasfilename(
      defaultextension=".xlsx",
      filetypes=[("Excel files", "*.xlsx")],
      initialdir=str(XAI_DIR / "local"),
  )

  if not file:
    return

  try:
    df_shap = pd.DataFrame(
        current_data["shap"], columns=["Feature", "SHAP_Value"]
    )
    df_lime = pd.DataFrame(
        current_data["lime"], columns=["Feature", "LIME_Value"]
    )

    with pd.ExcelWriter(file) as writer:
      df_shap.to_excel(writer, sheet_name="SHAP", index=False)
      df_lime.to_excel(writer, sheet_name="LIME", index=False)

    messagebox.showinfo(
        "Success", f"Đã export thành công:\n{file}"
    )
  except Exception as e:
    messagebox.showerror("Error", f"Export thất bại: {e}")

# =====================================================
# GLOBAL IMPORTANCE
# =====================================================

def show_global():
  try:
    sv = explainer.shap_values(X_test[:200])

    if isinstance(sv, list):
      sv = sv[1] if len(sv) > 1 else sv[0]

    sv = np.asarray(sv)

    if sv.ndim == 3:
      sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[:, :, 0]

    mean_abs = np.mean(np.abs(sv), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in top_idx[::-1]],
        mean_abs[top_idx][::-1],
    )
    plt.title(
        f"Global Feature Importance - {dataset_box.get()} |"
        f" {model_box.get()}"
    )
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.show()

  except Exception as e:
    messagebox.showerror("Error", f"Lỗi Global Importance: {e}")

# =====================================================
# BUTTONS
# =====================================================

btn_frame = tk.Frame(root)
btn_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
for i in range(3):
    btn_frame.grid_columnconfigure(i, weight=1)

btn_explain = tk.Button(
    btn_frame,
    text="Explain Sample",
    command=explain_sample,
    bg="#4CAF50",
    fg="white",
)
btn_explain.grid(row=0, column=0, sticky="ew", padx=(0, 10))

btn_export = tk.Button(
    btn_frame,
    text="Export Excel",
    command=export_excel,
    bg="#2196F3",
    fg="white",
)
btn_export.grid(row=0, column=1, sticky="ew", padx=(0, 10))

btn_global = tk.Button(
    btn_frame,
    text="Global Importance",
    command=show_global,
    bg="#FF9800",
    fg="white",
)
btn_global.grid(row=0, column=2, sticky="ew")

# =====================================================
# RUN GUI
# =====================================================

root.mainloop()
