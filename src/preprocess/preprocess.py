import pandas as pd
import numpy as np
import joblib
import os
import glob
import gc
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# PATH CONFIGURATION
# ======================
BASE_DATA_PATH = r'D:\luanvsn\.venv\final\data'
OUTPUT_DIR = os.path.join(BASE_DATA_PATH, 'preprocessed_multiclass_grouped')

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Thư mục lưu kết quả:", OUTPUT_DIR)

# ======================
# MAPPING DICTIONARIES (CHỈ DÙNG CHO NSL-KDD VÀ CIC17)
# ======================
NSL_5_CLASSES = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

CIC_7_CLASSES = {
    'benign': 'Normal',
    'dos hulk': 'DoS',
    'dos goldeneye': 'DoS',
    'dos slowloris': 'DoS',
    'dos slowhttptest': 'DoS',
    'heartbleed': 'DoS',
    'ddos': 'DDoS',
    'portscan': 'Port Scan',
    'ftp-patator': 'Brute Force',
    'ssh-patator': 'Brute Force',
    'web attack brute force': 'Web Attack',
    'web attack xss': 'Web Attack',
    'web attack sql injection': 'Web Attack',
    'bot': 'Botnet / Infiltration',
    'infiltration': 'Botnet / Infiltration'
}

DATASETS = {
    'cic17': {
        'folder_name': 'cicids-17',
        'label_column_patterns': ['label'],
        'drop_early_columns': ['flow id', 'source ip', 'destination ip', 'timestamp'],
        'encoding': 'latin1',
        'no_header': False,
        'max_samples': None
    },
    'nsl-kdd': {
        'folder_name': 'nsl-kdd',
        'encoding': 'utf-8',
        'no_header': True,
        'max_samples': None
    },
    'signal22': {
        'folder_name': 'Simargl2022',
        'drop_early_columns': ['flow_id', 'ipv4_src_addr', 'ipv4_dst_addr', 'analysis_timestamp', 'first_switched', 'last_switched', 'ip', 'mac'],
        'encoding': 'utf-8',
        'no_header': False,
        'max_samples': 200000
    }
}

K_FEATURES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

def optimize_dtypes(df):
    """ Tối ưu dung lượng RAM bằng cách downcast kiểu dữ liệu số """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# ======================
# MAIN LOOP
# ======================
for DATASET_KEY, config in DATASETS.items():

    print("\n" + "="*80)
    print("DATASET MULTI-CLASS (NO LEAKAGE - OPTIMIZED):", DATASET_KEY.upper())
    print("="*80)

    DATA_FOLDER = os.path.join(BASE_DATA_PATH, config['folder_name'])
    OUTPUT_PREFIX = f"preprocessed_multi_{DATASET_KEY}"

    if not os.path.exists(DATA_FOLDER):
        print("❌ Không tồn tại:", DATA_FOLDER)
        continue

    files = glob.glob(os.path.join(DATA_FOLDER, '*.csv')) + \
            glob.glob(os.path.join(DATA_FOLDER, '*.txt'))

    if DATASET_KEY == 'nsl-kdd':
        files = [
            f for f in files
            if ('KDDTrain+.txt' in f or 'KDDTest+.txt' in f)
            and 'attack_types' not in f
        ]

    if not files:
        print("❌ Không có file phù hợp")
        continue

    print("Found files:", len(files))
    dfs = []

    for f in files:
        fname = os.path.basename(f).lower()
        try:
            df_temp = pd.read_csv(
                f,
                encoding=config['encoding'],
                engine='c',
                low_memory=False,
                on_bad_lines='skip',
                header=None if config.get('no_header') else 0
            )

            if not config.get('no_header'):
                df_temp.columns = df_temp.columns.str.strip().str.lower()
                first_col = df_temp.columns[0]
                df_temp = df_temp[df_temp[first_col].astype(str).str.lower() != str(first_col).lower()]

            # ===== SỬA NGUYÊN NHÂN TRẢ VỀ NAN Ở SIMARGL2022 =====
            if DATASET_KEY == 'signal22':
                existing_label_col = next((c for c in df_temp.columns if c in ['alert', 'anomaly', 'label', 'label_custom']), None)

                # Xác định nhãn dự phòng từ tên file
                file_label = 'dos' if 'dos' in fname else ('normal' if 'normal' in fname else ('portscanning' if 'port' in fname else ('malware' if 'malware' in fname else 'unknown')))

                if existing_label_col:
                    df_temp['label_custom'] = df_temp[existing_label_col].fillna(file_label).astype(str).str.strip().str.lower()
                    # Xóa cột nhãn gốc để tránh bị biến thành feature khi One-Hot Encoding
                    if existing_label_col != 'label_custom':
                        df_temp.drop(columns=[existing_label_col], inplace=True)
                else:
                    df_temp['label_custom'] = file_label

            # Lấy mẫu ngẫu nhiên giảm RAM
            max_s = config.get('max_samples')
            if max_s and len(df_temp) > max_s:
                df_temp = df_temp.sample(max_s, random_state=RANDOM_STATE)

            df_temp = optimize_dtypes(df_temp)
            print(f"Loaded: {os.path.basename(f)} ({df_temp.shape})")
            dfs.append(df_temp)

        except Exception as e:
            print(f"❌ Lỗi đọc file {os.path.basename(f)}:", e)

    if not dfs:
        continue

    common_cols = list(set.intersection(*(set(d.columns) for d in dfs)))
    dfs = [d[common_cols] for d in dfs]

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print("Total rows loaded:", len(df))

    if DATASET_KEY == 'nsl-kdd':
        nsl_columns = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes',
            'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
            'num_compromised','root_shell','su_attempted','num_root',
            'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
            'is_host_login','is_guest_login','count','srv_count','serror_rate',
            'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
            'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
            'dst_host_serror_rate','dst_host_srv_serror_rate',
            'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
        ]
        if len(df.columns) >= 42:
            df = df.iloc[:, :42]
            df.columns = nsl_columns

    # Lấy tên cột nhãn
    if DATASET_KEY == 'signal22':
        label_col = 'label_custom'
    elif DATASET_KEY == 'nsl-kdd':
        label_col = 'label'
    else:
        label_col = next((c for c in df.columns if any(p in c for p in config['label_column_patterns'])), None)

    if label_col is None or label_col not in df.columns:
        print("❌ Không tìm thấy cột label")
        continue

    # Loại bỏ các cột định danh / metadata (giữ lại cột label)
    drop_cols = config.get('drop_early_columns', [])
    drop_cols += [c for c in df.columns if any(p in c for p in ['ip', 'time', 'date', 'mac']) and c != label_col]
    drop_cols = list(set([c for c in drop_cols if c in df.columns and c != label_col]))

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print("Dropped identifier/metadata columns:", drop_cols)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ======================
    # NHÃN VÀ MÃ HÓA
    # ======================
    raw_s = df[label_col].astype(str).str.strip().str.lower()

    if DATASET_KEY == 'signal22':
        # BỎ GOM NHÓM: Giữ nguyên nhãn gốc của Simargl2022
        final_labels = raw_s
    elif DATASET_KEY == 'nsl-kdd':
        final_labels = raw_s.map(NSL_5_CLASSES).fillna('Other')
    elif DATASET_KEY == 'cic17':
    # Loại bỏ ký tự lạ (như ), khoảng trắng thừa và ký tự điều khiển trong chuỗi gốc
        clean_raw = raw_s.str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)
        clean_raw = clean_raw.str.replace(r'\s+', ' ', regex=True).str.strip()
        final_labels = clean_raw.map(CIC_7_CLASSES).fillna('Other')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(final_labels)

    print("\nClasses distribution:")
    for idx, cls in enumerate(label_encoder.classes_):
        count = (y_encoded == idx).sum()
        print(f"  Class {idx}: {cls:<20} ({count:,} samples)")

    df.drop(columns=[label_col], inplace=True)

    # ======================
    # XỬ LÝ CỘT PHÂN LOẠI CATEGORICAL
    # ======================
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        unique_count = df[col].nunique()
        if unique_count <= 50:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype='float32')
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)).astype('int32')

    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')

    # ======================
    # SPLIT X, y
    # ======================
    X = df.astype('float32')
    y = y_encoded.astype(np.int64)

    del df
    gc.collect()

    print("X shape:", X.shape)

    # ======================
    # SPLIT TRAIN / TEST TRƯỚC (NO LEAKAGE)
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    del X
    gc.collect()

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ======================
    # SCALE (FIT CHỈ TRÊN TRAIN)
    # ======================
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # ======================
    # FEATURE SELECTION (FIT CHỈ TRÊN TRAIN)
    # ======================
    k_actual = min(K_FEATURES, X_train.shape[1])
    selector = SelectKBest(chi2, k=k_actual)

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = X_train.columns[selector.get_support()].tolist()
    print("Selected features:", selected_features)

    # ======================
    # SMOTE (CHỈ TRÊN TRAIN)
    # ======================
    unique, counts = np.unique(y_train, return_counts=True)
    min_samples = min(counts)

    if min_samples <= 5:
        k_neighbors = max(1, min_samples - 1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    else:
        smote = SMOTE(random_state=RANDOM_STATE)

    try:
        X_train_res, y_train_res = smote.fit_resample(X_train_selected, y_train)
        print("SMOTE applied successfully.")
    except Exception as e:
        print(f"⚠️ Bỏ qua SMOTE do lỗi: {e}")
        X_train_res, y_train_res = X_train_selected, y_train

    # ======================
    # SAVE OUTPUTS
    # ======================
    base_path = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)

    np.save(f'{base_path}_X_train.npy', X_train_res.astype(np.float32))
    np.save(f'{base_path}_X_test.npy', X_test_selected.astype(np.float32))
    np.save(f'{base_path}_y_train.npy', y_train_res.astype(np.int64))
    np.save(f'{base_path}_y_test.npy', y_test.astype(np.int64))

    joblib.dump(label_encoder, f'{base_path}_label_encoder.pkl')
    joblib.dump(scaler, f'{base_path}_scaler.pkl')
    joblib.dump(selector, f'{base_path}_selector.pkl')
    joblib.dump(selected_features, f'{base_path}_feature_names.pkl')

    print(f"\n✅ DONE MULTI-CLASS DATASET: {DATASET_KEY.upper()}")
    print("Saved to:", OUTPUT_DIR)

print("\n🔥 HOÀN TẤT TOÀN BỘ PROJECT!")
