import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# ── 1. Column names ──────────────────────────────────────────────────────────
col_names = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# ── 2. Load dataset (Yahan file path check karein) ──────────────────────────
# Kaggle se download karne ke baad agar file folder ke andar hai toh r'folder_name/filename' likhein
train_file = "KDDTrain+.txt" 
test_file = "KDDTest+.txt"

# Safe loading: Check if files exist
if os.path.exists(train_file) and os.path.exists(test_file):
    train_df = pd.read_csv(train_file, header=None, names=col_names)
    test_df  = pd.read_csv(test_file,  header=None, names=col_names)
else:
    print("❌ Error: Files nahi mili! Make sure 'KDDTrain+.txt' aapke script ke folder mein hai.")
    # Agar error aaye toh file ka sahi naam yahan check karke badlein

# Drop 'difficulty' column
train_df.drop("difficulty", axis=1, errors='ignore', inplace=True)
test_df.drop("difficulty",  axis=1, errors='ignore', inplace=True)

# ── 3. Label Transformation ──────────────────────────────────────────────────
# Kaggle files mein kabhi labels ke peeche '.' hota hai (e.g., 'normal.')
train_df["label"] = train_df["label"].str.replace('.', '', regex=False)
test_df["label"]  = test_df["label"].str.replace('.', '', regex=False)

def to_binary(label):
    return 0 if str(label).lower() == "normal" else 1

train_df["label"] = train_df["label"].apply(to_binary)
test_df["label"]  = test_df["label"].apply(to_binary)

# ── 4. Encoding & Scaling (Same as before) ───────────────────────────────────
categorical_cols = ["protocol_type", "service", "flag"]
encoder = LabelEncoder()

for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col]  = encoder.transform(test_df[col])

X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values
X_test  = test_df.drop("label", axis=1).values
y_test  = test_df["label"].values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Save ───────────────────────────────────────────────────────────────────
np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)

print("\n✅ Kaggle Data Preprocessed Successfully!")