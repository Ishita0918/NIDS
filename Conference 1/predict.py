import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 1. LOAD MODEL & SAVED DATA
# ═══════════════════════════════════════════════════════════
print("="*55)
print("   HYBRID METAHEURISTIC - INTRUSION DETECTOR")
print("="*55)

model            = keras.models.load_model("hybrid_metaheuristic_model.keras")
selected_features = np.load("selected_features.npy")

print(f"✅ Model loaded!")
print(f"✅ Using {len(selected_features)} selected features")

# ═══════════════════════════════════════════════════════════
# 2. COLUMN NAMES
# ═══════════════════════════════════════════════════════════
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
feature_cols = col_names[:41]  # first 41 = features only

# ═══════════════════════════════════════════════════════════
# 3. PREPROCESS FUNCTION
# ═══════════════════════════════════════════════════════════
def preprocess_input(df):
    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    encoder = LabelEncoder()

    # Load original train data to fit encoder on same categories
    train_df = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    test_df  = pd.read_csv("KDDTest+.txt",  header=None, names=col_names)

    for col in categorical_cols:
        combined = pd.concat([train_df[col], test_df[col], df[col]])
        encoder.fit(combined)
        df[col] = encoder.transform(df[col])

    # Scale
    X = df[feature_cols].values

    # Load original train for scaler
    train_full = train_df[feature_cols].copy()
    for col in categorical_cols:
        combined = pd.concat([train_df[col], test_df[col]])
        encoder.fit(combined)
        train_full[col] = encoder.transform(train_df[col])

    scaler = MinMaxScaler()
    scaler.fit(train_full.values)
    X_scaled = scaler.transform(X)

    # Select only GA-chosen features
    X_selected = X_scaled[:, selected_features]
    return X_selected

# ═══════════════════════════════════════════════════════════
# 4. PREDICT FUNCTION
# ═══════════════════════════════════════════════════════════
def predict_traffic(df):
    X = preprocess_input(df)
    probs = model.predict(X, verbose=0).flatten()

    results = []
    for i, prob in enumerate(probs):
        label     = "🔴 ATTACK"  if prob > 0.4 else "🟢 NORMAL"
        confidence = prob * 100  if prob > 0.4 else (1 - prob) * 100
        results.append({
            "Connection" : i + 1,
            "Result"     : label,
            "Confidence" : f"{confidence:.1f}%",
            "Raw Score"  : f"{prob:.4f}"
        })

    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════
# 5. DEMO — Test on sample connections from test file
# ═══════════════════════════════════════════════════════════
print("\n📡 Testing on 10 sample network connections...\n")

test_df = pd.read_csv("KDDTest+.txt", header=None, names=col_names)
test_df = test_df.drop(["label", "difficulty"], axis=1)

# Pick 10 random samples
sample = test_df.sample(10, random_state=42).reset_index(drop=True)

results = predict_traffic(sample)

print(results.to_string(index=False))

print("\n" + "="*55)
print("  HOW TO USE ON YOUR OWN DATA:")
print("="*55)
print("""
  1. Put your network data in same format as NSL-KDD
  2. Load it:
       df = pd.read_csv("your_data.csv", header=None, names=feature_cols)
  3. Run prediction:
       results = predict_traffic(df)
       print(results)
""")
print("="*55)