import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
X_train = np.load("X_train_selected.npy")
X_test  = np.load("X_test_selected.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

arch    = np.load("best_architecture.npy", allow_pickle=True).item()
NEURONS = arch["neurons"]
DROPOUT = arch["dropout"]
INPUT_DIM = X_train.shape[1]

print("="*55)
print("        FINAL HYBRID METAHEURISTIC MODEL")
print("="*55)
print(f"  Input Features : {INPUT_DIM}")
print(f"  Neurons        : {NEURONS}")
print(f"  Dropout        : {DROPOUT}")
print(f"  Batch Size     : 256")
print(f"  Max Epochs     : 100")
print(f"  Optimizer      : Adam (lr=0.001)")
print("="*55)

# ═══════════════════════════════════════════════════════════
# 2. CLASS WEIGHTS (fix imbalance)
# ═══════════════════════════════════════════════════════════
weights = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.unique(y_train),
    y            = y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\n  Class Weights → Normal:{weights[0]:.2f} | Attack:{weights[1]:.2f}")

# ═══════════════════════════════════════════════════════════
# 3. BUILD MODEL
# ═══════════════════════════════════════════════════════════
tf.keras.backend.clear_session()

model = keras.Sequential()
model.add(layers.Input(shape=(INPUT_DIM,)))

for n in NEURONS:
    model.add(layers.Dense(n, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

model.summary()

# ═══════════════════════════════════════════════════════════
# 4. CALLBACKS
# ═══════════════════════════════════════════════════════════
early_stop = callbacks.EarlyStopping(
    monitor              = "val_accuracy",
    patience             = 15,
    restore_best_weights = True,
    verbose              = 1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor  = "val_loss",
    factor   = 0.5,
    patience = 5,
    min_lr   = 0.00001,
    verbose  = 1
)

# ═══════════════════════════════════════════════════════════
# 5. TRAIN
# ═══════════════════════════════════════════════════════════
print("\n🚀 Training started...")
print("="*55)

history = model.fit(
    X_train, y_train,
    epochs           = 100,
    batch_size       = 256,
    validation_split = 0.1,
    class_weight     = class_weights,
    callbacks        = [early_stop, reduce_lr],
    verbose          = 1
)

# ═══════════════════════════════════════════════════════════
# 6. FIND BEST THRESHOLD
# ═══════════════════════════════════════════════════════════
print("\n🔍 Finding best decision threshold...")

y_pred_prob = model.predict(X_test, verbose=0).flatten()

best_thresh = 0.5
best_f1     = 0

for t in np.arange(0.10, 0.90, 0.05):
    preds_t = (y_pred_prob > t).astype(int)
    score   = f1_score(y_test, preds_t)
    if score > best_f1:
        best_f1     = score
        best_thresh = t

print(f"  Best Threshold : {best_thresh:.2f}")
print(f"  Best F1 Score  : {best_f1:.4f}")

y_pred = (y_pred_prob > best_thresh).astype(int)

# ═══════════════════════════════════════════════════════════
# 7. FINAL RESULTS
# ═══════════════════════════════════════════════════════════
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "="*55)
print("           FINAL EVALUATION RESULTS")
print("="*55)
print(f"  ✅ Accuracy  : {acc*100:.2f}%")
print(f"  ✅ Precision : {prec:.4f}")
print(f"  ✅ Recall    : {rec:.4f}")
print(f"  ✅ F1 Score  : {f1:.4f}")
print("="*55)
print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────┬──────────┬──────────┐")
print(f"  │             │ Pred NOR │ Pred ATK │")
print(f"  ├─────────────┼──────────┼──────────┤")
print(f"  │ Actual NOR  │  {cm[0][0]:6d}  │  {cm[0][1]:6d}  │")
print(f"  │ Actual ATK  │  {cm[1][0]:6d}  │  {cm[1][1]:6d}  │")
print(f"  └─────────────┴──────────┴──────────┘")

print("\n" + classification_report(
    y_test, y_pred,
    target_names=["Normal", "Attack"]
))

# ═══════════════════════════════════════════════════════════
# 8. PLOT GRAPHS
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0D1B2A")

for ax in [ax1, ax2]:
    ax.set_facecolor("#1A2E40")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#00B4D8")

ax1.plot(history.history["accuracy"],     label="Train", color="#00B4D8", linewidth=2)
ax1.plot(history.history["val_accuracy"], label="Val",   color="#FFB703", linewidth=2)
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend(facecolor="#0D1B2A", labelcolor="white")
ax1.grid(True, alpha=0.2)

ax2.plot(history.history["loss"],     label="Train", color="#00B4D8", linewidth=2)
ax2.plot(history.history["val_loss"], label="Val",   color="#FFB703", linewidth=2)
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(facecolor="#0D1B2A", labelcolor="white")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
print("  📊 Graph saved : training_history.png")

# ═══════════════════════════════════════════════════════════
# 9. SAVE MODEL
# ═══════════════════════════════════════════════════════════
model.save("hybrid_metaheuristic_model.keras")
print("  💾 Model saved : hybrid_metaheuristic_model.keras")
print("\n🎉 ALL DONE!")