import numpy as np
import pygad
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data & best architecture ─────────────────────────────────────────
X_train = np.load("X_train_selected.npy")
X_test  = np.load("X_test_selected.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

arch = np.load("best_architecture.npy", allow_pickle=True).item()
NUM_LAYERS  = arch["num_layers"]   # 3
NEURONS     = arch["neurons"]      # [67, 47, 71]
DROPOUT     = arch["dropout"]      # 0.13
INPUT_DIM   = X_train.shape[1]     # 18

print(f"Architecture loaded: {NUM_LAYERS} layers | Neurons: {NEURONS} | Dropout: {DROPOUT}")

# ── 2. Define search space ────────────────────────────────────────────────────
# GA will search over these options using indices

LEARNING_RATES = [0.001, 0.005, 0.01, 0.05]        # gene 0 → index 0-3
BATCH_SIZES    = [64, 128, 256, 512]                # gene 1 → index 0-3
EPOCHS_LIST    = [10, 20, 30, 50]                   # gene 2 → index 0-3
OPTIMIZERS     = ["adam", "rmsprop", "sgd", "adamax"] # gene 3 → index 0-3

def decode(solution):
    lr        = LEARNING_RATES[int(np.clip(round(solution[0]), 0, 3))]
    batch     = BATCH_SIZES   [int(np.clip(round(solution[1]), 0, 3))]
    epochs    = EPOCHS_LIST   [int(np.clip(round(solution[2]), 0, 3))]
    optimizer = OPTIMIZERS    [int(np.clip(round(solution[3]), 0, 3))]
    return lr, batch, epochs, optimizer

# ── 3. Build model with given hyperparameters ─────────────────────────────────
def build_model(lr, optimizer_name):
    model = keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))

    for n in NEURONS:
        model.add(layers.Dense(n, activation="relu"))
        model.add(layers.Dropout(DROPOUT))

    model.add(layers.Dense(1, activation="sigmoid"))

    # Select optimizer
    if optimizer_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer_name == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr)
    else:
        opt = keras.optimizers.Adamax(learning_rate=lr)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ── 4. Fitness function ───────────────────────────────────────────────────────
eval_counter = [0]

def fitness_function(ga_instance, solution, solution_idx):
    eval_counter[0] += 1
    lr, batch, epochs, optimizer = decode(solution)

    print(f"\n  [{eval_counter[0]:3d}] LR:{lr} | Batch:{batch} | Epochs:{epochs} | Opt:{optimizer}", end=" → ")

    model = build_model(lr, optimizer)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch,
        verbose=0,
        validation_split=0.1
    )

    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, preds)

    print(f"Acc: {acc:.4f}")

    # Clear session to free memory between evaluations
    keras.backend.clear_session()

    return acc

# ── 5. GA Callback ────────────────────────────────────────────────────────────
def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_fitness = ga_instance.best_solution()[1]
    best_sol     = ga_instance.best_solution()[0]
    lr, batch, epochs, optimizer = decode(best_sol)
    print(f"\n📊 Gen {gen} Best → Acc:{best_fitness:.4f} | LR:{lr} | Batch:{batch} | Epochs:{epochs} | Opt:{optimizer}")
    print("-"*60)

# ── 6. Run GA ─────────────────────────────────────────────────────────────────
print("\n⚙️  Starting Hyperparameter Optimization with GA...")
print("="*60)

ga_instance = pygad.GA(
    num_generations        = 5,    # 5 generations
    num_parents_mating     = 3,
    fitness_func           = fitness_function,
    sol_per_pop            = 6,    # 6 hyperparameter combos per generation
    num_genes              = 4,    # [lr, batch, epochs, optimizer]
    init_range_low         = 0,
    init_range_high        = 3,
    mutation_percent_genes = 25,
    parent_selection_type  = "rank",
    crossover_type         = "single_point",
    mutation_type          = "random",
    keep_parents           = 1,
    on_generation          = on_generation
)

ga_instance.run()

# ── 7. Best hyperparameters ───────────────────────────────────────────────────
best_solution, best_fitness, _ = ga_instance.best_solution()
best_lr, best_batch, best_epochs, best_optimizer = decode(best_solution)

print("\n" + "="*60)
print(f"✅ Hyperparameter Optimization Complete!")
print(f"   Best Accuracy:   {best_fitness:.4f}")
print(f"   Learning Rate:   {best_lr}")
print(f"   Batch Size:      {best_batch}")
print(f"   Epochs:          {best_epochs}")
print(f"   Optimizer:       {best_optimizer}")

# ── 8. Save best hyperparameters ──────────────────────────────────────────────
hyperparams = {
    "learning_rate" : best_lr,
    "batch_size"    : best_batch,
    "epochs"        : best_epochs,
    "optimizer"     : best_optimizer
}
np.save("best_hyperparams.npy", hyperparams)
print("\n   Saved: best_hyperparams.npy")
print("   Ready for Step 5 — Final Training & Evaluation!")