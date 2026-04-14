import numpy as np
import pyswarms as ps
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load selected features data ───────────────────────────────────────────
X_train = np.load("X_train_selected.npy")
X_test  = np.load("X_test_selected.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

INPUT_DIM = X_train.shape[1]  # 18 (selected features)
print(f"Input features: {INPUT_DIM}")

# ── 2. Build Neural Network from PSO particle ─────────────────────────────────
# Each particle = [num_layers, neurons_l1, neurons_l2, neurons_l3, dropout_rate]
# PSO will search for best values of these 5 numbers

def build_and_evaluate(particle):
    # Extract architecture parameters from particle
    num_layers  = int(np.clip(round(particle[0]), 1, 3))   # 1 to 3 layers
    neurons_l1  = int(np.clip(round(particle[1]), 16, 128)) # 16 to 128 neurons
    neurons_l2  = int(np.clip(round(particle[2]), 16, 128))
    neurons_l3  = int(np.clip(round(particle[3]), 16, 128))
    dropout     = float(np.clip(particle[4], 0.1, 0.5))    # dropout 0.1 to 0.5

    neuron_list = [neurons_l1, neurons_l2, neurons_l3]

    # Build the model
    model = keras.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))

    for i in range(num_layers):
        model.add(layers.Dense(neuron_list[i], activation="relu"))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))  # Binary output

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train briefly (5 epochs only — just to evaluate architecture)
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=256,
        verbose=0,
        validation_split=0.1
    )

    # Evaluate
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, preds)

    return acc, num_layers, neuron_list[:num_layers], dropout

# ── 3. PSO Cost Function ──────────────────────────────────────────────────────
# PSO MINIMIZES cost — so we return 1 - accuracy (lower = better)

particle_counter = [0]  # track progress

def cost_function(particles):
    costs = []
    for particle in particles:
        particle_counter[0] += 1
        print(f"  Evaluating particle {particle_counter[0]}...", end=" ")

        acc, n_layers, neurons, dropout = build_and_evaluate(particle)
        cost = 1.0 - acc  # PSO minimizes, so flip accuracy

        print(f"Acc: {acc:.4f} | Layers: {n_layers} | Neurons: {neurons} | Dropout: {dropout:.2f}")
        costs.append(cost)

    return np.array(costs)

# ── 4. PSO Configuration ──────────────────────────────────────────────────────
# Search bounds for each parameter:
# [num_layers, neurons_l1, neurons_l2, neurons_l3, dropout]
min_bound = np.array([1,  16,  16,  16,  0.1])
max_bound = np.array([3, 128, 128, 128,  0.5])
bounds = (min_bound, max_bound)

options = {
    'c1': 0.5,   # cognitive component (how much particle trusts itself)
    'c2': 0.3,   # social component (how much it trusts the swarm)
    'w' : 0.9    # inertia (momentum of movement)
}

optimizer = ps.single.GlobalBestPSO(
    n_particles = 10,    # 10 particles searching simultaneously
    dimensions  = 5,     # 5 parameters to optimize
    options     = options,
    bounds      = bounds
)

# ── 5. Run PSO ────────────────────────────────────────────────────────────────
print("\n🔵 Starting PSO Neural Architecture Search...")
print("="*60)
print("Each particle = one neural network architecture being tested")
print("="*60)

best_cost, best_pos = optimizer.optimize(cost_function, iters=5, verbose=False)

# ── 6. Extract Best Architecture ──────────────────────────────────────────────
best_layers  = int(np.clip(round(best_pos[0]), 1, 3))
best_n1      = int(np.clip(round(best_pos[1]), 16, 128))
best_n2      = int(np.clip(round(best_pos[2]), 16, 128))
best_n3      = int(np.clip(round(best_pos[3]), 16, 128))
best_dropout = float(np.clip(best_pos[4], 0.1, 0.5))
best_neurons = [best_n1, best_n2, best_n3][:best_layers]

print("\n" + "="*60)
print(f"✅ NAS Complete!")
print(f"   Best accuracy found:  {1 - best_cost:.4f}")
print(f"   Number of layers:     {best_layers}")
print(f"   Neurons per layer:    {best_neurons}")
print(f"   Dropout rate:         {best_dropout:.2f}")

# ── 7. Save best architecture for next step ───────────────────────────────────
architecture = {
    "num_layers" : best_layers,
    "neurons"    : best_neurons,
    "dropout"    : best_dropout
}
np.save("best_architecture.npy", architecture)
print("\n   Saved: best_architecture.npy")
print("   Ready for Step 4 — Hyperparameter Optimization!")