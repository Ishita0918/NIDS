import numpy as np
import pygad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

NUM_FEATURES = X_train.shape[1]  # 41
print(f"Total features before selection: {NUM_FEATURES}")

# ── 2. Fitness Function ───────────────────────────────────────────────────────
# This is the "judge" — it scores each feature subset
# GA will try to MAXIMIZE this score

def fitness_function(ga_instance, solution, solution_idx):
    # 'solution' is a list of 41 values — each either ~0 or ~1
    # We round them to get a binary mask: 1 = keep feature, 0 = drop it
    selected = np.where(np.round(solution) == 1)[0]

    # Edge case: if no features selected, return 0 score
    if len(selected) == 0:
        return 0.0

    # Use only selected features
    X_tr = X_train[:, selected]
    X_te = X_test[:, selected]

    # Train a quick Random Forest to evaluate this feature subset
    # We use a small model here just for speed (n_estimators=10)
    clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)

    accuracy = accuracy_score(y_test, preds)

    # Bonus: penalize using too many features (we want a small, accurate subset)
    # This encourages the GA to find minimal but effective features
    penalty = 0.001 * len(selected)

    return accuracy - penalty

# ── 3. Callback — prints progress every generation ────────────────────────────
def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_fitness = ga_instance.best_solution()[1]
    best_solution = ga_instance.best_solution()[0]
    num_selected = int(np.sum(np.round(best_solution)))
    print(f"Gen {gen:3d} | Best Fitness: {best_fitness:.4f} | Features selected: {num_selected}")

# ── 4. Configure the Genetic Algorithm ───────────────────────────────────────
ga_instance = pygad.GA(
    num_generations      = 20,          # How many rounds of evolution
    num_parents_mating   = 5,           # How many top solutions breed each round
    fitness_func         = fitness_function,
    sol_per_pop          = 20,          # Population size (20 different feature subsets)
    num_genes            = NUM_FEATURES,# Each "gene" = 1 feature (41 total)
    init_range_low       = 0,           # Gene values start between 0 and 1
    init_range_high      = 1,
    mutation_percent_genes = 10,        # 10% of genes randomly flip each generation
    parent_selection_type  = "rank",    # Select parents by ranking
    crossover_type         = "single_point",
    mutation_type          = "random",
    keep_parents           = 2,         # Keep top 2 solutions unchanged each gen
    on_generation          = on_generation
)

# ── 5. Run the GA ─────────────────────────────────────────────────────────────
print("\n🧬 Starting Genetic Algorithm for Feature Selection...")
print("="*60)
ga_instance.run()

# ── 6. Extract Results ────────────────────────────────────────────────────────
best_solution, best_fitness, _ = ga_instance.best_solution()
selected_features = np.where(np.round(best_solution) == 1)[0]

print("\n" + "="*60)
print(f"✅ Feature Selection Complete!")
print(f"   Features selected: {len(selected_features)} out of {NUM_FEATURES}")
print(f"   Best fitness score: {best_fitness:.4f}")
print(f"   Selected feature indices: {selected_features}")

# ── 7. Apply selection and save ───────────────────────────────────────────────
X_train_selected = X_train[:, selected_features]
X_test_selected  = X_test[:, selected_features]

np.save("X_train_selected.npy", X_train_selected)
np.save("X_test_selected.npy",  X_test_selected)
np.save("selected_features.npy", selected_features)

print(f"\n   New X_train shape: {X_train_selected.shape}")
print(f"   New X_test shape:  {X_test_selected.shape}")
print("   Files saved: X_train_selected.npy, X_test_selected.npy")