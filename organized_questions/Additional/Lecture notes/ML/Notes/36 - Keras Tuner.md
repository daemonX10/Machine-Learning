# Keras Tuner — Hyperparameter Tuning for Neural Networks

## Why Hyperparameter Tuning?

When building a neural network, many decisions are made manually:
- Number of hidden layers
- Neurons per layer
- Activation function
- Optimizer
- Loss function
- Batch size
- Dropout rate

**Problem:** No way to know the best values without trying many combinations.

**Solution:** Automate the search using **Keras Tuner**.

---

## Installation

```bash
pip install keras-tuner
```

```python
import keras_tuner as kt
```

---

## How Keras Tuner Works

1. **Define a model-building function** that accepts an `hp` (hyperparameter) object
2. **Create a Tuner object** (e.g., `RandomSearch`) with the function
3. **Run `tuner.search()`** — it builds multiple models with different hyperparameters and evaluates each
4. **Extract best hyperparameters and model**

---

## Hyperparameter Types

| Method | Use Case | Example |
|---|---|---|
| `hp.Choice(name, values)` | Choose from a list of discrete options | Optimizer, activation function |
| `hp.Int(name, min, max, step)` | Integer in a range | Number of neurons, number of layers |
| `hp.Float(name, min, max, step)` | Float in a range | Learning rate, dropout rate |
| `hp.Boolean(name)` | True/False toggle | Whether to include a layer |

---

## Example 1: Selecting Best Optimizer

### Build Function

```python
def build_model(hp):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=8))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Create Tuner & Search

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5
)

tuner.search(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)
```

### Get Best Results

```python
# Best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)  # e.g., {'optimizer': 'rmsprop'}

# Best model (already trained)
model = tuner.get_best_models(num_models=1)[0]
```

### Continue Training the Best Model

```python
model.fit(
    X_train, y_train,
    epochs=100,
    initial_epoch=5,   # continue from where tuner left off
    validation_data=(X_test, y_test)
)
```

---

## Example 2: Selecting Number of Neurons

### Build Function

```python
def build_model(hp):
    model = Sequential()

    units = hp.Int('units', min_value=8, max_value=128, step=8)

    model.add(Dense(units=units, activation='relu', input_dim=8))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

The tuner will try: 8, 16, 24, 32, …, 128 neurons and pick the best.

---

## Example 3: Selecting Number of Layers

### Build Function with Dynamic Layers

```python
def build_model(hp):
    model = Sequential()

    for i in range(hp.Int('num_layers', min_value=1, max_value=10, step=1)):
        if i == 0:
            model.add(Dense(72, activation='relu', input_dim=8))
        else:
            model.add(Dense(72, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

## Example 4: All-in-One — Tuning Everything

### Build Function

```python
def build_model(hp):
    model = Sequential()

    for i in range(hp.Int('num_layers', min_value=1, max_value=10, step=1)):
        # Tune neurons per layer
        units = hp.Int(f'units_{i}', min_value=8, max_value=128, step=8)

        # Tune activation per layer
        activation = hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])

        if i == 0:
            model.add(Dense(units=units, activation=activation, input_dim=8))
        else:
            model.add(Dense(units=units, activation=activation))

        # Tune dropout per layer
        dropout_rate = hp.Choice(f'dropout_{i}',
            values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    # Tune optimizer
    optimizer = hp.Choice('optimizer',
        values=['adam', 'sgd', 'rmsprop', 'adadelta', 'nadam'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

## Tuner Configuration

### RandomSearch Tuner

```python
tuner = kt.RandomSearch(
    build_model,                    # model-building function
    objective='val_accuracy',       # metric to optimize
    max_trials=5,                   # number of different configs to try
    directory='my_dir',             # where to save trial data
    project_name='my_project'       # subfolder name
)
```

### Running the Search

```python
tuner.search(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)
```

### Extracting Results

```python
# Best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
# e.g., {'num_layers': 7, 'units_0': 120, 'activation_0': 'relu',
#         'dropout_0': 0.1, 'optimizer': 'rmsprop', ...}

# Best model
model = tuner.get_best_models(num_models=1)[0]
model.summary()
```

---

## Project Directory Structure

When you specify `directory` and `project_name`, Keras Tuner saves:

```
my_dir/
└── my_project/
    ├── trial_0/
    │   └── trial data (hyperparams, metrics, checkpoints)
    ├── trial_1/
    ├── trial_2/
    └── ...
```

Each trial folder contains full information about that configuration's hyperparameters and results.

---

## Workflow Summary

```
1. Prepare Data
   └── Scale features, train/test split

2. Define build_model(hp)
   └── Use hp.Choice(), hp.Int(), hp.Float() for tunable params

3. Create Tuner
   └── kt.RandomSearch(build_model, objective, max_trials)

4. Run tuner.search(X_train, y_train, ...)

5. Extract Results
   ├── tuner.get_best_hyperparameters()
   └── tuner.get_best_models()

6. Retrain Best Model
   └── model.fit(..., initial_epoch=N, epochs=M)
```

---

## Key API Reference

| Method / Class | Purpose |
|---|---|
| `kt.RandomSearch` | Random search over hyperparameter space (like `RandomizedSearchCV` in sklearn) |
| `hp.Choice(name, values)` | Pick from a predefined list |
| `hp.Int(name, min, max, step)` | Pick an integer from a range |
| `hp.Float(name, min, max, step)` | Pick a float from a range |
| `tuner.search()` | Execute the hyperparameter search |
| `tuner.get_best_hyperparameters()` | Returns list of best HP configs |
| `tuner.get_best_models(num_models=N)` | Returns top N trained models |
| `tuner.results_summary()` | Print summary of all trials |

---

## Tips

- **Start simple:** Tune one hyperparameter at a time to understand its effect, then combine
- **Use `initial_epoch`:** When retraining the best model, set `initial_epoch` to continue from where tuner stopped — avoids losing learned weights
- **Epochs in search can be low** (e.g., 5–10) — enough to differentiate good from bad configs
- **Final training epochs should be high** (e.g., 100+) with Early Stopping
- **Dropout** is tuned per-layer just like neurons and activation functions
- **`max_trials`** controls compute budget — higher = better search but slower
