# Lecture 26: Early Stopping in Neural Networks

## 1. What is Early Stopping?

Early stopping is a **regularization technique** that automatically halts the training of a neural network when the model's performance on the validation set starts to **degrade** (i.e., overfitting begins).

> **Core Idea:** Instead of manually deciding the number of epochs, let the algorithm monitor validation loss and stop training at the optimal point.

---

## 2. The Overfitting Problem

| Phase | Training Loss | Validation Loss | Status |
|-------|--------------|----------------|--------|
| Early training | Decreasing | Decreasing | Underfitting → Learning |
| Optimal point | Decreasing | At minimum | Best generalization |
| Continued training | Still decreasing | **Increasing** | **Overfitting** |

- Training too long → model memorizes training data
- Training loss keeps decreasing, but validation loss starts **increasing**
- The divergence point between training and validation loss curves signals overfitting

---

## 3. How Early Stopping Works

```
For each epoch:
    1. Train the model
    2. Evaluate on validation set
    3. Check if monitored quantity improved
        → If improved: continue training
        → If not improved for `patience` epochs: STOP training
```

---

## 4. Keras Implementation

### Without Early Stopping
```python
model.fit(X_train, y_train, epochs=3500, validation_data=(X_test, y_test))
```
- Risk: model overfits after ~300 epochs
- Wastes compute on remaining 3200 epochs

### With Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    mode='auto',
    restore_best_weights=False,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=3500,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
# Training automatically stops at ~327 epochs
```

---

## 5. Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `monitor` | Quantity to monitor | `'val_loss'` (recommended) |
| `min_delta` | Minimum change to qualify as improvement | `0` |
| `patience` | Number of epochs with no improvement before stopping | `5`–`20` |
| `mode` | `'min'` (loss), `'max'` (accuracy), `'auto'` (infers) | `'auto'` |
| `restore_best_weights` | Whether to restore weights from the epoch with the best monitored value | `True` |
| `verbose` | Print early stopping messages | `1` |

### Parameter Details

- **`monitor`**: Usually `val_loss` (validation loss). Can also use `val_accuracy`
- **`patience`**: If `patience=5`, it waits 5 epochs without improvement before stopping. Prevents premature stopping due to temporary fluctuations
- **`mode`**:
  - `'min'` → stop when monitored quantity stops **decreasing** (for loss)
  - `'max'` → stop when monitored quantity stops **increasing** (for accuracy)
  - `'auto'` → automatically infers from the metric name
- **`restore_best_weights`**: If `True`, restores model weights from the epoch with the best value of the monitored metric. Recommended to set `True`

---

## 6. When to Use Early Stopping

- **Always** in real-world projects — let early stopping decide the optimal number of epochs
- Works with any number of max epochs (set a high value and let early stopping handle it)
- Saves computational resources and prevents overfitting
- Acts as a form of **regularization** by limiting model complexity through training duration

---

## 7. Key Takeaways

1. Early stopping monitors **validation performance** during training
2. It prevents overfitting by stopping training before the model starts memorizing
3. The **patience** parameter provides tolerance for temporary dips in performance
4. Use `restore_best_weights=True` to get the best model, not the last one
5. It is implemented via **Keras callbacks** — no changes to model architecture or compilation needed
