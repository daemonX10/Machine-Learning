# Keras Functional Model — Non-Linear Neural Networks

## Sequential vs Functional API

| Feature | Sequential API | Functional API |
|---|---|---|
| Topology | **Linear** (layer-by-layer) | **Non-linear** (arbitrary graph) |
| Inputs | Single | Single or **Multiple** |
| Outputs | Single | Single or **Multiple** |
| Layer sharing | Not supported | Supported |
| Branching / Merging | Not possible | Fully supported |
| Use case | Simple feed-forward models | Complex architectures |

---

## When to Use Functional API

### Example 1: Multi-Output — Face Analysis

**Task:** Given a face image, predict:
1. **Age** (regression)
2. **Emotion** — happy, sad, angry (classification)

**Architecture:**

```
Image Input → CNN → Flatten
                      ├── Dense → Age Output      (regression, linear activation)
                      └── Dense → Emotion Output   (classification, softmax activation)
```

> Cannot be built with Sequential API because of **branching** at the output.

### Example 2: Multi-Input — E-commerce Price Prediction

**Task:** Given product metadata (tabular), description (text), and photo (image), predict the **price**.

**Architecture:**

```
Tabular Data → ANN  ─┐
Text Data    → RNN  ─┤→ Concatenate → Dense → Price Output
Image Data   → CNN  ─┘
```

> Cannot be built with Sequential API because of **multiple inputs merging**.

---

## Building a Functional Model — Step by Step

### Imports

```python
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.utils import plot_model
```

### Step 1: Define Input

```python
input_layer = Input(shape=(3,))   # 3 input features
```

### Step 2: Build Hidden Layers (with explicit connections)

```python
hidden1 = Dense(128, activation='relu')(input_layer)    # connected to input
hidden2 = Dense(64, activation='relu')(hidden1)          # connected to hidden1
```

### Step 3: Create Branches for Multiple Outputs

```python
# Branch 1: Age prediction (regression)
output_age = Dense(1, activation='linear', name='age_output')(hidden2)

# Branch 2: Place prediction (classification)
output_place = Dense(2, activation='softmax', name='place_output')(hidden2)
```

### Step 4: Create Model Object

```python
model = Model(
    inputs=input_layer,
    outputs=[output_age, output_place]
)
```

### Visualize

```python
model.summary()
plot_model(model, show_shapes=True)
```

---

## Multi-Input Architecture Example

### Code

```python
# Two separate inputs
input_a = Input(shape=(32,), name='input_a')
input_b = Input(shape=(16,), name='input_b')

# Branch A
x = Dense(8, activation='relu')(input_a)
x1 = Dense(4, activation='relu')(x)

# Branch B
y = Dense(8, activation='relu')(input_b)
y1 = Dense(4, activation='relu')(y)
y2 = Dense(4, activation='relu')(y1)

# Concatenate branches
combined = Concatenate()([x1, y2])

# Shared layers after merge
z = Dense(8, activation='relu')(combined)
z1 = Dense(1, activation='sigmoid')(z)

# Build model
model = Model(inputs=[input_a, input_b], outputs=z1)
```

**Key operation:** `Concatenate()` merges feature vectors from different branches.

---

## Practical Example: Face Age + Gender Prediction

### Strategy
1. Use **VGG16** (pretrained, transfer learning) as feature extractor
2. Freeze convolutional base (`trainable = False`)
3. Add **Flatten** layer
4. Branch into two outputs: **Age** (regression) and **Gender** (classification)

### Code

```python
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten

# Load pretrained VGG16 without top layers
vgg_base = VGG16(include_top=False, input_shape=(224, 224, 3))
vgg_base.trainable = False

# Extract last conv layer output
x = vgg_base.layers[-1].output

# Flatten
flat = Flatten()(x)

# Branch 1: Age prediction
dense1 = Dense(128, activation='relu', name='age_dense1')(flat)
dense2 = Dense(64, activation='relu', name='age_dense2')(dense1)
output_age = Dense(1, activation='linear', name='age_output')(dense2)

# Branch 2: Gender prediction
dense3 = Dense(128, activation='relu', name='gender_dense1')(flat)
dense4 = Dense(64, activation='relu', name='gender_dense2')(dense3)
output_gender = Dense(1, activation='sigmoid', name='gender_output')(dense4)

# Create model
model = Model(inputs=vgg_base.input, outputs=[output_age, output_gender])
```

### Compiling with Multiple Outputs

```python
model.compile(
    optimizer='adam',
    loss={
        'age_output': 'mse',             # regression loss
        'gender_output': 'binary_crossentropy'  # classification loss
    },
    loss_weights={
        'age_output': 0.5,     # weight age loss less
        'gender_output': 1.0   # weight gender loss more
    },
    metrics=['accuracy']
)
```

> Use **dictionaries** to specify different loss functions and weights per output.

### Data Generator Setup

When using multi-output with `ImageDataGenerator`:
- Pass **both** output column names in a list: `y_col=['age', 'gender']`
- Set `class_mode='multi_output'`

### Training

```python
model.fit(train_generator, validation_data=test_generator, epochs=50)
```

---

## Key Concepts Summary

| Concept | Description |
|---|---|
| **Functional API** | Build any neural network graph (non-linear, branching, merging) |
| **Model()** | Takes `inputs` and `outputs` to define the computation graph |
| **Input()** | Declares input tensor with specified shape |
| **Concatenate()** | Merges multiple tensors along a given axis |
| **plot_model()** | Visualizes the architecture as a diagram |
| **Layer naming** | Each layer gets an explicit `name` parameter for clarity |
| **Connection syntax** | `Dense(64, activation='relu')(previous_layer)` — call layer on its input |
| **Multi-output loss** | Use dictionaries to assign different loss functions per output |
| **Loss weights** | Control relative importance of each output during training |
