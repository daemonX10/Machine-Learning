# CNN vs ANN

## How ANN Processes Images

1. Take a 2D image (e.g., $28 \times 28$) and **flatten** it to 1D → 784 inputs
2. Feed into hidden layers (fully connected / dense layers)
3. Output layer with softmax for classification (e.g., 10 digits)
4. All layers are **fully connected** — every neuron connects to every neuron in the next layer

## How CNN Processes Images

1. Keep image in its **original 2D form** (no flattening initially)
2. Apply **filters/kernels** via convolution operation → generates **feature maps**
3. Add **bias** to each feature map
4. Apply **activation function** (e.g., ReLU)
5. Apply **pooling** to downsample
6. Feed into **fully connected layers** at the end for classification

## Similarity Between ANN and CNN

### ANN Neuron Operation

$$z = \sum_{i} w_i \cdot x_i + b$$
$$a = \sigma(z)$$

- Take all inputs, multiply by weights, sum, add bias, apply activation function

### CNN Convolution Operation

- Take a **local patch** of the image (window of pixels)
- Multiply element-wise with filter weights, sum, add bias, apply activation

> **Key Insight**: Both perform **dot product → add bias → activation function**. The fundamental operation is the same!

### Shortcut to Remember

| ANN | CNN |
|-----|-----|
| Neurons | Filters |
| Weights | Filter values |
| Works on **all inputs at once** | Works on a **sliding window** over the input |
| Bias per neuron | Bias per filter |

> **ANN nodes and CNN filters operate similarly** — the basic principle is the same, but CNN captures **2D spatial patterns** through sliding windows.

## Key Differences

### 1. Computational Cost

**ANN**: Number of parameters **depends on input size**

- Input: $28 \times 28 \times 3 = 2{,}352$ neurons
- If hidden layer has $h$ units → weights = $2{,}352 \times h$
- If input grows to $1080 \times 1080 \times 3$ → weights explode into **millions**

**CNN**: Number of parameters **depends only on filter size and number of filters**

| Component | Formula |
|-----------|---------|
| Weights per filter | $F \times F \times C$ (where $C$ = channels) |
| Total weights | $(F \times F \times C) \times K$ (where $K$ = number of filters) |
| Total params | $(F \times F \times C + 1) \times K$ (including bias) |

#### Example Calculation

- Image: $228 \times 228 \times 3$
- Filter: $3 \times 3 \times 3$, with 5 filters
- Parameters per filter: $3 \times 3 \times 3 = 27$ weights + 1 bias = 28
- **Total: $28 \times 5 = 140$ trainable parameters**

> **If you change the image to $1080 \times 1080 \times 3$, parameters remain 140!** This is CNN's biggest advantage.

### 2. Overfitting

- ANN: More parameters → higher chance of **overfitting**
- CNN: Far fewer parameters → **much less prone** to overfitting

### 3. Spatial Feature Preservation

- ANN: Flattening loses 2D spatial arrangement
- CNN: Sliding window operation **preserves spatial relationships** between pixels

## Summary Table

| Aspect | ANN | CNN |
|--------|-----|-----|
| Input format | Flattened 1D vector | Original 2D/3D image |
| Core operation | Matrix multiplication | Convolution (sliding window) |
| Parameters depend on | Input size | Filter size & count |
| Spatial info | Lost during flattening | Preserved |
| Overfitting risk | Higher | Lower |
| Computational cost | Scales with input size | Independent of input size |
| Training speed | Slower for large images | Faster for large images |
