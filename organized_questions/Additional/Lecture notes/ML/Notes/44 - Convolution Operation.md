# Convolution Operation

## CNN Architecture Recap

Three types of layers:
1. **Convolution Layer** — extracts features (topic of this note)
2. **Pooling Layer** — downsamples feature maps
3. **Fully Connected (Dense) Layer** — classification

### High-Level Workflow

1. Early convolution layers detect **primitive features** (edges, lines)
2. Deeper layers combine these into **complex features** (eyes, nose → face)
3. Fully connected layers perform final **classification**

## Image Representation in Memory

### Grayscale Images

- A 2D array (grid) of pixel values
- Each pixel: value between **0** (black) and **255** (white)
- Example: MNIST digit → $28 \times 28$ pixels
- Shape: $(H, W)$ = $(28, 28)$

### RGB (Color) Images

- Three channels: **Red, Green, Blue**
- Each channel is like a separate grayscale image
- Shape: $(H, W, C)$ = $(28, 28, 3)$

| Type | Channels | Pixel Range | Shape Example |
|------|----------|-------------|---------------|
| Grayscale | 1 | 0–255 | $(28, 28)$ |
| RGB | 3 | 0–255 per channel | $(28, 28, 3)$ |

## What is Convolution?

Convolution is the **fundamental building block** of CNN. Its job: **extract features** from an image.

### Components

| Component | Description |
|-----------|-------------|
| **Input Image** | The 2D/3D pixel array |
| **Filter (Kernel)** | A small matrix (usually odd-sized: $3\times3$, $5\times5$) |
| **Feature Map** | The output of the convolution operation |

### How Convolution Works (Grayscale)

1. Place the filter on the **top-left corner** of the image
2. Perform **element-wise multiplication** between overlapping values
3. **Sum** all the products → one output number
4. **Slide** the filter one position to the right and repeat
5. When the row is done, move down one row and repeat
6. Continue until the entire image is covered

#### Example: $6 \times 6$ Image with $3 \times 3$ Filter

```
Image (6×6):                Filter (3×3):
0   0   0   0   0   0      -1  -1  -1
0   0   0   0   0   0       0   0   0
0   0   0   0   0   0       1   1   1
255 255 255 255 255 255
255 255 255 255 255 255
255 255 255 255 255 255
```

This filter is a **horizontal edge detector**:
- Top row: $-1$ values (negates dark pixels)
- Middle row: $0$ (ignores)
- Bottom row: $+1$ (keeps bright pixels)

**Result**: Feature map highlights where horizontal intensity changes occur (edges).

### Edge Detection

> **Edges** = sudden changes in pixel intensity

| Filter Type | Detects | Kernel Example |
|-------------|---------|----------------|
| Horizontal edge | Horizontal intensity changes | $\begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$ |
| Vertical edge | Vertical intensity changes | $\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$ |

## Output Size Formula

For an $N \times N$ image convolved with an $F \times F$ filter:

$$\text{Feature Map Size} = (N - F + 1) \times (N - F + 1)$$

**Example**: $28 \times 28$ image, $3 \times 3$ filter → $(28 - 3 + 1) = 26 \times 26$

## Filter Values Are Learned

- In deep learning, **you do not manually design filters**
- Filter values are **initialized randomly**
- During training, **backpropagation** automatically learns optimal filter values
- Filters are analogous to **weights in ANN** — they are trainable parameters

> This is what makes CNNs so powerful: they don't rely on pre-defined filters but **learn task-specific filters** from data.

## ReLU After Convolution

After computing the feature map:
- Apply **ReLU activation**: $f(x) = \max(0, x)$
- All **negative values → 0**, positive values unchanged
- Introduces **non-linearity**
- Retains only **positive activations** (features that were actually detected)

## Convolution on RGB (Color) Images

For an RGB image with 3 channels:
- The filter also has **3 channels** (one per R, G, B)
- Filter shape: $F \times F \times 3$

### Process

1. Align the 3D filter over a patch of the 3D image
2. Element-wise multiply **all 27 values** (for a $3 \times 3 \times 3$ filter)
3. Sum all products → **one single number**
4. Slide and repeat

> **Result**: An $N \times N \times 3$ image convolved with an $F \times F \times 3$ filter produces a **single-channel** $(N-F+1) \times (N-F+1)$ feature map.

## Multiple Filters

In practice, **multiple filters** are always applied:

- Each filter produces one feature map
- $K$ filters → $K$ feature maps
- These are stacked into a **volume** (3D tensor)

$$\text{Output Shape} = (N - F + 1) \times (N - F + 1) \times K$$

| Filters Used | Output Feature Maps |
|-------------|-------------------|
| 1 | $(N-F+1) \times (N-F+1) \times 1$ |
| 2 | $(N-F+1) \times (N-F+1) \times 2$ |
| 10 | $(N-F+1) \times (N-F+1) \times 10$ |
| $K$ | $(N-F+1) \times (N-F+1) \times K$ |

> The **number of channels in the output** = **number of filters used**. This output volume serves as **input to the next convolution layer**.
