# Padding and Strides in CNN

## Why Do We Need Padding?

Two problems with standard convolution:

### Problem 1: Shrinking Output (Information Loss)

- Each convolution layer **reduces** the feature map size
- Formula: $(N - F + 1)$ — output is always smaller than input
- Stacking multiple convolution layers → image keeps shrinking → **progressive information loss**

**Example**: $5 \times 5$ image with $3 \times 3$ filter → $3 \times 3$ output. Apply another $3 \times 3$ filter → $1 \times 1$ output!

### Problem 2: Border Pixels Are Underrepresented

- **Corner/border pixels** participate in fewer convolution operations
- **Center pixels** participate in many more operations
- Feature map is **biased toward center** — if important information is at the edges, it gets filtered out

## What is Padding?

**Padding** = adding extra rows and columns (typically zeros) around the border of the image **before** convolution.

### Zero Padding

- Surround the image with rows/columns of **zeros**
- Called **zero padding** because the added values are all zero

**Example**: $5 \times 5$ image with padding $P = 1$ → becomes $7 \times 7$

```
Original (5×5):        After padding P=1 (7×7):
x x x x x             0 0 0 0 0 0 0
x x x x x             0 x x x x x 0
x x x x x             0 x x x x x 0
x x x x x             0 x x x x x 0
x x x x x             0 x x x x x 0
                       0 0 0 0 0 0 0
                       0 0 0 0 0 0 0
```

## Output Size Formula with Padding

$$\text{Output Size} = (N + 2P - F + 1)$$

Where:
- $N$ = input size
- $P$ = padding amount
- $F$ = filter size

**Example**: $N=5$, $P=1$, $F=3$ → $(5 + 2(1) - 3 + 1) = 5$ ✓ Output = Input size!

## Types of Padding

| Type | Description | Effect |
|------|-------------|--------|
| **Valid (No Padding)** | $P = 0$; no padding applied | Output shrinks: $(N - F + 1)$ |
| **Same Padding** | Padding chosen so output size = input size | Keras auto-calculates $P$ |

### Keras Implementation

```python
# Valid padding (default)
Conv2D(32, (3,3), padding='valid', activation='relu', input_shape=(28,28,1))

# Same padding
Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1))
```

**With valid padding**: $28 \to 26 \to 24 \to 22$ (shrinks each layer)

**With same padding**: $28 \to 28 \to 28 \to 28$ (size preserved!)

## What are Strides?

**Stride** = the number of pixels the filter moves in each step during convolution.

- Default stride = 1 (filter shifts one pixel at a time)
- Stride = 2 → filter jumps 2 pixels at a time

### How Stride Works

With **stride = 1** (default):
```
Position 1: [0,0]  →  Position 2: [0,1]  →  Position 3: [0,2]  → ...
Then next row: [1,0]  →  [1,1]  →  ...
```

With **stride = 2**:
```
Position 1: [0,0]  →  Position 2: [0,2]  →  Position 3: [0,4]  → ...
Then jump 2 rows: [2,0]  →  [2,2]  →  ...
```

> Larger stride → filter skips more pixels → **smaller output** → less information captured

## Output Size Formula with Strides

$$\text{Output Size} = \left\lfloor \frac{N - F}{S} \right\rfloor + 1$$

Where $S$ = stride.

## Complete Formula (Padding + Strides)

$$\text{Output Size} = \left\lfloor \frac{N + 2P - F}{S} \right\rfloor + 1$$

> If the division gives a decimal, apply the **floor** operation (round down). This means the last position doesn't have enough pixels and is **skipped**.

### Example Calculations

| Input ($N$) | Filter ($F$) | Padding ($P$) | Stride ($S$) | Output |
|:-----------:|:------------:|:-------------:|:------------:|:------:|
| 28 | 3 | 0 | 1 | $\lfloor\frac{28-3}{1}\rfloor + 1 = 26$ |
| 28 | 3 | 1 | 1 | $\lfloor\frac{28+2-3}{1}\rfloor + 1 = 28$ |
| 7 | 3 | 0 | 2 | $\lfloor\frac{7-3}{2}\rfloor + 1 = 3$ |
| 7 | 3 | 1 | 2 | $\lfloor\frac{7+2-3}{2}\rfloor + 1 = 4$ |
| 6 | 3 | 0 | 2 | $\lfloor\frac{6-3}{2}\rfloor + 1 = \lfloor1.5\rfloor + 1 = 2$ |

> **Special case**: When $\frac{N-F}{S}$ is not an integer, the filter can't fully cover the last position → those pixels are **dropped**.

## Why Use Strides?

### Reason 1: Extract Only High-Level Features

- Small stride → captures **fine-grained, low-level details**
- Large stride → captures only **coarse, high-level features**
- Useful when you don't need detailed features for your task

### Reason 2: Reduce Computation

- Larger stride → smaller feature maps → **faster training**
- Less relevant today due to improved computing power, but still used in certain architectures

> When stride > 1, it's called **strided convolution**.

### Keras Implementation

```python
Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=(28,28,1))
```

- `strides=(2,2)` means 2-pixel movement in **both** horizontal and vertical directions
- You can set different strides per direction: `strides=(2,1)`

### Impact on Feature Map Size

With stride = 2 on a $28 \times 28$ input and $3 \times 3$ filter:

$$\lfloor\frac{28 + 2(0) - 3}{2}\rfloor + 1 = \lfloor12.5\rfloor + 1 = 13$$

Three layers with stride 2: $28 \to 13 \to 6 \to 3$ — **rapid size reduction!**
