# PyTorch Interview Questions - Theory Questions

## Question 1

**What is PyTorch and what are its main features?**

### Answer

**Definition**: PyTorch is an open-source deep learning framework developed by **Meta AI (Facebook)**. It's known for its **dynamic computation graphs** and **Pythonic** design.

### Main Features

| Feature | Description |
|---------|-------------|
| **Dynamic Graphs** | Build graphs on-the-fly (eager execution) |
| **Autograd** | Automatic differentiation |
| **GPU Acceleration** | CUDA support |
| **TorchScript** | Production deployment |
| **Pythonic** | Feels like native Python |
| **Research-friendly** | Flexible for experimentation |

### Python Code Example
```python
import torch
import torch.nn as nn

# Check PyTorch version and GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Basic tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print(f"x + y = {z}")

# Simple neural network
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

---

## Question 2

**What is the difference between PyTorch and TensorFlow?**

### Answer

### Comparison Table

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Execution** | Eager (dynamic) | Graph-based (eager in TF2) |
| **Graph** | Define-by-run | Define-and-run |
| **Debugging** | Easy (standard Python) | More complex |
| **API Style** | Pythonic, explicit | Keras (high-level) |
| **Deployment** | TorchServe, ONNX | TF Serving, TFLite |
| **Community** | Research-focused | Production-focused |

### Python Code Example
```python
# PyTorch: Explicit and Pythonic
import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Standard Python debugging works
model = PyTorchModel()
x = torch.randn(32, 10)
output = model(x)  # Can set breakpoints here
```

### When to Use
- **PyTorch**: Research, prototyping, flexibility needed
- **TensorFlow**: Production, mobile deployment, enterprise

---

## Question 3

**Explain the concept of tensors in PyTorch.**

### Answer

**Definition**: A tensor is a multi-dimensional array, similar to NumPy arrays, but with GPU support and automatic differentiation.

### Tensor Properties

| Property | Description |
|----------|-------------|
| **shape** | Dimensions of tensor |
| **dtype** | Data type (float32, int64, etc.) |
| **device** | CPU or CUDA |
| **requires_grad** | Track gradients |

### Python Code Example
```python
import torch

# Creating tensors
scalar = torch.tensor(5)           # 0D
vector = torch.tensor([1, 2, 3])   # 1D
matrix = torch.tensor([[1, 2], [3, 4]])  # 2D

# Tensor properties
print(f"Shape: {matrix.shape}")      # torch.Size([2, 2])
print(f"dtype: {matrix.dtype}")      # torch.int64
print(f"device: {matrix.device}")    # cpu

# Create with specific properties
x = torch.zeros(3, 4, dtype=torch.float32)
y = torch.ones(3, 4, device='cuda' if torch.cuda.is_available() else 'cpu')

# Enable gradient tracking
z = torch.randn(3, 3, requires_grad=True)

# From NumPy
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

# To NumPy
back_to_numpy = tensor_from_numpy.numpy()
```

---

## Question 4

**What is Autograd in PyTorch?**

### Answer

**Definition**: Autograd is PyTorch's automatic differentiation engine that computes gradients for backpropagation.

### How It Works

| Step | Description |
|------|-------------|
| 1. Forward | Operations recorded in computation graph |
| 2. Backward | `loss.backward()` computes gradients |
| 3. Update | Optimizer updates parameters |

### Python Code Example
```python
import torch

# Simple gradient computation
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x^2

y.backward()  # Compute dy/dx
print(f"dy/dx at x=3: {x.grad}")  # Output: 6.0

# In neural networks
model = torch.nn.Linear(10, 1)
x = torch.randn(32, 10)
y_true = torch.randn(32, 1)

# Forward pass
y_pred = model(x)
loss = torch.nn.functional.mse_loss(y_pred, y_true)

# Backward pass - compute gradients
loss.backward()

# Gradients stored in .grad
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")

# Gradient context managers
with torch.no_grad():  # Disable gradient tracking
    inference_output = model(x)

# Detach from graph
detached = y_pred.detach()
```

---

## Question 5

**What is the difference between torch.Tensor and torch.tensor?**

### Answer

### Comparison

| Aspect | `torch.Tensor` | `torch.tensor` |
|--------|----------------|----------------|
| **Type** | Class constructor | Factory function |
| **dtype** | Default float32 | Infers from data |
| **Copy** | May share memory | Always copies |
| **Recommended** | Legacy | ✅ Preferred |

### Python Code Example
```python
import torch

# torch.Tensor - Class constructor (legacy)
a = torch.Tensor([1, 2, 3])
print(f"torch.Tensor dtype: {a.dtype}")  # float32 (always)

# torch.tensor - Factory function (recommended)
b = torch.tensor([1, 2, 3])
print(f"torch.tensor dtype: {b.dtype}")  # int64 (inferred)

c = torch.tensor([1.0, 2.0, 3.0])
print(f"torch.tensor float dtype: {c.dtype}")  # float32

# Specify dtype explicitly
d = torch.tensor([1, 2, 3], dtype=torch.float32)

# Other factory functions
zeros = torch.zeros(3, 3)
ones = torch.ones(3, 3)
randn = torch.randn(3, 3)  # Normal distribution
rand = torch.rand(3, 3)    # Uniform [0, 1)
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
```

---

## Question 6

**Explain torch.nn.Module and how to create custom modules.**

### Answer

**Definition**: `nn.Module` is the base class for all neural network modules in PyTorch. All layers and models inherit from it.

### Key Methods

| Method | Purpose |
|--------|---------|
| `__init__` | Define layers |
| `forward` | Define forward pass |
| `parameters()` | Get trainable parameters |
| `to(device)` | Move to GPU/CPU |

### Python Code Example
```python
import torch
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Initialize parent class
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = CustomNetwork(input_size=10, hidden_size=64, output_size=2)

# View parameters
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
x = torch.randn(32, 10).to(device)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 2]
```

---

## Question 7

**What are the different optimizers available in PyTorch?**

### Answer

### Common Optimizers

| Optimizer | Use Case |
|-----------|----------|
| **SGD** | Simple, good with momentum |
| **Adam** | Most common default |
| **AdamW** | Adam with weight decay |
| **RMSprop** | Good for RNNs |
| **Adagrad** | Sparse gradients |

### Python Code Example
```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)

# SGD with momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most popular)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam with decoupled weight decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training step
def train_step(model, optimizer, x, y_true):
    optimizer.zero_grad()  # Clear gradients
    
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y_true)
    
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    return loss.item()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)

# In training loop
for epoch in range(100):
    # ... training code ...
    scheduler.step()  # Update learning rate
```

---

## Question 8

**Explain loss functions in PyTorch.**

### Answer

### Common Loss Functions

| Loss | Use Case |
|------|----------|
| `CrossEntropyLoss` | Multi-class classification |
| `BCELoss` | Binary classification |
| `MSELoss` | Regression |
| `L1Loss` | Robust regression |
| `NLLLoss` | With log_softmax output |

### Python Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-class classification
criterion = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)  # Batch of 32, 10 classes
labels = torch.randint(0, 10, (32,))  # Integer labels
loss = criterion(logits, labels)

# Binary classification
bce_loss = nn.BCELoss()
probs = torch.sigmoid(torch.randn(32, 1))
binary_labels = torch.randint(0, 2, (32, 1)).float()
loss = bce_loss(probs, binary_labels)

# BCEWithLogitsLoss (more stable)
bce_logits = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
loss = bce_logits(logits, binary_labels)

# Regression
mse_loss = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = mse_loss(predictions, targets)

# Custom loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

---

## Question 9

**What is the difference between model.train() and model.eval()?**

### Answer

### Comparison

| Mode | Dropout | BatchNorm | Gradients |
|------|---------|-----------|-----------|
| `train()` | Active | Updates stats | Computed |
| `eval()` | Disabled | Uses running stats | Optional |

### Python Code Example
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.BatchNorm1d(64),
    nn.Dropout(0.5),
    nn.Linear(64, 1)
)

# Training mode
model.train()
x_train = torch.randn(32, 10)
output_train = model(x_train)  # Dropout active, BatchNorm updates

# Evaluation mode
model.eval()
x_test = torch.randn(32, 10)

with torch.no_grad():  # Also disable gradient computation
    output_eval = model(x_test)  # Dropout inactive, BatchNorm uses learned stats

# Common pattern
def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)

# Don't forget to switch back for training!
model.train()
```

### Key Points
- Always call `model.eval()` before inference
- Use `torch.no_grad()` for faster inference
- Remember to call `model.train()` when resuming training

---

## Question 10

**How does PyTorch handle GPU computation?**

### Answer

### GPU Operations

| Operation | Code |
|-----------|------|
| Check availability | `torch.cuda.is_available()` |
| Move to GPU | `tensor.to('cuda')` or `tensor.cuda()` |
| Move to CPU | `tensor.to('cpu')` or `tensor.cpu()` |
| Device count | `torch.cuda.device_count()` |

### Python Code Example
```python
import torch
import torch.nn as nn

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Move tensor to GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)
# or x_gpu = x.cuda()

# Move model to GPU
model = nn.Linear(1000, 100)
model = model.to(device)

# Ensure data and model are on same device
x_gpu = torch.randn(32, 1000, device=device)
output = model(x_gpu)

# Mixed precision training (faster)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Use float16 where safe
    output = model(x_gpu)
    loss = output.sum()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Clear GPU memory
torch.cuda.empty_cache()
```


---

# --- Missing Questions Restored from Source (Q11-Q21) ---

## Question 11

**What is the purpose of zero_grad() in PyTorch, and when is it used?**

**Answer:**

### Definition
`zero_grad()` resets (zeros out) the gradients of all model parameters before each backward pass. PyTorch **accumulates gradients by default**, so without calling `zero_grad()`, gradients from multiple backward passes stack up.

### Why It's Needed

| Without `zero_grad()` | With `zero_grad()` |
|----------------------|--------------------|
| Gradients accumulate across batches | Fresh gradients each batch |
| Incorrect parameter updates | Correct parameter updates |
| Training diverges | Training converges |

### Code Example
```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()          # Step 1: Zero gradients
    output = model(batch_x)         # Step 2: Forward pass
    loss = criterion(output, batch_y)  # Step 3: Compute loss
    loss.backward()                 # Step 4: Backward pass (compute gradients)
    optimizer.step()                # Step 5: Update parameters

# Alternative: zero_grad on model
model.zero_grad()  # Equivalent if optimizer covers all model params

# PyTorch 1.7+: set_to_none for better performance
optimizer.zero_grad(set_to_none=True)  # Sets grads to None instead of 0
```

### When Gradient Accumulation is Intentional
```python
# Simulate larger batch size with gradient accumulation
accum_steps = 4
for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accum_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    if (i + 1) % accum_steps == 0:
        optimizer.step()        # Update every N batches
        optimizer.zero_grad()   # Then zero
```

### Interview Tip
Always explain that PyTorch accumulates gradients by design (useful for gradient accumulation). `optimizer.zero_grad(set_to_none=True)` (PyTorch 1.7+) is more memory-efficient than the default because it deallocates gradient tensors instead of filling them with zeros.

---

## Question 12

**Describe the process of backpropagation in PyTorch**

**Answer:**

### Definition
Backpropagation is the algorithm that computes gradients of the loss with respect to model parameters using the **chain rule of calculus**. PyTorch's **Autograd engine** handles this automatically.

### The Process

| Step | Code | What Happens |
|------|------|--------------|
| 1. **Forward pass** | `output = model(x)` | Build computation graph |
| 2. **Compute loss** | `loss = criterion(output, y)` | Scalar output |
| 3. **Backward pass** | `loss.backward()` | Compute all gradients |
| 4. **Update weights** | `optimizer.step()` | Apply gradients |

### Code Example
```python
import torch
import torch.nn as nn

# Simple network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop with backpropagation
for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()       # Clear old gradients
        
        # Forward: input → hidden → output (graph built dynamically)
        output = model(x)           
        loss = criterion(output, y) 
        
        # Backward: compute ∂loss/∂w for every parameter
        loss.backward()             
        
        # Inspect gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm():.4f}")
        
        optimizer.step()            # w = w - lr * ∂loss/∂w
```

### How Autograd Builds the Graph
```python
x = torch.randn(3, requires_grad=True)
y = x * 2         # y.grad_fn = MulBackward
z = y.sum()        # z.grad_fn = SumBackward
z.backward()       # Traverses graph backward: Sum → Mul → x
print(x.grad)      # tensor([2., 2., 2.])
```

### Interview Tip
PyTorch uses **dynamic computation graphs** (define-by-run), meaning the graph is rebuilt every forward pass. This enables conditional logic and variable-length inputs, unlike TensorFlow 1.x's static graphs. The graph is automatically freed after `backward()` unless `retain_graph=True`.

---

## Question 13

**Explain how gradient clipping works in PyTorch and why it may be necessary**

**Answer:**

### Definition
Gradient clipping limits the magnitude of gradients during training to prevent the **exploding gradient problem**, where gradients become extremely large and cause unstable training.

### Types of Gradient Clipping

| Type | Method | How It Works |
|------|--------|-------------|
| **Clip by norm** | `clip_grad_norm_()` | Scales gradient vector if norm exceeds threshold |
| **Clip by value** | `clip_grad_value_()` | Clamps each gradient element to [-value, value] |

### Code Example
```python
import torch
import torch.nn as nn

model = nn.LSTM(input_size=100, hidden_size=256, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for x, y in dataloader:
    optimizer.zero_grad()
    output, _ = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Clip by norm (most common) - scales gradients if total norm > max_norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # OR Clip by value - clamps each gradient independently
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    
    optimizer.step()

# Monitor gradient norms
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {total_norm:.4f}")  # Returns norm before clipping
```

### When Gradient Clipping Is Necessary
- **RNNs/LSTMs**: Long sequences cause exploding gradients
- **Transformers**: Deep attention layers
- **GANs**: Unstable training dynamics
- **Very deep networks**: Gradients compound across layers
- **Large learning rates**: Amplify gradient issues

### How Clip by Norm Works
```
if ||gradient|| > max_norm:
    gradient = gradient * (max_norm / ||gradient||)
# Direction preserved, magnitude capped
```

### Interview Tip
`clip_grad_norm_()` is preferred over `clip_grad_value_()` because it preserves gradient direction while only reducing magnitude. The returned value is the original norm, which is useful for monitoring — if it's consistently above your threshold, your learning rate may be too high.

---

## Question 14

**Explain batch normalization and its effects on training convergence**

**Answer:**

### Definition
Batch Normalization (BatchNorm) normalizes layer inputs by re-centering and re-scaling across the batch dimension, stabilizing and accelerating training.

### Formula
$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}} \cdot \gamma + \beta$$

Where $\gamma$ (scale) and $\beta$ (shift) are **learnable parameters**.

### Effects on Training

| Effect | Description |
|--------|-------------|
| **Faster convergence** | Allows higher learning rates |
| **Reduces internal covariate shift** | Stabilizes layer input distributions |
| **Regularization** | Mini-batch noise acts as regularizer |
| **Gradient flow** | Mitigates vanishing/exploding gradients |
| **Less sensitive to initialization** | More forgiving weight init |

### Code Example
```python
import torch.nn as nn

# BatchNorm in CNN
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),          # BatchNorm for 2D (after conv)
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),         # BatchNorm for 1D (after linear)
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Train vs Eval mode matters!
model.train()   # Uses batch statistics (mean, var of current batch)
model.eval()    # Uses running statistics (accumulated during training)
```

### BatchNorm Variants

| Variant | Class | Use Case |
|---------|-------|----------|
| **BatchNorm1d** | `nn.BatchNorm1d` | After Linear layers |
| **BatchNorm2d** | `nn.BatchNorm2d` | After Conv2d layers |
| **LayerNorm** | `nn.LayerNorm` | Transformers, RNNs |
| **GroupNorm** | `nn.GroupNorm` | Small batch sizes |
| **InstanceNorm** | `nn.InstanceNorm2d` | Style transfer |

### Interview Tip
Critical distinction: **train mode** uses current batch statistics, **eval mode** uses running averages accumulated during training. Forgetting to call `model.eval()` before inference is a common bug that causes inconsistent predictions.

---

## Question 15

**How does PyTorch handle weight initialization for neural networks?**

**Answer:**

### Default Initialization
PyTorch initializes layers with specific default strategies:

| Layer | Default Init |
|-------|--------------|
| **Linear** | Kaiming Uniform |
| **Conv2d** | Kaiming Uniform |
| **BatchNorm** | weight=1, bias=0 |
| **LSTM/GRU** | Uniform(-1/√h, 1/√h) |
| **Embedding** | Normal(0, 1) |

### Common Initialization Methods
```python
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)      # Good for sigmoid/tanh
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Best for ReLU
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# Apply to model
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
)
model.apply(init_weights)  # Applies recursively to all submodules
```

### Initialization Methods Comparison

| Method | Best For | Formula |
|--------|----------|---------|
| **Xavier/Glorot Uniform** | Sigmoid, Tanh | $U(-\sqrt{6/(fan_{in}+fan_{out})}, \sqrt{6/(fan_{in}+fan_{out})})$ |
| **Xavier Normal** | Sigmoid, Tanh | $N(0, \sqrt{2/(fan_{in}+fan_{out})})$ |
| **Kaiming/He Uniform** | ReLU, LeakyReLU | $U(-\sqrt{6/fan_{in}}, \sqrt{6/fan_{in}})$ |
| **Kaiming Normal** | ReLU, LeakyReLU | $N(0, \sqrt{2/fan_{in}})$ |
| **Orthogonal** | RNNs | Orthogonal matrix |
| **Zeros** | Biases | All zeros |

### Interview Tip
The key insight: Xavier init assumes linear activations, Kaiming init accounts for ReLU's zero-killing. Using the wrong init can cause vanishing/exploding activations. With modern architectures + BatchNorm + Adam, initialization matters less, but it's still important for deep networks without normalization.

---

## Question 16

**What are some common issues you may encounter when training models in PyTorch, and how do you troubleshoot them?**

**Answer:**

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Exploding gradients** | Loss becomes NaN/Inf | Gradient clipping, lower LR |
| **Vanishing gradients** | No learning, loss plateaus | Better init, BatchNorm, skip connections |
| **Overfitting** | Train acc high, val acc low | Dropout, augmentation, more data |
| **Underfitting** | Both accuracies low | Larger model, longer training, higher LR |
| **CUDA OOM** | `RuntimeError: CUDA out of memory` | Smaller batch, mixed precision, gradient checkpointing |
| **Shape mismatch** | `RuntimeError: size mismatch` | Print shapes at each layer |
| **Not learning** | Loss doesn't decrease | Check LR, loss function, data pipeline |

### Debugging Techniques
```python
import torch

# 1. Check for NaN/Inf in loss
if torch.isnan(loss) or torch.isinf(loss):
    print("NaN/Inf detected in loss!")
    # Check inputs, gradients, learning rate

# 2. Monitor gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 100:
            print(f"Exploding: {name} grad={grad_norm:.2f}")
        elif grad_norm < 1e-7:
            print(f"Vanishing: {name} grad={grad_norm:.2e}")

# 3. Overfit on one batch (sanity check)
for x, y in train_loader:
    for i in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0: print(f"Step {i}: loss={loss.item():.4f}")
    break  # Only one batch

# 4. Check data pipeline
for x, y in train_loader:
    print(f"Input: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.2f}, {x.max():.2f}]")
    print(f"Labels: shape={y.shape}, unique={y.unique()}")
    break

# 5. GPU memory debugging
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
torch.cuda.empty_cache()
```

### Common Mistakes
1. **Forgetting `model.eval()`** before inference (BatchNorm/Dropout behave differently)
2. **Forgetting `torch.no_grad()`** during inference (wastes memory)
3. **Forgetting `zero_grad()`** (gradients accumulate)
4. **Wrong loss function** (BCE for multi-class instead of CE)
5. **Data not on same device** (CPU tensor vs CUDA tensor)

### Interview Tip
The best debugging technique is the **overfit-one-batch test**: if your model can't perfectly memorize a single batch, the bug is in your model, data loading, or loss function — not in hyperparameters.

---

## Question 17

**What is the use of transforms in PyTorch’s torchvision package?**

**Answer:**

### Definition
Transforms in `torchvision.transforms` are preprocessing and data augmentation operations applied to images before feeding them to a model.

### Common Transforms

| Transform | Purpose | Example |
|-----------|---------|--------|
| **Resize** | Standardize size | `Resize(224)` |
| **ToTensor** | PIL/ndarray → Tensor | `ToTensor()` |
| **Normalize** | Match pre-trained stats | `Normalize(mean, std)` |
| **RandomCrop** | Data augmentation | `RandomCrop(224)` |
| **RandomHorizontalFlip** | Data augmentation | `RandomHorizontalFlip(p=0.5)` |
| **ColorJitter** | Color augmentation | `ColorJitter(0.2, 0.2, 0.2)` |
| **Compose** | Chain transforms | `Compose([...])` |

### Code Example
```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply to dataset
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

### torchvision v2 Transforms (Modern API)
```python
from torchvision.transforms import v2

# v2: Works on tensors, batches, bounding boxes, masks
transform_v2 = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Interview Tip
Always apply augmentation only to training data, not validation/test. The `Normalize` values `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` are ImageNet statistics — use them when using pre-trained models. For models trained from scratch, compute your dataset's statistics.

---

## Question 18

**What is PyTorch’s TorchScript , and how does it aid in deploying PyTorch models in production environments?**

**Answer:**

### Definition
TorchScript is a way to serialize and optimize PyTorch models for production deployment, enabling them to run **without a Python runtime** (e.g., in C++, mobile, or embedded systems).

### Two Approaches

| Method | How | Best For |
|--------|-----|----------|
| **Tracing** (`torch.jit.trace`) | Records operations on example input | Simple models without control flow |
| **Scripting** (`torch.jit.script`) | Analyzes Python code directly | Models with if/else, loops |

### Code Example
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
model.eval()

# Method 1: Tracing (for models without control flow)
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Method 2: Scripting (for models with control flow)
class ConditionalModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:      # Control flow!
            return x * 2
        return x * 3

scripted_model = torch.jit.script(ConditionalModel())
scripted_model.save('model_scripted.pt')

# Load and run without Python
loaded = torch.jit.load('model_traced.pt')
output = loaded(torch.randn(1, 10))

# C++ deployment
# #include <torch/script.h>
# auto model = torch::jit::load("model_traced.pt");
# auto output = model.forward({input_tensor});
```

### Benefits for Production

| Benefit | Description |
|---------|-------------|
| **No Python needed** | Run in C++, Java, mobile |
| **Optimizations** | Graph fusion, constant folding |
| **Portability** | Same model file across platforms |
| **Mobile deployment** | PyTorch Mobile (iOS/Android) |
| **Reproducibility** | Frozen, serialized computation |

### Interview Tip
Use `torch.jit.trace` for simple feed-forward models and `torch.jit.script` for models with control flow (if/else, loops). In practice, tracing is more reliable and widely used. For maximum performance, combine TorchScript with `torch.compile()` (PyTorch 2.0+).

---

## Question 19

**Explain the concept of “model quantization” in PyTorch and when it is useful.**

**Answer:**

### Definition
Quantization reduces model size and increases inference speed by converting weights and activations from 32-bit floating point (FP32) to lower-precision formats like 8-bit integers (INT8).

### Quantization Types

| Type | When Applied | Accuracy | Speed |
|------|-------------|----------|-------|
| **Dynamic** | At runtime | Good | 2-3x |
| **Static (PTQ)** | Post-training with calibration | Better | 3-4x |
| **Quantization-Aware Training (QAT)** | During training | Best | 3-4x |

### Code Example
```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize, prepare, convert

model = MyModel()
model.eval()

# 1. Dynamic Quantization (easiest, great for NLP)
quant_model = quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# 2. Static Quantization (Post-Training)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86
prepared = prepare(model)  # Insert observers

# Calibrate with representative data
with torch.no_grad():
    for x, _ in calibration_loader:
        prepared(x)  # Collect statistics

quant_model_static = convert(prepared)  # Convert to quantized

# 3. Quantization-Aware Training (best accuracy)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_qat = prepare_qat(model.train())  # Insert fake quantize modules

for epoch in range(5):
    for x, y in train_loader:
        output = model_qat(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

quant_model_qat = convert(model_qat.eval())
```

### Size and Speed Comparison

| Metric | FP32 | INT8 (Quantized) |
|--------|------|------------------|
| **Model size** | 100 MB | ~25 MB (4x smaller) |
| **Inference speed** | 1x | 2-4x faster |
| **Memory** | 100% | ~25% |
| **Accuracy loss** | Baseline | 0.5-2% typical |

### When to Quantize
- **Edge/mobile deployment** — limited memory and compute
- **CPU inference** — INT8 is much faster on CPU
- **Cost reduction** — smaller models need less compute
- **Latency-sensitive** — real-time applications

### Interview Tip
Dynamic quantization is a "free lunch" for inference — one line of code for 2-3x speedup with minimal accuracy loss. It's especially effective for NLP models (LSTM, Transformer) where Linear layers dominate compute.

---

## Question 20

**What is the role of PyTorch in reinforcement learning research, and can you provide an example?**

**Answer:**

### PyTorch in RL
PyTorch is the dominant framework for RL research due to its **dynamic computation graphs**, which naturally handle variable-length episodes, conditional actions, and stochastic policies.

### Why PyTorch for RL

| Feature | RL Benefit |
|---------|--------|
| **Dynamic graphs** | Variable-length episodes, conditional logic |
| **Autograd** | Easy policy gradient computation |
| **Distributions** | `torch.distributions` for stochastic policies |
| **GPU support** | Fast environment simulation |
| **Ecosystem** | Stable-Baselines3, RLlib, TorchRL |

### REINFORCE (Policy Gradient) Example
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

# Policy network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# Training
env = gym.make('CartPole-v1')
policy = PolicyNet(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for episode in range(1000):
    state, _ = env.reset()
    log_probs, rewards = [], []
    
    # Collect trajectory
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        dist = Categorical(probs)           # Stochastic policy
        action = dist.sample()              # Sample action
        log_probs.append(dist.log_prob(action))
        
        state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)
    
    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Policy gradient update
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Popular RL Libraries Built on PyTorch

| Library | Focus |
|---------|-------|
| **Stable-Baselines3** | Reliable RL algorithms (PPO, SAC, DQN) |
| **TorchRL** | Official PyTorch RL library |
| **RLlib (Ray)** | Distributed, scalable RL |
| **CleanRL** | Single-file RL implementations |

### Interview Tip
PyTorch dominates RL research because `torch.distributions` + autograd makes implementing policy gradients natural. The key line is `dist.log_prob(action)` — this gives you the log probability needed for the REINFORCE gradient: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$.

---

## Question 21

**Describe your experience contributing to PyTorch’s open-source community or using community-created tools.**

**Answer:**

### PyTorch Ecosystem Overview

| Category | Key Projects | Description |
|----------|-------------|-------------|
| **Vision** | torchvision, timm, Detectron2 | Image models, detection |
| **NLP** | HuggingFace Transformers, torchtext | Language models, tokenizers |
| **Audio** | torchaudio, SpeechBrain | Audio processing, ASR |
| **Graphs** | PyG (PyTorch Geometric), DGL | Graph neural networks |
| **RL** | Stable-Baselines3, TorchRL | Reinforcement learning |
| **Production** | TorchServe, ONNX | Model serving, export |
| **Research** | PyTorch Lightning, FastAI | Training abstractions |

### Ways to Contribute and Engage

```
Contribution Levels:
1. 📚 User: Use PyTorch + community tools in projects
2. 🐛 Bug Reporter: File issues with reproducible examples
3. 📖 Documentation: Improve tutorials, fix docs
4. 🔧 Code: Fix bugs, add features, review PRs
5. 🏠 Maintainer: Own a module or ecosystem project
```

### Community Tools Example
```python
# PyTorch Lightning - reduces boilerplate
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)

# timm - huge model zoo
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# HuggingFace - state-of-the-art NLP
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

### Contributing Best Practices
1. **Start small**: Fix typos, improve docstrings
2. **Read contributing guidelines**: Each project has different standards
3. **Write tests**: All contributions should include tests
4. **Follow code style**: Use the project's linting/formatting rules
5. **Engage in discussions**: GitHub Issues, PyTorch Forums, Discord

### Interview Tip
Mention specific community tools you've used (Lightning for training, timm for models, HuggingFace for NLP) and how they improved your workflow. Even using community tools extensively counts as ecosystem engagement — you don't need to be a core contributor.

---

## Question 22

**What is PyTorch and how does it differ from other deep learning frameworks like TensorFlow ?**

*Answer to be added.*

---

## Question 23

**In PyTorch, what is the difference between a Tensor and a Variable ?**

*Answer to be added.*

---

## Question 24

**How can you convert a NumPy array to a PyTorch Tensor ?**

*Answer to be added.*

---

## Question 25

**What is the purpose of the .grad attribute in PyTorch Tensors ?**

*Answer to be added.*

---

## Question 26

**Explain what CUDA is and how it relates to PyTorch**

*Answer to be added.*

---

## Question 27

**How does automatic differentiation work in PyTorch using Autograd ?**

*Answer to be added.*

---

## Question 28

**Describe the steps for creating a neural network model in PyTorch**

*Answer to be added.*

---

## Question 29

**What is a Sequential model in PyTorch, and how does it differ from using the Module class?**

*Answer to be added.*

---

## Question 30

**What is the role of the forward method in a PyTorch Module ?**

*Answer to be added.*

---

## Question 31

**In PyTorch, what are optimizers , and how do you use them?**

*Answer to be added.*

---

## Question 32

**How do you create a data loader in PyTorch for custom datasets ?**

*Answer to be added.*

---

## Question 33

**How do you manage and preprocess time-series data in PyTorch for RNNs ?**

*Answer to be added.*

---

## Question 34

**Explain the concept of data augmentation and its implementation in PyTorch**

*Answer to be added.*

---

## Question 35

**How do you use GPU accelerators for distributed training in PyTorch?**

*Answer to be added.*

---

## Question 36

**Compare recurrent neural networks (RNNs) , long short-term memory networks (LSTMs) , and gated recurrent units (GRUs) in the context of PyTorch**

*Answer to be added.*

---

## Question 37

**Discuss the latest research on neural architecture search (NAS) and its application within PyTorch**

*Answer to be added.*

---

## Question 38

**How can generative adversarial networks (GANs) be implemented in PyTorch, and what are some of their challenges?**

*Answer to be added.*

---
## Question 39

**How do you implement custom layers in PyTorch?**

**Answer:**

Custom layers in PyTorch extend `nn.Module` with learnable parameters and a `forward()` method.

```python
import torch
import torch.nn as nn

# === Basic Custom Layer ===
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter makes tensors trainable
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias

# === Custom Layer with Existing Layers ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.relu(out + residual)  # Skip connection

# === Attention Layer ===
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

# === Usage ===
model = nn.Sequential(
    LinearLayer(784, 128),
    nn.ReLU(),
    ResidualBlock(128),
    nn.Linear(128, 10)
)

x = torch.randn(32, 784)
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

| Component | Purpose |
|-----------|--------|
| `nn.Module` | Base class for all layers/models |
| `nn.Parameter` | Registers tensor as trainable parameter |
| `forward()` | Defines computation |
| `__init__()` | Define sub-layers and parameters |

> **Interview Tip:** Every custom layer must call `super().__init__()`. Use `nn.Parameter` for learnable weights; plain tensors won't be updated by the optimizer.

---

## Question 40

**How can you implement learning rate scheduling in PyTorch?**

**Answer:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === 1. StepLR: Decay every N epochs ===
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# LR: 0.01 -> 0.005 (epoch 10) -> 0.0025 (epoch 20)

# === 2. ExponentialLR ===
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# LR *= 0.95 each epoch

# === 3. CosineAnnealingLR ===
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
# Cosine decay to eta_min over T_max epochs

# === 4. ReduceLROnPlateau (most practical) ===
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

# === 5. OneCycleLR (best for training from scratch) ===
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=num_epochs * len(train_loader)
)

# === 6. Warmup + Decay (Custom) ===
def warmup_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warmup
    return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / 45 * 3.14159)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# === Training Loop ===
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        # For OneCycleLR: scheduler.step() HERE (per batch)
    
    # For most schedulers: step per epoch
    scheduler.step()  # StepLR, ExponentialLR, CosineAnnealing
    # scheduler.step(val_loss)  # For ReduceLROnPlateau only
    
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

| Scheduler | Best For | Step When |
|-----------|----------|----------|
| StepLR | Simple decay | Per epoch |
| CosineAnnealing | Training from scratch | Per epoch |
| ReduceLROnPlateau | Any (adaptive) | Per epoch (pass val_loss) |
| OneCycleLR | Fast convergence | Per batch |
| Warmup + Decay | Large models, transformers | Per epoch |

> **Interview Tip:** `OneCycleLR` often gives the best results for training from scratch. For fine-tuning, use `CosineAnnealingLR` or `ReduceLROnPlateau`. Always call `scheduler.step()` **after** `optimizer.step()`.

---

## Question 41

**Explain transfer learning and its implementation in PyTorch.**

**Answer:**

Transfer learning uses a model pre-trained on a large dataset and adapts it to a new, smaller dataset.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# === 1. Load Pre-trained Model ===
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# === 2. Freeze All Layers ===
for param in model.parameters():
    param.requires_grad = False

# === 3. Replace Classifier Head ===
num_classes = 10
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
# New layers are trainable by default

# === 4. Setup Data ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. Phase 1: Train Head Only ===
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# === 6. Phase 2: Fine-tune Last Layers ===
# Unfreeze last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Low LR for pre-trained
    {'params': model.fc.parameters(), 'lr': 1e-4}       # Higher LR for new layers
], weight_decay=1e-4)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
```

| Strategy | Freeze | When to Use |
|----------|--------|-------------|
| Feature extraction | All base layers | Very small dataset, similar domain |
| Fine-tuning top layers | Most base layers | Medium dataset |
| Full fine-tuning | Nothing | Large dataset, different domain |

> **Interview Tip:** PyTorch allows per-parameter-group learning rates in the optimizer—use lower LR for pre-trained layers and higher LR for new layers. Always use ImageNet normalization when using ImageNet-pretrained models.

---

## Question 42

**What are Graph Neural Networks (GNNs) and how can they be implemented in PyTorch?**

**Answer:**

GNNs operate on graph-structured data (nodes + edges), learning representations by aggregating information from neighbors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Simple GCN from Scratch ===
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x, adj):
        # x: (N, in_features), adj: (N, N) adjacency matrix
        support = torch.mm(x, self.weight)    # Transform features
        output = torch.mm(adj, support)        # Aggregate neighbors
        return output + self.bias

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()
        self.conv1 = GraphConvolution(n_features, n_hidden)
        self.conv2 = GraphConvolution(n_hidden, n_classes)
    
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, adj)

# === 2. Using PyTorch Geometric (Production) ===
# pip install torch-geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data

class GCN_PyG(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Graph-level pooling
        return self.classifier(x)

# Create graph data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
node_features = torch.randn(3, 16)  # 3 nodes, 16 features each
graph = Data(x=node_features, edge_index=edge_index)

# === 3. Graph Attention Network (GAT) ===
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

| GNN Type | Mechanism | Best For |
|----------|-----------|----------|
| GCN | Spectral convolution | Node classification |
| GAT | Attention on neighbors | When neighbor importance varies |
| GraphSAGE | Sampling + aggregating | Large-scale graphs |
| GIN | Isomorphism-based | Graph-level classification |

| Application | Input Graph |
|------------|-------------|
| Drug discovery | Molecular structure |
| Social networks | User connections |
| Fraud detection | Transaction networks |
| Recommendation | User-item interactions |

> **Interview Tip:** GNNs follow a message-passing paradigm: each node aggregates features from its neighbors, applies a transformation, and updates its representation. PyTorch Geometric provides optimized implementations for production use.

## Question 43

**How do you check if your PyTorch model is utilizing the GPU ?**

*Answer to be added.*

---

## Question 44

**What strategies can you use to monitor and decrease overfitting in a PyTorch model?**

*Answer to be added.*

---

## Question 45

**How would you create a PyTorch extension module with custom C++/CUDA operations ?**

*Answer to be added.*

---

## Question 46

**How do you ensure reproducibility of experiments when using PyTorch?**

*Answer to be added.*

---

## Question 47

**Portray how PyTorch Lightning can simplify the standard PyTorch workflow**

*Answer to be added.*

---
