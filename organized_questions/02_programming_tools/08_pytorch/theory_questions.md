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
