# Deep Learning Interview Questions - Theory Questions

## Question 1: What is an artificial neural network?

### Definition
An artificial neural network (ANN) is a computational model inspired by biological neurons, consisting of interconnected nodes organized in layers that learn to map inputs to outputs through weighted connections and non-linear activation functions.

### Core Concepts

| Component | Description |
|-----------|-------------|
| **Neuron** | Basic computational unit |
| **Weights** | Learnable parameters controlling signal strength |
| **Bias** | Learnable offset term |
| **Activation** | Non-linear transformation |
| **Layer** | Collection of neurons |

### Mathematical Formulation
For a single neuron:
$$y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

### Practical Relevance
- Foundation of all deep learning
- Universal function approximators
- Powers image recognition, NLP, recommendation systems

---

## Question 2: Explain the concept of 'depth' in deep learning

### Definition
Depth refers to the number of hidden layers in a neural network. Deep learning uses networks with multiple layers (typically >2) to learn hierarchical representations of data automatically.

### Core Concepts

| Network Type | Hidden Layers | Characteristics |
|--------------|---------------|-----------------|
| **Shallow** | 1 layer | Simple patterns |
| **Deep** | 2+ layers | Hierarchical features |
| **Very Deep** | 50+ layers | Complex abstractions (ResNet) |

### Why Depth Matters
- Each layer learns increasingly abstract features
- Early layers: edges, textures
- Middle layers: parts, shapes
- Deep layers: objects, concepts

### Practical Relevance
- Depth enables automatic feature learning
- Eliminates manual feature engineering
- More parameter-efficient than wide shallow networks

---

## Question 3: What are activation functions, and why are they necessary?

### Definition
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without them, any deep network would collapse to a single linear transformation.

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, x)$ | [0, ∞) | Hidden layers |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0, 1) | Binary output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Zero-centered |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | (0, 1) | Multi-class output |

### Why Necessary
- Stack of linear functions = still linear
- Non-linearity enables learning complex decision boundaries
- Without activation: $f(g(x)) = Wx$ (just linear)

---

## Question 4: Describe the role of weights and biases in neural networks

### Definition
**Weights** control the strength of connections between neurons, determining how much each input contributes to the output. **Biases** allow shifting the activation function, enabling neurons to activate even when inputs are zero.

### Mathematical Role
$$z = \sum_{i} w_i x_i + b$$
$$a = \sigma(z)$$

| Parameter | Role | Analogy |
|-----------|------|---------|
| **Weight ($w$)** | Scales input importance | Volume knob |
| **Bias ($b$)** | Shifts activation threshold | Baseline offset |

### Learning Process
- Initialize randomly (or with Xavier/He)
- Update via gradient descent: $w = w - \eta \cdot \frac{\partial L}{\partial w}$
- Network learns optimal values through training

---

## Question 5: What is the vanishing gradient problem, and how can it be avoided?

### Definition
The vanishing gradient problem occurs when gradients become exponentially small during backpropagation through many layers, causing early layers to learn extremely slowly or not at all.

### Cause
- Sigmoid/Tanh derivatives are < 1
- Chain rule multiplies small values: $0.25^{10} \approx 0$

### Solutions

| Solution | How It Helps |
|----------|--------------|
| **ReLU Activation** | Gradient = 1 for positive inputs |
| **Skip Connections** | Direct gradient pathways |
| **Batch Normalization** | Keeps activations in good range |
| **Proper Initialization** | Xavier/He initialization |
| **LSTM/GRU** | Gating mechanisms for RNNs |

---

## Question 6: Explain the difference between shallow and deep neural networks

### Definition
**Shallow networks** have one hidden layer; **deep networks** have two or more. Deep networks learn hierarchical features automatically, while shallow networks require manual feature engineering.

### Comparison

| Aspect | Shallow (1 layer) | Deep (2+ layers) |
|--------|-------------------|------------------|
| **Feature Learning** | Manual engineering needed | Automatic |
| **Representation** | Single level | Hierarchical |
| **Capacity** | Limited | High |
| **Data Requirement** | Less | More |
| **Use Cases** | Simple patterns | Images, text, speech |

### Why Deep Works Better
- Composition of simple functions → complex functions
- Each layer builds on previous abstractions
- More parameter-efficient than exponentially wide shallow nets

---

## Question 7: What is the universal approximation theorem?

### Definition
The universal approximation theorem states that a neural network with a single hidden layer containing enough neurons can approximate any continuous function on a bounded domain to arbitrary precision.

### Core Concepts
- Applies to networks with non-linear activations
- Says nothing about learning (only representational power)
- Deep networks are more efficient approximators

### Practical Implications

| Theorem Says | Theorem Does NOT Say |
|--------------|---------------------|
| Can represent any function | How to find the weights |
| Single layer is sufficient | Single layer is efficient |
| Given enough neurons | How many neurons needed |

### Why We Use Deep Networks Anyway
- Shallow networks may need exponentially many neurons
- Deep networks are more parameter-efficient
- Deep networks generalize better in practice

---

## Question 8: What is forward propagation and backpropagation?

### Definition
**Forward propagation** computes the output by passing inputs through the network layer by layer. **Backpropagation** computes gradients of the loss with respect to all weights by propagating error backwards using the chain rule.

### Forward Propagation
```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output → Loss
```
$$a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

### Backpropagation
```
Loss → ∂L/∂Output → ∂L/∂Hidden → ... → ∂L/∂Weights
```
Using chain rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Algorithm Steps
1. **Forward**: Compute activations, store intermediate values
2. **Compute Loss**: Compare output to target
3. **Backward**: Compute gradients layer by layer
4. **Update**: $w = w - \eta \cdot \nabla_w L$

---

## Question 9: What is a Convolutional Neural Network (CNN), and when would you use it?

### Definition
A CNN is a neural network that uses convolutional layers to automatically learn spatial hierarchies of features from grid-like data (images), leveraging local connectivity and weight sharing.

### Key Components

| Component | Purpose |
|-----------|---------|
| **Convolution** | Extract local features |
| **Pooling** | Reduce dimensions, add invariance |
| **Fully Connected** | Final classification |

### When to Use CNN

| Use Case | Why CNN |
|----------|---------|
| Image Classification | Learns spatial features |
| Object Detection | Localize and classify |
| Image Segmentation | Pixel-level predictions |
| Medical Imaging | Pattern recognition |
| Audio (spectrograms) | 2D structure |

### Python Code Example
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)
```

---

## Question 10: Explain Recurrent Neural Networks (RNNs) and their use cases

### Definition
RNNs are neural networks with recurrent connections that maintain a hidden state across time steps, enabling them to process sequential data of variable length.

### Mathematical Formulation
$$h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$$
$$y_t = W_y h_t$$

### Use Cases

| Application | Why RNN |
|-------------|---------|
| Language Modeling | Sequential word dependencies |
| Machine Translation | Sequence-to-sequence |
| Speech Recognition | Audio is sequential |
| Time Series | Temporal patterns |
| Sentiment Analysis | Word order matters |

### Limitations
- Vanishing gradients for long sequences
- Solution: LSTM/GRU with gating mechanisms

---

## Question 11: What is the significance of Residual Networks (ResNets)?

### Definition
ResNets introduce skip connections that add the input of a layer directly to its output, enabling training of very deep networks (100+ layers) by solving the degradation problem.

### Core Concept: Residual Block
$$y = F(x) + x$$

Instead of learning $H(x)$, learn residual $F(x) = H(x) - x$

### Why ResNets Work

| Benefit | Explanation |
|---------|-------------|
| **Gradient Flow** | Skip connections provide direct gradient path |
| **Identity Mapping** | Easy to learn "do nothing" |
| **Very Deep** | Enables 100+ layer networks |
| **Better Accuracy** | Won ImageNet 2015 |

### Python Code Example
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)  # Skip connection
```

---

## Question 12: How does a Transformer architecture function?

### Definition
Transformers use self-attention mechanisms to process all positions in parallel, capturing long-range dependencies without recurrence. They form the basis of modern NLP (BERT, GPT).

### Core Components

| Component | Purpose |
|-----------|---------|
| **Self-Attention** | Relate all positions to each other |
| **Multi-Head Attention** | Multiple attention perspectives |
| **Positional Encoding** | Inject sequence order |
| **Feed-Forward** | Non-linear transformation |

### Self-Attention Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Advantages Over RNNs

| Transformer | RNN |
|-------------|-----|
| Parallel processing | Sequential |
| O(1) path length | O(n) path length |
| Captures long-range | Struggles with long-range |

### Applications
- BERT: Bidirectional understanding
- GPT: Text generation
- Vision Transformer (ViT): Image classification

---

## Question 13: What are Generative Adversarial Networks (GANs)?

### Definition
GANs consist of two networks trained adversarially: a **Generator** creates fake samples trying to fool a **Discriminator** that distinguishes real from fake. Through competition, the generator learns to create realistic data.

### Architecture
```
Random Noise → Generator → Fake Image
                              ↓
Real Image ────────────→ Discriminator → Real/Fake
```

### Training Objective
$$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

### Applications

| Application | Example |
|-------------|---------|
| Image Generation | StyleGAN faces |
| Image-to-Image | Pix2Pix |
| Super-Resolution | SRGAN |
| Data Augmentation | Generate training data |

### Challenges
- Mode collapse (generator produces limited variety)
- Training instability
- Difficult to evaluate

---

## Question 14: Describe how U-Net architecture works for image segmentation

### Definition
U-Net is an encoder-decoder architecture with skip connections between corresponding encoder and decoder layers, designed for precise image segmentation with limited training data.

### Architecture
```
Encoder (contracting)     Decoder (expanding)
    ↓                          ↑
[Conv + Pool] ─────────→ [Upsample + Conv]
    ↓                          ↑
[Conv + Pool] ─────────→ [Upsample + Conv]
    ↓                          ↑
      └───→ Bottleneck ────┘
```

### Key Features

| Feature | Purpose |
|---------|---------|
| **Encoder** | Capture context (what) |
| **Decoder** | Enable localization (where) |
| **Skip Connections** | Combine low/high level features |

### Why It Works
- Skip connections preserve spatial information lost during downsampling
- Combines semantic information (deep) with localization (shallow)
- Works well with limited data (data augmentation)

### Applications
- Medical image segmentation
- Satellite imagery
- Cell segmentation

---

## Question 15: Explain the concept of attention mechanisms in deep learning

### Definition
Attention mechanisms allow models to dynamically focus on relevant parts of the input when producing each output element, computing weighted combinations where weights indicate importance.

### Core Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Types

| Type | Description |
|------|-------------|
| **Self-Attention** | Query, Key, Value from same sequence |
| **Cross-Attention** | Query from one, Key/Value from another |
| **Multi-Head** | Multiple parallel attention operations |

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Long-range Dependencies** | Direct path between any positions |
| **Parallelization** | No sequential dependency |
| **Interpretability** | Visualize attention weights |

---

## Question 16: What is a Siamese Neural Network?

### Definition
A Siamese network uses two identical subnetworks with shared weights to learn a similarity function between input pairs, enabling comparison-based tasks like verification and one-shot learning.

### Architecture
```
Input A → [Shared Encoder] → Embedding A
                                        ↘
                                         Distance → Similar/Different
                                        ↗
Input B → [Shared Encoder] → Embedding B
```

### Use Cases

| Application | How It Helps |
|-------------|--------------|
| Face Verification | Are these the same person? |
| Signature Verification | Is this genuine? |
| One-Shot Learning | Learn from single example |
| Duplicate Detection | Find similar items |

### Training Loss
**Contrastive Loss:**
$$L = (1-y) \cdot D^2 + y \cdot \max(0, m-D)^2$$

---

## Question 17: What are loss functions, and why are they important?

### Definition
Loss functions quantify the error between model predictions and true values, providing a scalar objective that the optimization algorithm minimizes during training.

### Common Loss Functions

| Problem | Loss Function | Formula |
|---------|---------------|---------|
| **Regression** | MSE | $\frac{1}{N}\sum(y - \hat{y})^2$ |
| **Regression** | MAE | $\frac{1}{N}\sum|y - \hat{y}|$ |
| **Binary Classification** | BCE | $-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ |
| **Multi-class** | Cross-Entropy | $-\sum y_c \log(\hat{y}_c)$ |

### Why Important
- Defines what "correct" means
- Gradients derived from loss guide learning
- Choice affects model behavior (outlier sensitivity, etc.)

---

## Question 18: Explain the concept of gradient descent

### Definition
Gradient descent is an iterative optimization algorithm that finds a function's minimum by repeatedly moving in the direction opposite to the gradient (direction of steepest descent).

### Update Rule
$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta L$$

### Intuition
- Gradient points uphill
- Move opposite (downhill) to minimize
- Learning rate controls step size

### Algorithm Steps
1. Initialize parameters randomly
2. Compute loss on training data
3. Compute gradients: $\nabla_\theta L$
4. Update: $\theta = \theta - \eta \cdot \nabla_\theta L$
5. Repeat until convergence

---

## Question 19: Differences between batch, stochastic, and mini-batch gradient descent

### Definition
The three variants differ in how much data is used to compute each gradient update.

### Comparison

| Variant | Data per Update | Speed | Stability |
|---------|-----------------|-------|-----------|
| **Batch GD** | Entire dataset | Slow | Very stable |
| **SGD** | 1 sample | Fast | Noisy |
| **Mini-batch** | 32-256 samples | Balanced | Balanced |

### When to Use

| Variant | Use Case |
|---------|----------|
| **Batch** | Small datasets, convex problems |
| **SGD** | Online learning, escaping local minima |
| **Mini-batch** | Deep learning (default choice) |

### Python Code Example
```python
# Mini-batch gradient descent
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:  # Mini-batches
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

---

## Question 20: What are optimization algorithms like Adam, RMSprop, and AdaGrad?

### Definition
These are adaptive learning rate optimizers that adjust the learning rate per-parameter based on historical gradient information.

### Comparison

| Optimizer | Key Idea | Best For |
|-----------|----------|----------|
| **AdaGrad** | Accumulate squared gradients | Sparse features |
| **RMSprop** | Exponential moving average of squared gradients | Non-stationary |
| **Adam** | Momentum + RMSprop | General default |

### Adam Update
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Python Code Example
```python
# Different optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
```

---

## Question 21: How does Batch Normalization work?

### Definition
Batch Normalization normalizes layer inputs across the mini-batch to have zero mean and unit variance, then applies learnable scale and shift parameters. This stabilizes training and enables higher learning rates.

### Formula
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Faster Training** | Allows higher learning rates |
| **Regularization** | Slight noise from batch statistics |
| **Reduces Internal Covariate Shift** | Stable input distributions |

### Python Code Example
```python
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.BatchNorm1d(256),  # After linear, before activation
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

---

## Question 22: Describe the process of hyperparameter tuning

### Definition
Hyperparameter tuning searches for optimal values of parameters set before training (learning rate, batch size, architecture) to maximize model performance.

### Key Hyperparameters

| Category | Parameters |
|----------|------------|
| **Optimization** | Learning rate, batch size, optimizer |
| **Regularization** | Dropout, weight decay |
| **Architecture** | Layers, neurons, kernel size |

### Tuning Methods

| Method | Description |
|--------|-------------|
| **Grid Search** | Try all combinations |
| **Random Search** | Sample randomly (often better) |
| **Bayesian Optimization** | Model the objective function |

### Best Practices
1. Tune learning rate first (most important)
2. Use random search over grid search
3. Start coarse, then refine
4. Always validate on held-out data

---

## Question 23: What is early stopping, and how does it prevent overfitting?

### Definition
Early stopping monitors validation loss during training and halts when it stops improving for a specified number of epochs (patience), preventing the model from memorizing training noise.

### How It Works
1. Track best validation loss
2. If no improvement for `patience` epochs → stop
3. Restore weights from best epoch

### Python Code Example
```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

### Benefits
- Free regularization
- Saves training time
- No hyperparameter to tune (except patience)

---

## Question 24: Explain the trade-off between bias and variance

### Definition
The bias-variance trade-off describes the tension between model complexity: simple models have high bias (underfit), complex models have high variance (overfit). The goal is finding optimal complexity.

### Mathematical Decomposition
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

### In Deep Learning

| Model Property | Bias | Variance |
|----------------|------|----------|
| Few layers/neurons | High | Low |
| Many layers/neurons | Low | High |
| With regularization | Higher | Lower |
| More data | Same | Lower |

### Managing Trade-off
- Start with complex model (low bias)
- Add regularization to reduce variance
- Collect more data if possible

---

## Question 25: What are popular libraries and frameworks for deep learning?

### Major Frameworks

| Framework | Developer | Strengths |
|-----------|-----------|-----------|
| **PyTorch** | Meta | Pythonic, dynamic graphs, research |
| **TensorFlow** | Google | Production, TF Serving, mobile |
| **JAX** | Google | Functional, composable transforms |
| **Keras** | Google | High-level API, easy prototyping |

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| **Hugging Face** | Pre-trained models (NLP, vision) |
| **Lightning** | PyTorch training boilerplate |
| **Weights & Biases** | Experiment tracking |
| **ONNX** | Model interoperability |

---

## Question 26: Explain how a deep learning model can be deployed into production

### Deployment Pipeline

| Step | Description |
|------|-------------|
| **1. Export Model** | Save weights (ONNX, TorchScript) |
| **2. Optimize** | Quantization, pruning, TensorRT |
| **3. Containerize** | Docker for consistent environment |
| **4. Serve** | API (FastAPI, Flask) or inference server |
| **5. Monitor** | Track latency, accuracy drift |

### Deployment Options

| Option | Use Case |
|--------|----------|
| **REST API** | Web applications |
| **TensorFlow Serving** | High-throughput |
| **Edge/Mobile** | TFLite, CoreML |
| **Serverless** | AWS Lambda, Cloud Functions |

### Python Code Example
```python
# Export to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

# Serve with FastAPI
@app.post("/predict")
async def predict(input_data: InputData):
    tensor = preprocess(input_data)
    output = model(tensor)
    return {"prediction": output.tolist()}
```

---

## Question 27: What are considerations for scaling deep learning models?

### Key Considerations

| Aspect | Techniques |
|--------|------------|
| **Data Parallelism** | Distribute batches across GPUs |
| **Model Parallelism** | Split model across devices |
| **Mixed Precision** | FP16 for 2x memory/speed |
| **Gradient Accumulation** | Larger effective batch size |
| **Distributed Training** | Multi-node training |

### Practical Tips
- Start with single GPU, scale when needed
- Use efficient data loading (num_workers, pin_memory)
- Profile to find bottlenecks
- Consider spot/preemptible instances for cost

---

## Question 28: Explain how to perform feature extraction using pretrained models

### Definition
Feature extraction uses a pretrained model as a fixed feature extractor, removing the classification head and using intermediate representations as input to a new classifier.

### Process
1. Load pretrained model
2. Remove classification layer
3. Freeze pretrained weights
4. Add new classifier for your task
5. Train only new layers

### Python Code Example
```python
import torchvision.models as models

# Load pretrained ResNet, remove final layer
resnet = models.resnet50(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# Freeze weights
for param in feature_extractor.parameters():
    param.requires_grad = False

# Add new classifier
model = nn.Sequential(
    feature_extractor,
    nn.Flatten(),
    nn.Linear(2048, num_classes)
)
```

### When to Use
- Limited training data
- Similar domain to pretrained model
- Fast prototyping

---

## Question 29: What are adversarial examples, and why do they pose a threat?

### Definition
Adversarial examples are inputs with small, carefully crafted perturbations that cause neural networks to make incorrect predictions with high confidence, despite being imperceptible to humans.

### How They're Created
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$$

### Threats

| Domain | Risk |
|--------|------|
| Autonomous Vehicles | Misread traffic signs |
| Security | Bypass face recognition |
| Medical | Incorrect diagnosis |
| Finance | Manipulate fraud detection |

### Defenses
- Adversarial training
- Input preprocessing
- Certified robustness
- Ensemble methods

---

## Question 30: What are challenges in training deep reinforcement learning models?

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| **Sample Inefficiency** | Needs millions of environment interactions |
| **Exploration vs Exploitation** | Balancing new vs known actions |
| **Credit Assignment** | Which action caused the reward? |
| **Non-Stationarity** | Distribution changes as policy improves |
| **Hyperparameter Sensitivity** | Very sensitive to tuning |
| **Reproducibility** | High variance across runs |

### Solutions
- Experience replay buffers
- Target networks for stability
- Reward shaping
- Curriculum learning

---

## Question 31: Explain few-shot learning and its significance

### Definition
Few-shot learning enables models to learn new classes from only a few examples (1-10) per class, rather than thousands, by leveraging prior knowledge from related tasks.

### Approaches

| Approach | Description |
|----------|-------------|
| **Metric Learning** | Learn to compare (Siamese, Prototypical) |
| **Meta-Learning** | Learn to learn (MAML) |
| **Transfer Learning** | Fine-tune from pretrained |

### Significance
- Reduces data annotation burden
- Enables rapid adaptation
- Closer to human learning

---

## Question 32: What are zero-shot learning and one-shot learning?

### Definition
**Zero-shot**: Classify classes never seen during training using auxiliary information (attributes, descriptions). **One-shot**: Learn from exactly one example per new class.

### Comparison

| Type | Training Examples | Key Requirement |
|------|-------------------|-----------------|
| **Zero-shot** | 0 per new class | Semantic descriptions |
| **One-shot** | 1 per new class | Good similarity metric |
| **Few-shot** | 2-10 per new class | Meta-learning |

### Zero-Shot Example
- Train on: cat, dog (with attribute vectors)
- Test on: zebra (using attribute: "striped", "horse-like")

---

## Question 33: Relationship between deep learning and computer vision

### Definition
Deep learning revolutionized computer vision by enabling automatic feature learning from raw pixels, replacing hand-crafted features (SIFT, HOG) with learned hierarchical representations.

### Impact

| Era | Approach | Example |
|-----|----------|---------|
| **Pre-2012** | Hand-crafted features | SIFT + SVM |
| **Post-2012** | CNN features | AlexNet, VGG |
| **Current** | Transformers | ViT, CLIP |

### Key Applications
- Image classification
- Object detection (YOLO, Faster R-CNN)
- Semantic segmentation (U-Net, DeepLab)
- Face recognition
- Autonomous driving

---

## Question 34: How does deep learning contribute to speech recognition and synthesis?

### Speech Recognition (ASR)
- **Input**: Audio waveform or spectrogram
- **Architecture**: CNN + Transformer (Whisper, Wav2Vec)
- **Output**: Text transcription

### Speech Synthesis (TTS)
- **Input**: Text
- **Architecture**: Transformer + Vocoder (Tacotron, WaveNet)
- **Output**: Audio waveform

### Key Components

| Component | Models |
|-----------|--------|
| **Encoder** | Extract audio features |
| **Decoder** | Generate text/audio |
| **Vocoder** | Convert spectrograms to audio |

---

## Question 35: Describe reinforcement learning and its connection to deep learning

### Definition
Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward. Deep RL uses neural networks to approximate value functions or policies for complex, high-dimensional problems.

### Key Algorithms

| Algorithm | Neural Network Use |
|-----------|-------------------|
| **DQN** | Approximate Q-function |
| **Policy Gradient** | Parameterize policy directly |
| **Actor-Critic** | Both policy and value networks |
| **PPO** | Stable policy optimization |

### Applications
- Game playing (AlphaGo, Atari)
- Robotics
- Recommendation systems
- Resource management

---

## Question 36: What is multimodal learning in deep learning?

### Definition
Multimodal learning processes and relates information from multiple modalities (text, image, audio, video) to enable cross-modal understanding and generation.

### Examples

| Model | Modalities | Task |
|-------|------------|------|
| **CLIP** | Image + Text | Zero-shot classification |
| **DALL-E** | Text → Image | Image generation |
| **Whisper** | Audio → Text | Speech recognition |
| **GPT-4V** | Image + Text | Visual Q&A |

### Challenges
- Aligning representations across modalities
- Handling missing modalities
- Computational cost

---

## Question 37: Steps to create a recommendation system using deep learning

### Architecture Approaches

| Approach | Description |
|----------|-------------|
| **Neural Collaborative Filtering** | Learn user-item interactions |
| **Content-Based** | Encode item features |
| **Two-Tower** | Separate user and item encoders |
| **Sequential** | Model user history with RNN/Transformer |

### Pipeline
1. **Data**: User-item interactions, item features
2. **Embedding**: Learn user and item embeddings
3. **Interaction**: Combine embeddings (dot product, MLP)
4. **Training**: Binary cross-entropy or ranking loss
5. **Serving**: Approximate nearest neighbor for retrieval

### Python Code Example
```python
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
    def forward(self, user_id, item_id):
        user_vec = self.user_embed(user_id)
        item_vec = self.item_embed(item_id)
        return (user_vec * item_vec).sum(dim=1)  # Dot product
```

---

## Question 38: Approach for sentiment analysis on social media using deep learning

### Pipeline

| Step | Implementation |
|------|----------------|
| **Preprocessing** | Handle mentions, hashtags, emojis |
| **Tokenization** | Subword tokenization (BPE) |
| **Model** | Fine-tuned BERT or DistilBERT |
| **Output** | Positive/Negative/Neutral |

### Preprocessing
```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'http\S+', '<URL>', text)
    return text
```

### Model Approach
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This product is amazing!")
# {'label': 'POSITIVE', 'score': 0.99}
```

---

## Question 39: Significance of ROC curves and AUC in model performance

### Definition
**ROC curve** plots True Positive Rate vs False Positive Rate at various thresholds. **AUC** (Area Under Curve) summarizes discriminative ability as a single number (0.5 = random, 1.0 = perfect).

### Metrics

| Metric | Formula |
|--------|---------|
| **TPR (Recall)** | TP / (TP + FN) |
| **FPR** | FP / (FP + TN) |
| **AUC** | Area under ROC curve |

### Why Use AUC
- Threshold-independent
- Works for imbalanced data
- Single summary metric

### Python Code Example
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
```

---

## Question 40: Methods for model introspection and feature importance

### Techniques

| Method | Description |
|--------|-------------|
| **Saliency Maps** | Gradient with respect to input |
| **Grad-CAM** | Class activation maps for CNNs |
| **SHAP** | Game-theoretic feature attribution |
| **LIME** | Local linear approximation |
| **Attention Visualization** | Show attention weights |
| **Probing** | Test what representations encode |

### Python Code Example
```python
# Simple gradient-based saliency
input_image.requires_grad = True
output = model(input_image)
output[0, target_class].backward()
saliency = input_image.grad.abs()
```

---

## Question 41: What is model explainability, and why is it important?

### Definition
Model explainability is the ability to understand and interpret why a model makes specific predictions, making it trustworthy, debuggable, and compliant with regulations.

### Why Important

| Reason | Context |
|--------|---------|
| **Trust** | Users need to understand predictions |
| **Debugging** | Identify model failures |
| **Regulatory** | GDPR "right to explanation" |
| **Fairness** | Detect bias in decisions |
| **Safety** | Critical applications (medical, legal) |

### Methods

| Method | Type |
|--------|------|
| **SHAP** | Feature attribution |
| **LIME** | Local explanation |
| **Grad-CAM** | Visual explanation |
| **Attention** | Architecture-based |

### Trade-off
- Simpler models (linear, trees) are inherently interpretable
- Deep models need post-hoc explanation methods
- Sometimes trade accuracy for interpretability

---
