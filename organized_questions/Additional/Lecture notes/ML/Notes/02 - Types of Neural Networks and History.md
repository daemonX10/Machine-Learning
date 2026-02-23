# Types of Neural Networks, History & Applications of Deep Learning

## 1. Types of Neural Networks

### 1.1 Artificial Neural Network (ANN) / Multi-Layer Perceptron (MLP)
- Simplest and most fundamental neural network
- Used for **tabular data** — regression and classification
- Fully connected layers (dense layers)
- Can capture **non-linear** relationships by adding more layers

### 1.2 Convolutional Neural Network (CNN)
- Specialized for **image and video** data
- Uses convolution operations to detect spatial features
- Applications: image classification, object detection, medical imaging, self-driving cars

### 1.3 Recurrent Neural Network (RNN)
- Designed for **sequential/temporal** data
- Has memory — output depends on current input + previous hidden state
- Variants:
  - **LSTM** (Long Short-Term Memory) — solves vanishing gradient problem
  - **GRU** (Gated Recurrent Unit) — lighter version of LSTM
- Applications: speech recognition, text generation, time series, NLP

### 1.4 Generative Adversarial Network (GAN)
- Two networks competing: **Generator** vs **Discriminator**
- Can generate realistic images, text, music, and more
- Applications: face generation, image super-resolution, style transfer

### 1.5 Autoencoders
- Learns compressed representation of data
- Used for: dimensionality reduction, denoising, anomaly detection, image compression

### 1.6 Transformers
- Attention-based architecture, no recurrence
- State-of-the-art for NLP tasks
- Basis for BERT, GPT, and modern LLMs

### 1.7 Restricted Boltzmann Machines (RBM)
- Generative stochastic neural network
- Used in recommendation systems and feature learning

---

## 2. History of Deep Learning — Timeline

| Year | Event | Key Person(s) |
|------|-------|---------------|
| **1958** | **Perceptron** invented | Frank Rosenblatt |
| 1960s | Perceptron heavily promoted; early AI optimism | — |
| **1969** | Minsky & Papert publish *Perceptrons* — proved single perceptron **cannot learn XOR** | Marvin Minsky, Seymour Papert |
| 1970s–80s | **AI Winter** — funding and interest dried up | — |
| **1986** | **Backpropagation** paper — showed multi-layer networks can learn any function | Rumelhart, Hinton, Williams |
| 1989 | **CNN** (LeNet) applied to handwriting recognition | Yann LeCun |
| 1990s–2000s | Some progress but limited by compute and data; another slowdown | — |
| **2006** | **Deep Belief Networks** — reignited deep learning research | Geoffrey Hinton |
| **2012** | **AlexNet** wins ImageNet competition by huge margin using GPU-trained CNN | Alex Krizhevsky, Hinton |
| 2014 | **GANs** invented | Ian Goodfellow |
| **2016** | **AlphaGo** defeats world Go champion Lee Sedol 4-1 | DeepMind (Google) |
| 2017 | **Transformer** architecture ("Attention is All You Need") | Vaswani et al. (Google) |
| 2018 | AlphaGo retires — AI deemed unbeatable at Go | DeepMind |
| 2018+ | Explosion of DL in every industry; current era | — |

### Key Historical Insight
- The **XOR problem** (1969) killed single perceptron research
- **Backpropagation** (1986) was the breakthrough — multi-layer networks can approximate **any arbitrary function**
- **AlexNet** (2012) was the inflection point — proved DL + GPU is transformative
- Post-2012: Google, Facebook, and others invested heavily → rapid progress

---

## 3. Applications of Deep Learning

### Computer Vision
| Application | Description |
|-------------|-------------|
| Image Classification | Categorize images (dog vs cat) |
| Object Detection | Locate objects with bounding boxes (YOLO) |
| Image Segmentation | Pixel-level classification |
| Face Recognition | Google Photos auto-tagging faces |
| Image Colorization | Add color to B&W photos |
| Image Super-Resolution | Convert low-quality → high-quality images |
| Photo Restoration | Restore old/damaged photos |

### Natural Language Processing (NLP)
| Application | Description |
|-------------|-------------|
| Machine Translation | Google Translate (real-time) |
| Sentiment Analysis | Classify text as positive/negative |
| Text Generation | Story writing, article generation |
| Question Answering | SQuAD-based systems |
| Chatbots | Conversational AI |

### Speech & Audio
| Application | Description |
|-------------|-------------|
| Speech Recognition | Voice assistants (Siri, Alexa) |
| Text-to-Speech | Generate natural-sounding speech |
| Music Generation | AI-composed music |

### Healthcare & Science
| Application | Description |
|-------------|-------------|
| Medical Imaging | Cancer detection from X-rays, MRI |
| Drug Discovery | Predicting molecular interactions |
| Protein Folding | AlphaFold by DeepMind |

### Other Domains
| Application | Description |
|-------------|-------------|
| Self-Driving Cars | Tesla, Waymo |
| Game Playing | AlphaGo, OpenAI Five (Dota 2) |
| Recommendation Systems | Netflix, YouTube, Spotify |
| Robotics | Industrial automation |
| Climate Science | Weather prediction, modeling |
| Education | Personalized learning |

---

## 4. Key Takeaways
1. Different neural network architectures exist for different data types (images → CNN, sequences → RNN, generation → GAN)
2. DL history spans from 1958 to present — with two **AI winters** and a major revival post-2012
3. The **XOR problem** → **backpropagation** → **multi-layer networks** is the foundational story arc
4. **AlexNet (2012)** was the turning point — GPU + CNN + large data = breakthrough
5. DL is now applied across virtually every industry and domain
