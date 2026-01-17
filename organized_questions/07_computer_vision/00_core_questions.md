# Core Computer Vision Questions

## Fundamentals

### Question 1
**What is computer vision and how does it relate to human vision?**

**Answer:**

Computer vision is a field of AI that enables machines to interpret and understand visual information from images or videos, mimicking how humans perceive and process visual data. It extracts meaningful information from pixels to make decisions or take actions.

**Core Concepts:**
- **Human vision:** Brain processes light signals from retina → extracts edges, textures, shapes → recognizes objects and scenes
- **Computer vision:** Camera captures pixels → algorithms extract features → neural networks recognize patterns → outputs predictions
- **Hierarchy:** Both systems process information hierarchically (low-level edges → mid-level parts → high-level objects)

**Key Differences:**

| Aspect | Human Vision | Computer Vision |
|--------|-------------|-----------------|
| Input | Continuous light signals | Discrete pixel arrays |
| Processing | Parallel, biological neurons | Sequential/parallel, digital |
| Learning | Few examples needed | Often needs large datasets |
| Robustness | Handles novel situations well | Can fail on edge cases |

**Practical Relevance:**
- Autonomous vehicles (scene understanding)
- Medical imaging (diagnosis assistance)
- Surveillance systems (anomaly detection)
- Augmented reality (spatial understanding)

**Interview Tip:** Emphasize that CV attempts to replicate human vision capabilities but uses fundamentally different mechanisms (mathematical transformations vs. biological processes).

---

### Question 2
**Describe the key components of a computer vision system.**

**Answer:**

A computer vision system consists of interconnected components: image acquisition (camera/sensor), preprocessing (noise removal, normalization), feature extraction (edges, textures, learned features), model/algorithm (CNN, detector), and output/decision module that produces final predictions or actions.

**Core Components:**

1. **Image Acquisition**
   - Camera/sensor captures raw image data
   - Considerations: resolution, frame rate, lighting conditions

2. **Preprocessing**
   - Noise reduction (Gaussian blur, median filter)
   - Normalization (pixel value scaling to [0,1] or standardization)
   - Resizing, color space conversion (RGB→BGR, HSV)

3. **Feature Extraction**
   - Traditional: SIFT, HOG, edge detectors
   - Deep learning: CNN layers automatically learn features

4. **Model/Algorithm**
   - Classification: CNNs (ResNet, EfficientNet)
   - Detection: YOLO, Faster R-CNN
   - Segmentation: U-Net, Mask R-CNN

5. **Post-processing**
   - NMS for detection, CRF for segmentation
   - Thresholding, filtering predictions

6. **Output/Decision**
   - Class labels, bounding boxes, masks, actions

**Pipeline Flow:**
```
Camera → Preprocessing → Feature Extraction → Model Inference → Post-processing → Output
```

**Practical Relevance:** Understanding this pipeline helps debug issues at each stage and optimize bottlenecks in production systems.

---

### Question 3
**What is the difference between image processing and computer vision?**

**Answer:**

Image processing transforms images into other images (input: image → output: image), focusing on enhancement, filtering, and manipulation. Computer vision extracts understanding and meaning from images (input: image → output: decision/information) to interpret visual content like humans do.

**Core Differences:**

| Aspect | Image Processing | Computer Vision |
|--------|-----------------|-----------------|
| **Goal** | Transform/enhance images | Understand/interpret images |
| **Output** | Modified image | Labels, coordinates, decisions |
| **Examples** | Sharpening, denoising, resizing | Object detection, classification |
| **Level** | Low-level operations | High-level understanding |
| **Human analogy** | Adjusting brightness on TV | Recognizing who's on TV |

**Image Processing Tasks:**
- Filtering (blur, sharpen, edge detection)
- Geometric transformations (rotation, scaling)
- Color manipulation (histogram equalization)
- Compression (JPEG encoding)

**Computer Vision Tasks:**
- Object detection and recognition
- Image classification
- Semantic segmentation
- Pose estimation

**Relationship:**
- Image processing is often a **preprocessing step** for computer vision
- CV systems use image processing techniques internally
- Both operate on pixel data but with different end goals

**Interview Tip:** Image processing = pixel manipulation; Computer vision = scene understanding. They complement each other in real pipelines.

---

## Image Analysis & Processing

### Question 4
**How does edge detection work in image analysis?**

**Answer:**

Edge detection identifies boundaries in images by detecting discontinuities in pixel intensity. It uses gradient-based operators (Sobel, Prewitt) or second-derivative methods (Laplacian, Canny) to find locations where intensity changes rapidly, indicating object boundaries or texture changes.

**Core Concepts:**
- **Edge:** Sharp change in intensity (high gradient magnitude)
- **Gradient:** Rate of change in intensity across pixels
- **Direction:** Edges have orientation perpendicular to gradient direction

**Mathematical Formulation:**

Gradient magnitude and direction:
$$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

Sobel operators:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$

**Common Edge Detectors:**

| Method | Approach | Pros/Cons |
|--------|----------|-----------|
| Sobel | First derivative | Simple, sensitive to noise |
| Canny | Multi-stage (smooth→gradient→NMS→hysteresis) | Best results, more complex |
| Laplacian | Second derivative | Detects edges, very noise-sensitive |

**Canny Edge Detection Steps:**
1. Gaussian smoothing to reduce noise
2. Compute gradient magnitude and direction
3. Non-maximum suppression (thin edges)
4. Hysteresis thresholding (connect edges)

**Python Example:**
```python
import cv2
import numpy as np

# Load grayscale image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel edges
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

# Canny edges (recommended)
canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)
```

**Practical Relevance:** Edge detection is used for object boundary detection, lane detection in autonomous vehicles, and as preprocessing for contour extraction.

---

### Question 5
**What are the common image preprocessing steps in a computer vision pipeline?**

**Answer:**

Common preprocessing steps include: resizing (standardize dimensions), normalization (scale pixel values), noise reduction (Gaussian/median filtering), color space conversion, data augmentation (flips, rotations), and histogram equalization (contrast enhancement). These prepare raw images for model consumption.

**Core Preprocessing Steps:**

1. **Resizing/Rescaling**
   - Match model input dimensions (e.g., 224×224 for ResNet)
   - Maintain aspect ratio with padding if needed

2. **Normalization**
   - Scale to [0, 1]: `pixel / 255.0`
   - Standardize: `(pixel - mean) / std` (ImageNet: mean=[0.485, 0.456, 0.406])

3. **Noise Reduction**
   - Gaussian blur: smooths image, reduces high-frequency noise
   - Median filter: removes salt-and-pepper noise

4. **Color Space Conversion**
   - RGB → BGR (OpenCV default)
   - RGB → Grayscale (reduce dimensionality)
   - RGB → HSV (color-based segmentation)

5. **Data Augmentation**
   - Geometric: flip, rotate, crop, scale
   - Photometric: brightness, contrast, saturation
   - Advanced: mixup, cutout, mosaic

6. **Histogram Equalization**
   - Enhances contrast in low-contrast images
   - CLAHE for adaptive local enhancement

**Python Example:**
```python
import cv2
import numpy as np

# Load image
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img_resized = cv2.resize(img, (224, 224))

# Normalize to [0, 1]
img_normalized = img_resized / 255.0

# Standardize (ImageNet stats)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_standardized = (img_normalized - mean) / std

# Denoise
img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
```

**Interview Tip:** Always mention normalization matching the pretrained model's training distribution. Mismatched normalization is a common bug.

---

### Question 6
**How does image resizing affect model performance?**

**Answer:**

Image resizing affects model performance through information loss (downscaling removes fine details), aspect ratio distortion (squeezing can confuse models), and distribution shift (resized images differ from training data). Larger input sizes generally improve accuracy but increase computational cost quadratically.

**Core Concepts:**

**Effects of Downscaling:**
- Loss of fine-grained details and small objects
- Small objects may become undetectable
- Faster inference, lower memory usage

**Effects of Upscaling:**
- Introduces interpolation artifacts
- No new information added
- Increased compute without accuracy gain

**Aspect Ratio Handling:**

| Method | Approach | Trade-off |
|--------|----------|-----------|
| Stretch | Force to target size | Distorts objects |
| Crop | Cut to target ratio | Loses content |
| Pad | Add borders | Wastes compute on padding |
| Letterbox | Pad to maintain ratio | Best balance |

**Resolution vs. Accuracy Trade-off:**
- Higher resolution → better small object detection
- Computational cost scales as O(n²) with image dimension
- Common sizes: 224×224 (classification), 640×640 (YOLO)

**Python Example:**
```python
import cv2

img = cv2.imread('image.jpg')

# Method 1: Stretch (may distort)
resized_stretch = cv2.resize(img, (224, 224))

# Method 2: Letterbox (maintain aspect ratio)
def letterbox(img, target_size=224):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to target size
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

resized_letterbox = letterbox(img, 224)
```

**Interview Tip:** For detection tasks, mention letterboxing to preserve aspect ratios. For classification, match the pretrained model's training resolution.

---

### Question 7
**What are some techniques to reduce noise in an image?**

**Answer:**

Noise reduction techniques include: Gaussian blur (smooths by averaging neighbors), median filter (replaces with median value, good for salt-and-pepper noise), bilateral filter (edge-preserving smoothing), and deep learning denoisers (learned filters). Choice depends on noise type and edge preservation requirements.

**Core Techniques:**

| Filter | Best For | Edge Preservation |
|--------|----------|-------------------|
| Gaussian Blur | General smoothing | Poor |
| Median Filter | Salt-and-pepper noise | Moderate |
| Bilateral Filter | General noise | Good |
| Non-local Means | General noise | Very Good |

**Mathematical Formulation:**

Gaussian filter:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Bilateral filter (combines spatial and intensity):
$$I_{filtered}(x) = \frac{1}{W} \sum_{x_i \in \Omega} I(x_i) \cdot f_s(||x_i - x||) \cdot f_r(|I(x_i) - I(x)|)$$

Where $f_s$ = spatial kernel, $f_r$ = range (intensity) kernel

**Noise Types:**
- **Gaussian noise:** Random intensity variations (use Gaussian blur)
- **Salt-and-pepper:** Random black/white pixels (use median filter)
- **Speckle:** Multiplicative noise (use bilateral or adaptive filters)

**Python Example:**
```python
import cv2

img = cv2.imread('noisy_image.jpg')

# Gaussian Blur (kernel size must be odd)
gaussian = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

# Median Filter (good for salt-and-pepper)
median = cv2.medianBlur(img, 5)

# Bilateral Filter (preserves edges)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Non-local Means (best quality, slower)
nlm = cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=7)
```

**Interview Tip:** Median filter is best for salt-and-pepper noise; bilateral filter when edge preservation matters.

---

### Question 8
**Explain how image augmentation can improve the performance of a vision model.**

**Answer:**

Image augmentation artificially expands training data by applying transformations (flips, rotations, color jittering) to create varied versions of images. This improves model generalization by exposing it to diverse conditions, reduces overfitting, and makes models invariant to transformations they may encounter in real-world deployment.

**Core Concepts:**
- **Data Diversity:** Model sees more variations without collecting new data
- **Regularization Effect:** Prevents memorization of training samples
- **Invariance Learning:** Model learns to ignore irrelevant variations

**Types of Augmentation:**

| Category | Techniques | Purpose |
|----------|------------|---------|
| Geometric | Flip, rotate, crop, scale | Position/orientation invariance |
| Photometric | Brightness, contrast, saturation | Lighting invariance |
| Noise | Gaussian noise, blur | Robustness to noise |
| Advanced | Mixup, CutMix, Mosaic | Better generalization |

**Common Augmentations:**
- **Horizontal Flip:** Doubles data, useful for most objects
- **Random Crop:** Forces model to recognize partial objects
- **Color Jitter:** Robustness to lighting variations
- **Cutout/Random Erasing:** Occlusion robustness

**Python Example:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Apply to image
augmented = transform(image=image)
augmented_image = augmented['image']
```

**Interview Tip:** 
- Only use augmentations that make semantic sense (don't flip text, digits)
- Augmentation is applied during training only, not inference
- For detection/segmentation, transform bounding boxes/masks too

---

### Question 9
**Discuss the concept of color spaces and their importance in image processing.**

**Answer:**

Color spaces are mathematical models that represent colors as tuples of numbers. Different color spaces (RGB, HSV, LAB) organize color information differently, making certain tasks easier. RGB is standard for displays; HSV separates color (hue) from intensity; LAB is perceptually uniform for color comparison.

**Core Color Spaces:**

| Color Space | Components | Best For |
|-------------|------------|----------|
| **RGB** | Red, Green, Blue | Display, neural networks |
| **BGR** | Blue, Green, Red | OpenCV default |
| **HSV** | Hue, Saturation, Value | Color-based segmentation |
| **LAB** | Lightness, A, B | Color comparison, correction |
| **Grayscale** | Intensity | Edge detection, reduce complexity |
| **YCbCr** | Luma, Chroma | Video compression |

**Why Color Spaces Matter:**

1. **HSV for Color Segmentation:**
   - Hue separates color from brightness
   - Easy to threshold specific colors (e.g., detect red objects)

2. **LAB for Color Matching:**
   - Perceptually uniform: equal distances = equal perceived differences
   - Used in color transfer and histogram matching

3. **Grayscale for Efficiency:**
   - Reduces 3 channels → 1 channel
   - Edge detection, document analysis

**Python Example:**
```python
import cv2

# Load RGB image (OpenCV loads as BGR)
img_bgr = cv2.imread('image.jpg')

# Convert to different color spaces
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Color segmentation in HSV (detect red)
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
mask = cv2.inRange(img_hsv, lower_red, upper_red)
```

**Interview Tip:** Use HSV when you need to isolate colors regardless of lighting; use RGB/BGR for deep learning models.

---

## Segmentation

### Question 10
**Explain the concept of image segmentation in computer vision.**

**Answer:**

Image segmentation partitions an image into meaningful regions by assigning a label to every pixel. It goes beyond classification (one label per image) to provide pixel-level understanding. Types include semantic segmentation (class per pixel), instance segmentation (separate objects), and panoptic segmentation (combines both).

**Core Concepts:**

| Type | Output | Example |
|------|--------|---------|
| **Semantic** | Class label per pixel | All cars = same label |
| **Instance** | Separate each object | Car1, Car2, Car3 |
| **Panoptic** | Both semantic + instance | Background classes + instances |

**Segmentation Approaches:**

1. **Traditional Methods:**
   - Thresholding (binary segmentation)
   - Region growing (expand from seeds)
   - Watershed (treat gradients as topography)

2. **Deep Learning Methods:**
   - **FCN:** Fully convolutional for dense prediction
   - **U-Net:** Encoder-decoder with skip connections
   - **DeepLab:** Atrous convolutions for multi-scale
   - **Mask R-CNN:** Detection + instance masks

**Mathematical View:**
Given image $I$ with pixels $p$, segmentation produces label map $L$:
$$L(p) = \arg\max_c P(class = c | p, I)$$

**Python Example:**
```python
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# Load pretrained model
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Preprocess image (add batch dim, normalize)
# input_tensor shape: [1, 3, H, W]

with torch.no_grad():
    output = model(input_tensor)['out']  # [1, num_classes, H, W]
    pred_mask = output.argmax(dim=1)     # [1, H, W] - class per pixel
```

**Practical Relevance:**
- Medical imaging (tumor segmentation)
- Autonomous driving (road, pedestrians, vehicles)
- Satellite imagery (land use classification)

---

### Question 11
**What is the difference between semantic and instance segmentation?**

**Answer:**

Semantic segmentation assigns a class label to every pixel but treats all instances of a class identically (all cars get same label). Instance segmentation differentiates individual objects of the same class (car1, car2, car3 as separate entities). Instance = semantic + object detection.

**Core Differences:**

| Aspect | Semantic Segmentation | Instance Segmentation |
|--------|----------------------|----------------------|
| **Output** | Class label per pixel | Object ID + class per pixel |
| **Same-class objects** | Merged together | Separated individually |
| **Architecture** | FCN, U-Net, DeepLab | Mask R-CNN, YOLACT |
| **Use case** | Scene parsing, road segmentation | Counting, tracking objects |

**Visual Example:**
```
Image: Two cars and background

Semantic:     [Background=0, Car=1, Car=1]  (both cars = label 1)
Instance:     [Background=0, Car_1=1, Car_2=2]  (separate IDs)
```

**Architectures:**

**Semantic Segmentation:**
- Fully Convolutional Networks (FCN)
- U-Net (encoder-decoder)
- DeepLabv3+ (atrous convolutions)

**Instance Segmentation:**
- Mask R-CNN (Faster R-CNN + mask head)
- YOLACT (single-stage)
- SOLOv2 (segment objects by locations)

**Panoptic Segmentation:**
- Combines both: instance labels for "things" (countable objects)
- Semantic labels for "stuff" (uncountable: sky, road)

**Practical Relevance:**
- **Semantic:** Autonomous driving road parsing, medical tissue classification
- **Instance:** Object counting, multi-object tracking, robotics grasping

**Interview Tip:** Instance segmentation is harder because it requires detecting and separating overlapping objects of the same class.

---

### Question 12
**Explain the Fully Convolutional Network (FCN) and its role in segmentation.**

**Answer:**

FCN replaces fully connected layers in classification CNNs with convolutional layers, enabling dense pixel-wise predictions for any input size. It pioneered end-to-end segmentation by using upsampling (transposed convolutions) to restore spatial resolution after feature extraction, producing output masks matching input dimensions.

**Core Concepts:**

**Key Innovation:**
- Classification CNNs: Conv → FC → single label
- FCN: Conv → Conv (1×1) → Upsample → pixel-wise labels

**Architecture Components:**

1. **Encoder (Downsampling):**
   - Pretrained backbone (VGG, ResNet)
   - Extracts hierarchical features
   - Reduces spatial resolution

2. **1×1 Convolution:**
   - Replaces FC layers
   - Converts feature maps to class scores

3. **Decoder (Upsampling):**
   - Transposed convolution / bilinear interpolation
   - Restores original resolution
   - Skip connections for fine details

**FCN Variants:**
- **FCN-32s:** Single 32× upsample (coarse)
- **FCN-16s:** Combines pool4 + 2× upsample of FCN-32s
- **FCN-8s:** Adds pool3, finest predictions

**Mathematical View:**
$$\text{Output} = \text{Upsample}(\text{Conv}_{1\times1}(\text{Encoder}(I)))$$

**Python Example:**
```python
import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder (simplified)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /4
        )
        # 1x1 conv for classification
        self.classifier = nn.Conv2d(128, num_classes, 1)
        # Upsample back to original size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
    
    def forward(self, x):
        features = self.encoder(x)
        scores = self.classifier(features)
        output = self.upsample(scores)
        return output  # [B, num_classes, H, W]
```

**Practical Relevance:** FCN is the foundation for all modern segmentation networks (U-Net, DeepLab). Understanding it is essential.

---

## Feature Extraction & Detection

### Question 13
**What are feature descriptors, and why are they important in computer vision?**

**Answer:**

Feature descriptors are compact numerical representations that describe distinctive image regions (keypoints). They encode local patterns like edges, gradients, and textures in a way that's invariant to transformations. They enable matching, recognition, and tracking by comparing descriptor vectors instead of raw pixels.

**Core Concepts:**

**Two-Stage Process:**
1. **Detection:** Find interesting keypoints (corners, blobs)
2. **Description:** Compute descriptor vector for each keypoint

**Properties of Good Descriptors:**
- **Distinctiveness:** Different regions produce different descriptors
- **Invariance:** Robust to rotation, scale, illumination
- **Compactness:** Low-dimensional for efficient matching
- **Repeatability:** Same point detected across views

**Common Feature Descriptors:**

| Descriptor | Dimension | Invariance | Speed |
|------------|-----------|------------|-------|
| SIFT | 128 | Scale, rotation | Slow |
| SURF | 64 | Scale, rotation | Medium |
| ORB | 256 (binary) | Rotation | Fast |
| BRIEF | 256 (binary) | None | Very Fast |

**Matching Process:**
1. Extract keypoints and descriptors from both images
2. For each descriptor in image1, find nearest neighbor in image2
3. Apply ratio test to filter false matches
4. Use RANSAC to remove outliers

**Python Example:**
```python
import cv2

# Load images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Create ORB detector (fast, patent-free)
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors
kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

**Practical Relevance:** Image stitching, visual SLAM, object recognition, augmented reality marker tracking.

---

### Question 14
**Explain the Scale-Invariant Feature Transform (SIFT) algorithm.**

**Answer:**

SIFT detects keypoints that are invariant to scale and rotation, then computes 128-dimensional descriptors for each. It builds a scale-space pyramid using Difference of Gaussians (DoG) to find keypoints, assigns dominant orientations, and describes local gradients in orientation histograms.

**Algorithm Steps:**

**1. Scale-Space Extrema Detection:**
- Build Gaussian pyramid at multiple scales: $L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$
- Compute DoG: $D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$
- Find local extrema (keypoint candidates) in 3×3×3 neighborhood

**2. Keypoint Localization:**
- Refine location using Taylor expansion
- Reject low-contrast points and edge responses (using Hessian ratio)

**3. Orientation Assignment:**
- Compute gradient magnitude and orientation in region around keypoint
- Build histogram of orientations (36 bins)
- Assign dominant orientation → rotation invariance

**4. Descriptor Generation:**
- Take 16×16 window around keypoint
- Divide into 4×4 grid of cells
- Compute 8-bin orientation histogram per cell
- Result: 4×4×8 = **128-dimensional vector**
- Normalize descriptor for illumination invariance

**Mathematical Formulation:**
Gradient magnitude and orientation:
$$m(x,y) = \sqrt{(L_{x+1,y} - L_{x-1,y})^2 + (L_{x,y+1} - L_{x,y-1})^2}$$
$$\theta(x,y) = \arctan\left(\frac{L_{x,y+1} - L_{x,y-1}}{L_{x+1,y} - L_{x-1,y}}\right)$$

**Python Example:**
```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)

# descriptors shape: (num_keypoints, 128)
print(f"Found {len(keypoints)} keypoints")
print(f"Descriptor shape: {descriptors.shape}")
```

**Interview Tip:** SIFT is patented (use ORB for free alternative). Key insight: DoG approximates Laplacian of Gaussian for scale-space blob detection.

---

### Question 15
**Describe how the Histogram of Oriented Gradients (HOG) descriptor works.**

**Answer:**

HOG captures local shape information by computing gradient orientations in dense grid cells and accumulating them into histograms. It divides the image into cells, computes oriented gradient histograms per cell, normalizes across blocks of cells, and concatenates into a feature vector for object detection.

**Algorithm Steps:**

**1. Preprocessing:**
- Convert to grayscale (optional)
- Resize to fixed dimensions (e.g., 64×128 for pedestrians)
- Gamma correction for lighting normalization

**2. Gradient Computation:**
- Compute horizontal and vertical gradients: $G_x = I * [-1, 0, 1]$, $G_y = I * [-1, 0, 1]^T$
- Magnitude: $|G| = \sqrt{G_x^2 + G_y^2}$
- Orientation: $\theta = \arctan(G_y / G_x)$

**3. Cell Histograms:**
- Divide image into small cells (e.g., 8×8 pixels)
- Create histogram of 9 orientation bins (0°-180°, unsigned)
- Vote weighted by gradient magnitude

**4. Block Normalization:**
- Group cells into overlapping blocks (e.g., 2×2 cells)
- L2-normalize each block for illumination invariance
- Slide block across image

**5. Feature Vector:**
- Concatenate all normalized block histograms
- For 64×128 image: 7×15 blocks × 4 cells × 9 bins = **3780 features**

**Python Example:**
```python
import cv2
from skimage.feature import hog

img = cv2.imread('person.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 128))

# Compute HOG features
features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

print(f"HOG feature vector length: {len(features)}")
```

**Practical Relevance:** HOG + SVM was the classic pedestrian detection method (Dalal & Triggs, 2005). Now largely replaced by deep learning but still useful for understanding gradient-based features.

---

### Question 16
**Compare different image feature extraction methods.**

**Answer:**

Feature extraction methods range from handcrafted (SIFT, HOG, ORB) to learned (CNN features). Handcrafted methods are interpretable and work without training data but lack flexibility. CNN features are learned from data, capture hierarchical patterns, and dominate modern vision tasks.

**Comparison Table:**

| Method | Type | Dimension | Speed | Invariance | Best For |
|--------|------|-----------|-------|------------|----------|
| **SIFT** | Keypoint | 128 | Slow | Scale, rotation | Matching, stitching |
| **SURF** | Keypoint | 64 | Medium | Scale, rotation | Real-time matching |
| **ORB** | Keypoint | 256 (binary) | Fast | Rotation | Mobile/embedded |
| **HOG** | Dense | ~3780 | Medium | - | Object detection |
| **LBP** | Dense | 256 | Fast | Illumination | Texture, faces |
| **CNN** | Learned | 512-2048 | GPU-dependent | Learned | General-purpose |

**Handcrafted vs. Learned:**

| Aspect | Handcrafted | Deep Learning |
|--------|-------------|---------------|
| Design | Human-engineered | Data-driven |
| Training | Not needed | Requires large dataset |
| Interpretability | High | Low |
| Flexibility | Limited | High |
| Performance | Good on specific tasks | State-of-the-art |

**When to Use What:**
- **SIFT/ORB:** Image matching, homography estimation, SLAM
- **HOG:** When deep learning isn't available, embedded systems
- **CNN features:** Classification, detection, general feature extraction

**Python Example - CNN Feature Extraction:**
```python
import torch
from torchvision import models, transforms

# Load pretrained ResNet
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features
with torch.no_grad():
    features = model(input_tensor)  # [1, 2048, 1, 1]
    features = features.flatten()   # [2048]
```

**Interview Tip:** CNN features from pretrained models (ImageNet) often outperform handcrafted features even for tasks not related to classification.

---

### Question 17
**What are Haar Cascades, and how are they used for object detection?**

**Answer:**

Haar Cascades are a machine learning-based object detection method using simple rectangular features (Haar-like features) and a cascade of AdaBoost classifiers. They scan images at multiple scales, quickly rejecting non-object regions in early cascade stages, making real-time detection possible (Viola-Jones framework).

**Core Concepts:**

**Haar-like Features:**
- Simple rectangular filters that compute intensity differences
- Types: edge, line, and center-surround features
- Fast computation using **integral images**

**Integral Image:**
$$II(x,y) = \sum_{x' \leq x, y' \leq y} I(x', y')$$
Any rectangular sum computed in O(1) using 4 corner lookups.

**Cascade Architecture:**
```
Image Region → Stage 1 → Stage 2 → Stage 3 → ... → Detection
                  ↓          ↓          ↓
              Reject     Reject     Reject
```
- Each stage is a strong classifier (AdaBoost ensemble)
- Early stages reject obvious negatives quickly
- Later stages handle harder cases

**Algorithm Steps:**
1. Compute integral image for fast feature calculation
2. Slide detection window across image at multiple scales
3. For each window, pass through cascade stages
4. If all stages pass, mark as detection
5. Apply non-maximum suppression to merge overlapping boxes

**Python Example:**
```python
import cv2

# Load pretrained Haar Cascade (face detection)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load image
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # Scale pyramid step
    minNeighbors=5,    # Robustness threshold
    minSize=(30, 30)   # Minimum object size
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**Practical Relevance:** Haar Cascades are fast and work on CPU, suitable for embedded systems. However, deep learning (YOLO, SSD) provides better accuracy for complex detection tasks.

---

### Question 18
**What's the significance of depth perception in computer vision applications?**

**Answer:**

Depth perception enables 3D scene understanding by estimating distance from camera to objects. It's crucial for robotics (navigation, grasping), autonomous vehicles (obstacle avoidance), AR/VR (spatial mapping), and 3D reconstruction. Methods include stereo vision, structured light, ToF sensors, and monocular depth estimation.

**Core Concepts:**

**Depth Acquisition Methods:**

| Method | Principle | Range | Accuracy |
|--------|-----------|-------|----------|
| **Stereo Vision** | Triangulation from 2 cameras | Medium | Moderate |
| **Structured Light** | Project patterns, analyze distortion | Short | High |
| **Time-of-Flight (ToF)** | Measure light travel time | Medium | High |
| **LiDAR** | Laser scanning | Long | Very High |
| **Monocular (DL)** | CNN learns depth from single image | Any | Moderate |

**Stereo Vision Math:**
$$Z = \frac{f \cdot B}{d}$$
Where: $Z$ = depth, $f$ = focal length, $B$ = baseline, $d$ = disparity

**Applications:**

1. **Autonomous Vehicles:**
   - Obstacle detection and distance estimation
   - Path planning and collision avoidance

2. **Robotics:**
   - Object grasping and manipulation
   - SLAM (Simultaneous Localization and Mapping)

3. **AR/VR:**
   - Scene reconstruction
   - Object occlusion handling

4. **Medical:**
   - Surgical navigation
   - 3D organ modeling

**Python Example - Monocular Depth Estimation:**
```python
import torch

# Load MiDaS depth estimation model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Estimate depth
input_batch = transform(img)
with torch.no_grad():
    depth = model(input_batch)  # Relative depth map
```

**Interview Tip:** RGB-D sensors (Kinect, RealSense) combine color and depth, providing rich 3D information for indoor applications.

---

### Question 19
**Explain the challenges of object recognition in varied lighting and orientations.**

**Answer:**

Varied lighting causes appearance changes (shadows, highlights, color shifts) while different orientations alter object shape in images. These variations make features inconsistent between training and test images. Solutions include data augmentation, illumination-invariant features, and rotation-equivariant networks.

**Lighting Challenges:**

| Issue | Effect | Solution |
|-------|--------|----------|
| Shadows | Hide features, create false edges | Shadow removal, HDR imaging |
| Overexposure | Loss of detail in bright regions | Histogram equalization, HDR |
| Color shifts | Different color appearance | Color normalization, white balance |
| Low light | Noise, reduced contrast | Denoising, low-light enhancement |

**Orientation Challenges:**

| Issue | Effect | Solution |
|-------|--------|----------|
| Rotation | Features at different angles | Rotation augmentation, rotation-invariant features |
| Scale | Objects appear larger/smaller | Multi-scale detection, scale augmentation |
| Viewpoint | 3D shape changes | 3D-aware models, multiple training views |
| Deformation | Non-rigid shape changes | Deformable convolutions, spatial transformer |

**Solutions:**

1. **Data Augmentation:**
   - Random brightness, contrast, saturation
   - Rotation, flipping, perspective transforms

2. **Normalization:**
   - Histogram equalization for lighting
   - Instance normalization in neural networks

3. **Robust Features:**
   - SIFT/ORB: rotation/scale invariant
   - CNN: learns invariance from data

4. **Specialized Architectures:**
   - Spatial Transformer Networks
   - Group equivariant CNNs

**Python Example:**
```python
import albumentations as A

# Augmentation for lighting and orientation robustness
transform = A.Compose([
    # Lighting variations
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
    # Orientation variations
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.3),
])
```

**Interview Tip:** Heavy augmentation during training makes models robust at inference. Test-time augmentation (TTA) can also improve predictions.

---

## Neural Networks & Deep Learning

### Question 20
**Discuss the role of convolutional neural networks (CNNs) in computer vision.**

**Answer:**

CNNs are the foundation of modern computer vision, automatically learning hierarchical features from raw pixels through convolutional layers. They capture spatial patterns (edges → textures → parts → objects) via local receptive fields and weight sharing, achieving state-of-the-art results in classification, detection, and segmentation.

**Core Concepts:**

**Why CNNs for Vision:**
- **Local connectivity:** Exploit spatial structure of images
- **Weight sharing:** Same filter applied across image → parameter efficiency
- **Translation equivariance:** Detect features regardless of position
- **Hierarchical learning:** Low-level → high-level features automatically

**Feature Hierarchy:**
```
Input → Conv1 (edges) → Conv2 (textures) → Conv3 (parts) → Conv4 (objects)
```

**Key Components:**

| Layer | Purpose |
|-------|---------|
| Convolution | Extract local features |
| Activation (ReLU) | Non-linearity |
| Pooling | Downsample, increase invariance |
| Batch Normalization | Stabilize training |
| Fully Connected | Classification head |

**Impact on Computer Vision:**

| Task | Before CNNs | With CNNs |
|------|-------------|-----------|
| Classification | SIFT + SVM (~74% ImageNet) | ResNet (~96% ImageNet) |
| Detection | HOG + DPM | YOLO, Faster R-CNN |
| Segmentation | Graph cuts | U-Net, DeepLab |

**Python Example:**
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 64, H/4, W/4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # For 224x224 input
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Interview Tip:** CNNs learn what features to extract (end-to-end), unlike traditional methods where features are hand-designed.

---

### Question 21
**How do CNNs differ from traditional neural networks in architecture?**

**Answer:**

CNNs use convolutional layers with local receptive fields and shared weights instead of fully connected layers. This exploits image spatial structure, dramatically reduces parameters (e.g., 3×3 filter vs. connecting every pixel), and provides translation equivariance. Traditional MLPs treat inputs as flat vectors, ignoring spatial relationships.

**Key Architectural Differences:**

| Aspect | Traditional NN (MLP) | CNN |
|--------|---------------------|-----|
| Input | Flattened vector | 2D/3D tensor (H×W×C) |
| Connectivity | Fully connected | Local (receptive field) |
| Weights | Unique per connection | Shared (same filter everywhere) |
| Parameters | Very high | Much fewer |
| Spatial info | Lost | Preserved |

**Parameter Comparison:**

For 224×224×3 input to 1000 hidden units:
- **MLP:** 224 × 224 × 3 × 1000 = **150 million parameters**
- **CNN (3×3 conv, 64 filters):** 3 × 3 × 3 × 64 = **1,728 parameters**

**Mathematical Formulation:**

**Fully Connected:**
$$y_j = \sigma\left(\sum_{i=1}^{n} w_{ij} x_i + b_j\right)$$

**Convolution:**
$$y_{i,j} = \sigma\left(\sum_{m}\sum_{n} w_{m,n} \cdot x_{i+m, j+n} + b\right)$$

**Key CNN Properties:**
1. **Local Connectivity:** Each neuron connected to small region
2. **Weight Sharing:** Same filter applied everywhere
3. **Translation Equivariance:** Shifting input shifts output
4. **Hierarchical Features:** Stacked convolutions = larger receptive field

**Visual Comparison:**
```
MLP:        [Input: 50176] → [FC: 1000] → [FC: 10]
             Every pixel connected to every neuron

CNN:        [Input: 224×224×3] → [Conv: 64 3×3 filters] → [Pool] → [Conv] → ...
             Local filters slide across spatial dimensions
```

**Interview Tip:** Weight sharing in CNNs provides built-in regularization and translation equivariance, making them ideal for vision tasks.

---

### Question 22
**Explain the purpose of pooling layers in a CNN.**

**Answer:**

Pooling layers downsample feature maps by summarizing regions (max or average), reducing spatial dimensions while retaining important features. This decreases computational cost, provides translation invariance (small shifts don't change pooled output), and increases receptive field without adding parameters.

**Core Concepts:**

**Types of Pooling:**

| Type | Operation | Use Case |
|------|-----------|----------|
| Max Pooling | Take maximum value | Most common, preserves strong features |
| Average Pooling | Take average value | Smoother, global context |
| Global Average Pooling | Average entire feature map | Replace FC layer at end |

**Mathematical Formulation:**

For 2×2 pooling with stride 2:
$$\text{MaxPool}(x)_{i,j} = \max(x_{2i,2j}, x_{2i+1,2j}, x_{2i,2j+1}, x_{2i+1,2j+1})$$

**Benefits:**
1. **Dimensionality Reduction:** H×W → H/2 × W/2 (with 2×2 pool, stride 2)
2. **Translation Invariance:** Small shifts produce same pooled output
3. **Computational Efficiency:** Fewer pixels to process downstream
4. **Larger Receptive Field:** Each output pixel "sees" more of input

**Example:**
```
Input 4×4:          MaxPool 2×2:
[1  2  5  6]        
[3  4  7  8]   →    [4   8]
[9  10 13 14]       [12  16]
[11 12 15 16]
```

**Python Example:**
```python
import torch
import torch.nn as nn

# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Global average pooling (common before final classification)
global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: [B, C, 1, 1]

x = torch.randn(1, 64, 224, 224)  # [B, C, H, W]
out_max = max_pool(x)              # [1, 64, 112, 112]
out_global = global_avg_pool(x)    # [1, 64, 1, 1]
```

**Interview Tip:** Modern architectures often use strided convolutions instead of pooling (more learnable). Global Average Pooling replaced FC layers in networks like ResNet.

---

### Question 23
**What is transfer learning, and when would you apply it in computer vision?**

**Answer:**

Transfer learning uses a model pretrained on a large dataset (like ImageNet) as a starting point for a new task. The pretrained weights capture general visual features (edges, textures, shapes) that transfer to new domains. Apply it when you have limited data, similar domain to pretrain data, or need faster training.

**Core Concepts:**

**Why It Works:**
- Early CNN layers learn general features (edges, corners)
- Later layers learn task-specific features
- General features transfer across tasks

**Transfer Learning Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature Extraction** | Freeze pretrained layers, train only new head | Very small dataset, similar domain |
| **Fine-tuning** | Unfreeze some/all layers, train with low LR | Medium dataset, related domain |
| **Full Training** | Use pretrained weights as initialization | Large dataset |

**When to Apply:**
- Limited labeled data (< 10,000 images)
- Similar visual domain to pretrain dataset
- Faster convergence needed
- Resource constraints

**When NOT to Apply:**
- Vastly different domain (e.g., ImageNet → X-ray)
- Sufficient data to train from scratch
- Privacy concerns with pretrained models

**Python Example:**
```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Strategy 1: Feature extraction (freeze backbone)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for new task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Strategy 2: Fine-tuning (unfreeze later layers)
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Use lower learning rate for pretrained layers
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

**Interview Tip:** Start with frozen backbone, train head, then gradually unfreeze layers from top to bottom if more data is available.

---

### Question 24
**Discuss using pre-trained models in computer vision.**

**Answer:**

Pre-trained models are neural networks trained on large datasets (ImageNet, COCO) whose learned weights can be reused. They provide strong feature extractors out-of-the-box, reduce training time/data requirements, and often outperform training from scratch. Common sources: torchvision, timm, TensorFlow Hub.

**Core Concepts:**

**Popular Pre-trained Models:**

| Model | Parameters | Top-1 Acc | Use Case |
|-------|------------|-----------|----------|
| ResNet-50 | 25M | 76.1% | General backbone |
| EfficientNet-B0 | 5M | 77.1% | Efficient/mobile |
| ViT-B/16 | 86M | 81.8% | High accuracy |
| MobileNet-V3 | 5.4M | 75.2% | Mobile/edge |
| CLIP | 400M | Zero-shot | Multi-modal |

**Using Pre-trained Models:**

1. **As Feature Extractor:**
   - Remove classification head
   - Use features for downstream tasks

2. **Fine-tuning:**
   - Adapt to new dataset/classes
   - Train with small learning rate

3. **Zero-shot (CLIP-style):**
   - Use without task-specific training

**Model Sources:**
- `torchvision.models` - PyTorch official
- `timm` - Extensive model zoo (1000+ models)
- `transformers` - Vision Transformers
- TensorFlow Hub, Keras Applications

**Python Example:**
```python
import timm
import torch

# List available models
available = timm.list_models('*efficientnet*')

# Load pretrained model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# Use for inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)  # [B, 10]

# Feature extraction (no classification head)
feature_extractor = timm.create_model('resnet50', pretrained=True, 
                                       num_classes=0)  # Removes head
features = feature_extractor(input_tensor)  # [B, 2048]
```

**Common Pitfalls:**
- Wrong input normalization (must match training preprocessing)
- Wrong input size (some models require specific dimensions)
- Not matching RGB vs BGR channel order

**Interview Tip:** Always check the model's expected preprocessing (mean, std, size). Mismatched normalization is a silent killer.

---

## Object Detection & Classification

### Question 25
**What's the difference between object detection and image classification?**

**Answer:**

Image classification assigns a single label to the entire image (what is in the image?). Object detection identifies multiple objects, providing both class labels AND bounding box locations (what objects are where?). Detection = classification + localization for potentially multiple instances.

**Core Differences:**

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| **Output** | Single class label | Multiple (class, bbox) pairs |
| **Localization** | No | Yes (bounding boxes) |
| **Multiple objects** | No (or multi-label) | Yes |
| **Complexity** | Lower | Higher |
| **Examples** | "This is a cat" | "Cat at (x1,y1,x2,y2), Dog at (...)" |

**Output Comparison:**

```
Classification: {class: "dog", confidence: 0.95}

Detection: [
    {class: "dog", confidence: 0.95, bbox: [100, 50, 200, 150]},
    {class: "cat", confidence: 0.87, bbox: [300, 100, 400, 250]},
    {class: "car", confidence: 0.92, bbox: [50, 200, 150, 280]}
]
```

**Architectures:**

| Task | Architectures |
|------|---------------|
| Classification | ResNet, EfficientNet, ViT |
| Detection | YOLO, Faster R-CNN, SSD, DETR |

**Detection Components:**
1. **Backbone:** Feature extraction (ResNet, CSPDarknet)
2. **Neck:** Feature aggregation (FPN, PANet)
3. **Head:** Predict boxes + classes

**Python Example - Classification vs Detection:**
```python
# Classification
from torchvision.models import resnet50
classifier = resnet50(pretrained=True)
# Output: [batch, 1000] class probabilities

# Detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval()
# Output: list of {boxes: [N, 4], labels: [N], scores: [N]}
```

**Interview Tip:** Detection is harder because it requires predicting a variable number of outputs (boxes) per image, unlike classification's fixed output size.

---

### Question 26
**What algorithms can you use for real-time object detection?**

**Answer:**

Real-time detectors prioritize speed while maintaining accuracy. Single-stage detectors (YOLO, SSD) are fastest as they predict in one pass. YOLO variants dominate real-time detection, with YOLOv8/v10 achieving >100 FPS. For edge devices, use MobileNet-SSD, YOLO-Nano, or apply quantization/pruning.

**Real-Time Detectors:**

| Model | Speed (FPS) | mAP | Architecture |
|-------|-------------|-----|--------------|
| YOLOv8-n | ~200 | 37.3 | Single-stage, anchor-free |
| YOLOv8-s | ~130 | 44.9 | Single-stage, anchor-free |
| YOLOv10 | ~150 | 46.3 | NMS-free |
| SSD-MobileNet | ~60 | 21.0 | Single-stage |
| RT-DETR | ~100 | 53.1 | Transformer-based |

**Single-Stage vs Two-Stage:**

| Aspect | Single-Stage (YOLO, SSD) | Two-Stage (Faster R-CNN) |
|--------|-------------------------|--------------------------|
| Speed | Fast (real-time) | Slower |
| Accuracy | Good | Better |
| Pipeline | One forward pass | Region proposal + classification |
| Use case | Real-time, edge | High accuracy requirements |

**Optimization Techniques:**

1. **Model Architecture:**
   - Use efficient backbones (MobileNet, CSPNet)
   - Depthwise separable convolutions

2. **Model Compression:**
   - Quantization (FP32 → INT8)
   - Pruning (remove unnecessary weights)
   - Knowledge distillation

3. **Inference Optimization:**
   - TensorRT, ONNX Runtime
   - Batch inference
   - Hardware-specific optimizations

**Python Example:**
```python
from ultralytics import YOLO

# Load YOLOv8 nano (fastest)
model = YOLO('yolov8n.pt')

# Inference on image
results = model('image.jpg')

# Real-time video inference
results = model.predict(source=0, stream=True)  # 0 = webcam
for r in results:
    boxes = r.boxes
    # Process boxes in real-time
```

**Interview Tip:** YOLO series is the go-to for real-time detection. Mention FPS targets (30 FPS for video, 60+ for games) and trade-offs with accuracy.

---

### Question 27
**Explain the YOLO approach to object detection.**

**Answer:**

YOLO (You Only Look Once) frames detection as a single regression problem. It divides the image into a grid, and each cell predicts bounding boxes and class probabilities directly in one forward pass. This unified architecture enables real-time detection by eliminating the region proposal step of two-stage detectors.

**Core Concepts:**

**Key Idea:**
- Divide image into S×S grid
- Each cell predicts B bounding boxes + confidence + C class probabilities
- Single forward pass → all predictions simultaneously

**YOLO Output:**
Each cell predicts: $(x, y, w, h, confidence, class_1, class_2, ..., class_C)$
- $(x, y)$: Box center relative to grid cell
- $(w, h)$: Box dimensions relative to image
- $confidence = P(object) \times IoU$

**YOLO Evolution:**

| Version | Key Innovation |
|---------|---------------|
| YOLOv1 | Original grid-based approach |
| YOLOv2 | Anchor boxes, batch norm, multi-scale |
| YOLOv3 | Multi-scale predictions, FPN-like |
| YOLOv4 | CSPDarknet, mosaic augmentation |
| YOLOv5 | PyTorch, easier training |
| YOLOv8 | Anchor-free, decoupled head |
| YOLOv10 | NMS-free, end-to-end |

**Architecture (YOLOv8):**
```
Input → Backbone (CSPDarknet) → Neck (PANet) → Detection Heads
                                                ├─ P3 (small objects)
                                                ├─ P4 (medium objects)
                                                └─ P5 (large objects)
```

**Loss Function:**
$$\mathcal{L} = \lambda_{coord} \mathcal{L}_{box} + \mathcal{L}_{obj} + \lambda_{cls} \mathcal{L}_{cls}$$

**Python Example:**
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m.pt')

# Train on custom data
model.train(data='dataset.yaml', epochs=100, imgsz=640)

# Inference
results = model.predict('image.jpg', conf=0.5)

for r in results:
    boxes = r.boxes.xyxy    # [N, 4] - x1, y1, x2, y2
    scores = r.boxes.conf   # [N] - confidence
    classes = r.boxes.cls   # [N] - class indices
```

**Interview Tip:** YOLO's speed comes from single-pass prediction. The trade-off is slightly lower accuracy on small objects compared to two-stage detectors.

---

### Question 28
**Discuss the R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN).**

**Answer:**

The R-CNN family uses a two-stage approach: first generate region proposals, then classify each region. R-CNN was slow (per-region CNN), Fast R-CNN shared computation via ROI pooling, and Faster R-CNN integrated proposal generation with Region Proposal Network (RPN), achieving near real-time speeds.

**Evolution:**

| Model | Region Proposals | Feature Sharing | Speed |
|-------|-----------------|-----------------|-------|
| R-CNN | Selective Search (external) | None | ~47s/image |
| Fast R-CNN | Selective Search (external) | CNN features shared | ~2s/image |
| Faster R-CNN | RPN (learned) | Full end-to-end | ~0.2s/image |

**R-CNN (2014):**
```
Image → Selective Search (2000 regions) → CNN per region → SVM classifier
```
- Slow: CNN runs 2000 times per image
- Not end-to-end

**Fast R-CNN (2015):**
```
Image → CNN (once) → ROI Pooling → FC layers → Class + BBox
```
- Key: Compute CNN features once, share across regions
- ROI Pooling: Extract fixed-size features from any region

**Faster R-CNN (2016):**
```
Image → CNN → RPN (proposals) → ROI Pooling → Class + BBox
           ↓
    Feature Map (shared)
```
- Region Proposal Network: 3×3 conv + anchors
- Anchors: predefined boxes at multiple scales/ratios
- Fully end-to-end trainable

**Faster R-CNN Components:**
1. **Backbone:** ResNet, VGG (feature extraction)
2. **RPN:** Predicts objectness + refines anchor boxes
3. **ROI Pooling/Align:** Extract fixed features from proposals
4. **Detection Head:** Final classification + bbox regression

**Python Example:**
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference (input: list of tensors [C, H, W])
images = [torch.randn(3, 800, 600)]
predictions = model(images)

# predictions[0]: {'boxes': [N,4], 'labels': [N], 'scores': [N]}
```

**Interview Tip:** Know the bottleneck each version solved: R-CNN's repeated CNN → Fast R-CNN's external proposals → Faster R-CNN's learned RPN.

---

### Question 29
**Compare one-stage vs. two-stage detectors for object detection.**

**Answer:**

Two-stage detectors (Faster R-CNN) first generate region proposals, then classify them—achieving higher accuracy but slower speed. One-stage detectors (YOLO, SSD) directly predict boxes and classes in a single pass—faster but traditionally less accurate. Modern one-stage detectors have closed the accuracy gap.

**Core Comparison:**

| Aspect | One-Stage (YOLO, SSD) | Two-Stage (Faster R-CNN) |
|--------|----------------------|--------------------------|
| **Pipeline** | Single forward pass | Region proposal + classification |
| **Speed** | Fast (real-time) | Slower |
| **Accuracy** | Good (catching up) | Higher |
| **Small Objects** | Challenging | Better (with FPN) |
| **Complexity** | Simpler | More components |
| **Training** | End-to-end | Can be end-to-end |

**Architecture Comparison:**

**One-Stage:**
```
Image → Backbone → Neck → Prediction Heads → NMS → Detections
                   (directly predicts boxes + classes)
```

**Two-Stage:**
```
Image → Backbone → RPN (proposals) → ROI Align → Head → Detections
                   ↓
              Region Proposals (stage 1)
                               ↓
                        Classification (stage 2)
```

**When to Choose:**

| Scenario | Choice |
|----------|--------|
| Real-time applications | One-stage (YOLO) |
| Maximum accuracy needed | Two-stage (Faster R-CNN) |
| Edge/mobile deployment | One-stage (MobileNet-SSD) |
| Small object detection | Two-stage with FPN |
| Instance segmentation | Two-stage (Mask R-CNN) |

**Modern Developments:**
- **DETR:** Transformer-based, no NMS, competitive accuracy
- **YOLOv8/v10:** One-stage matching two-stage accuracy
- **RT-DETR:** Real-time transformer detector

**Interview Tip:** The gap is closing. YOLOv8 achieves accuracy comparable to Faster R-CNN while being 10× faster. Choice depends on latency requirements.

---

### Question 30
**How do image recognition models deal with occlusion?**

**Answer:**

Models handle occlusion through: data augmentation (random erasing, cutout), part-based representations (recognize visible parts independently), attention mechanisms (focus on non-occluded regions), and multi-scale features (different levels capture different object portions). Robust training on occluded examples is key.

**Core Strategies:**

**1. Data Augmentation:**
| Technique | Description |
|-----------|-------------|
| Random Erasing | Randomly mask rectangular regions |
| Cutout | Zero out random square patches |
| GridMask | Structured grid-based masking |
| Copy-paste | Paste objects creating occlusion |

**2. Architectural Solutions:**
- **Attention Mechanisms:** Focus on visible, discriminative parts
- **Part-based Models:** Detect object parts independently
- **Multi-scale Features:** FPN captures both local and global context
- **Deformable Convolutions:** Adapt receptive field to visible regions

**3. Training Strategies:**
- Include heavily occluded samples in training data
- Hard negative mining for occluded cases
- Self-supervised pre-training on masked images (MAE)

**4. Detection-Specific:**
- Soft-NMS handles overlapping/occluded objects better
- Repulsion loss pushes apart nearby detections
- CrowdDet for crowded scenes

**Python Example - Occlusion Augmentation:**
```python
import albumentations as A

# Augmentations for occlusion robustness
transform = A.Compose([
    A.CoarseDropout(
        max_holes=8, 
        max_height=32, 
        max_width=32,
        fill_value=0,
        p=0.5
    ),
    A.RandomGridShuffle(grid=(3, 3), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply augmentation
augmented = transform(image=image)
```

**Interview Tip:** Mention that CNNs are inherently robust to moderate occlusion due to hierarchical feature learning. Training with Cutout/Random Erasing explicitly improves this.

---

## Advanced Techniques

### Question 31
**What is pose estimation, and what are its applications?**

**Answer:**

Pose estimation detects human body keypoints (joints like wrists, elbows, shoulders) from images or video to understand body posture and movement. It outputs 2D or 3D coordinates of keypoints and their connections (skeleton). Applications include fitness tracking, gesture recognition, sports analysis, and AR.

**Core Concepts:**

**Types of Pose Estimation:**

| Type | Description | Output |
|------|-------------|--------|
| **Single-person** | One person in image | N keypoints (x, y) |
| **Multi-person** | Multiple people | Keypoints per person |
| **2D Pose** | Keypoints in image plane | (x, y) coordinates |
| **3D Pose** | Keypoints in 3D space | (x, y, z) coordinates |

**Approaches:**

**Top-Down:**
1. Detect all people (bounding boxes)
2. Apply single-person pose estimator to each box
- Pros: Higher accuracy
- Cons: Speed depends on number of people

**Bottom-Up:**
1. Detect all keypoints in image
2. Group keypoints into person instances
- Pros: Speed independent of people count
- Cons: Grouping is challenging

**Common Models:**
- **OpenPose:** Bottom-up, Part Affinity Fields
- **HRNet:** High-Resolution Net, maintains resolution
- **MediaPipe Pose:** Real-time, mobile-friendly
- **ViTPose:** Transformer-based, state-of-the-art

**Applications:**
- Fitness apps (form correction)
- Sports analytics (motion analysis)
- Gaming (gesture control)
- Healthcare (gait analysis, rehabilitation)
- AR/VR (avatar animation)

**Python Example:**
```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Process image
image = cv2.imread('person.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Draw keypoints
if results.pose_landmarks:
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Access individual keypoints
    for landmark in results.pose_landmarks.landmark:
        x, y = landmark.x, landmark.y  # Normalized coordinates
```

**Interview Tip:** MediaPipe is fastest for real-time. HRNet/ViTPose for accuracy. Know the difference between top-down and bottom-up approaches.

---

### Question 32
**How does optical flow contribute to understanding motion in videos?**

**Answer:**

Optical flow estimates per-pixel motion vectors between consecutive frames, showing how each pixel moves from frame t to t+1. It captures apparent motion patterns, enabling motion segmentation, video stabilization, action recognition, and tracking. Dense flow provides motion for every pixel; sparse flow tracks selected points.

**Core Concepts:**

**Optical Flow Output:**
- For each pixel $(x, y)$: motion vector $(u, v)$
- $u$ = horizontal displacement, $v$ = vertical displacement
- Visualized as color-coded flow field (hue = direction, saturation = magnitude)

**Brightness Constancy Assumption:**
$$I(x, y, t) = I(x + u, y + v, t + 1)$$

**Optical Flow Constraint Equation:**
$$I_x u + I_y v + I_t = 0$$

Where $I_x, I_y$ = spatial gradients, $I_t$ = temporal gradient

**Types:**

| Type | Description | Methods |
|------|-------------|---------|
| **Sparse** | Track specific points | Lucas-Kanade |
| **Dense** | Flow for every pixel | Farneback, RAFT |

**Classical Methods:**
- **Lucas-Kanade:** Sparse, assumes local constancy
- **Horn-Schunck:** Dense, global smoothness constraint
- **Farneback:** Dense, polynomial expansion

**Deep Learning Methods:**
- **FlowNet/FlowNet2:** CNN-based, fast
- **RAFT:** Recurrent All-pairs Field Transforms, state-of-the-art
- **PWC-Net:** Pyramid, warping, cost volume

**Applications:**
- Video object tracking
- Action recognition
- Video stabilization
- Autonomous vehicle motion estimation
- Video interpolation

**Python Example:**
```python
import cv2
import numpy as np

# Read two consecutive frames
frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# Dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    frame1, frame2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# flow shape: (H, W, 2) -> [u, v] per pixel
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

**Interview Tip:** Dense optical flow is computationally expensive. For real-time applications, use sparse flow (Lucas-Kanade) or efficient deep models (RAFT-Small).

---

### Question 33
**Explain how CNNs can be used for human activity recognition in video data.**

**Answer:**

CNNs extract spatial features from video frames, combined with temporal modeling for activity recognition. Approaches include: 2D CNNs on individual frames + temporal aggregation (LSTM/pooling), 3D CNNs (spatiotemporal convolutions), two-stream networks (RGB + optical flow), and video transformers for long-range dependencies.

**Core Approaches:**

**1. 2D CNN + Temporal Aggregation:**
```
Frame 1 → CNN → Features ─┐
Frame 2 → CNN → Features ─┼→ Temporal Model (LSTM/Pool) → Class
Frame 3 → CNN → Features ─┘
```
- Extract per-frame features with pretrained CNN
- Aggregate with LSTM, temporal pooling, or attention

**2. 3D CNN (Spatiotemporal Convolutions):**
```
Video Clip [T, H, W, C] → 3D Conv layers → 3D Pool → FC → Class
```
- C3D, I3D: 3D kernels capture both space and time
- Computationally expensive

**3. Two-Stream Networks:**
```
RGB frames → Spatial CNN ─┐
                          ├→ Fusion → Class
Optical flow → Motion CNN ┘
```
- Separate streams for appearance and motion
- Complementary information

**4. Video Transformers:**
- ViViT, Video Swin: Attention across space and time
- Handle long-range temporal dependencies

**Model Comparison:**

| Model | Input | Temporal | Pros |
|-------|-------|----------|------|
| 2D CNN + LSTM | Frames | Sequential | Simple, pretrained weights |
| C3D/I3D | Clips | 3D conv | End-to-end, captures motion |
| Two-Stream | RGB + Flow | Parallel | Strong motion features |
| ViViT | Patches | Attention | Long-range, SOTA |

**Python Example:**
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ActivityRecognizer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 2D CNN backbone (remove FC)
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        # Temporal modeling
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, video):
        # video: [B, T, C, H, W]
        B, T = video.shape[:2]
        
        # Extract frame features
        frames = video.view(B * T, *video.shape[2:])
        features = self.cnn(frames)  # [B*T, 2048]
        features = features.view(B, T, -1)  # [B, T, 2048]
        
        # Temporal aggregation
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])  # Last timestep
        return out
```

**Interview Tip:** I3D inflated from ImageNet weights is a strong baseline. For efficiency, use 2D CNN with temporal pooling. For SOTA, use Video Swin or ViViT.

---

### Question 34
**What are Generative Adversarial Networks (GANs) and their role in computer vision?**

**Answer:**

GANs consist of two neural networks—generator (creates fake images) and discriminator (distinguishes real from fake)—trained adversarially. The generator learns to produce realistic images to fool the discriminator. In CV, GANs enable image synthesis, super-resolution, style transfer, data augmentation, and image-to-image translation.

**Core Concepts:**

**Architecture:**
```
Random Noise (z) → Generator → Fake Image
                                    ↓
                              Discriminator ← Real Image
                                    ↓
                              Real or Fake?
```

**Training Dynamics:**
- **Generator:** Minimize $\log(1 - D(G(z)))$ → fool discriminator
- **Discriminator:** Maximize $\log(D(x)) + \log(1 - D(G(z)))$ → distinguish real/fake

**Minimax Objective:**
$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

**Computer Vision Applications:**

| Application | GAN Type |
|-------------|----------|
| Image synthesis | StyleGAN, BigGAN |
| Super-resolution | SRGAN, ESRGAN |
| Image-to-image | Pix2Pix, CycleGAN |
| Data augmentation | Conditional GANs |
| Inpainting | DeepFill, LaMa |
| Face aging | Age-cGAN |

**Key Variants:**
- **DCGAN:** Convolutional architecture, stable training
- **StyleGAN:** High-quality faces, style control
- **CycleGAN:** Unpaired image translation
- **Conditional GAN:** Control output with labels

**Python Example:**
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),  # 28x28 image
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
```

**Interview Tip:** GANs are being replaced by diffusion models for image generation (better quality, stable training), but remain important for image-to-image tasks.

---

### Question 35
**Discuss few-shot learning and its challenges in computer vision.**

**Answer:**

Few-shot learning trains models to recognize new classes from very few examples (1-5 images per class). It addresses the data scarcity problem by learning to learn—using meta-learning, metric learning, or transfer learning. Challenges include overfitting to limited samples, domain shift, and evaluating generalization properly.

**Core Concepts:**

**Problem Setting:**
- **N-way K-shot:** Classify among N classes, K examples each
- **Support set:** Few labeled examples for new classes
- **Query set:** Images to classify

**Approaches:**

| Method | Idea |
|--------|------|
| **Metric Learning** | Learn embedding space, classify by similarity |
| **Meta-Learning** | Learn to adapt quickly from few examples |
| **Transfer Learning** | Pretrain on large data, fine-tune on few samples |
| **Data Augmentation** | Generate synthetic examples |

**Metric Learning (Siamese, Prototypical):**
- Learn embeddings where same-class samples are close
- Classify query by nearest neighbor in embedding space

**Meta-Learning (MAML, Reptile):**
- Train on many few-shot tasks
- Learn initialization that adapts quickly

**Challenges:**

| Challenge | Description |
|-----------|-------------|
| Overfitting | Few samples → easy to memorize |
| Domain shift | Support set may differ from deployment |
| Intra-class variation | Few samples don't cover class diversity |
| Evaluation | Standard benchmarks may not reflect real use |

**Python Example - Prototypical Networks:**
```python
import torch
import torch.nn.functional as F

def prototypical_loss(embeddings, labels, n_way, k_shot, n_query):
    """
    embeddings: [n_way * (k_shot + n_query), embed_dim]
    """
    # Reshape
    support = embeddings[:n_way * k_shot]  # Support set
    query = embeddings[n_way * k_shot:]     # Query set
    
    # Compute prototypes (class mean)
    support = support.view(n_way, k_shot, -1)
    prototypes = support.mean(dim=1)  # [n_way, embed_dim]
    
    # Compute distances from query to prototypes
    dists = torch.cdist(query, prototypes)  # [n_query, n_way]
    
    # Classification: nearest prototype
    log_probs = F.log_softmax(-dists, dim=1)
    return log_probs  # Use cross-entropy loss
```

**Interview Tip:** Prototypical Networks are a simple, strong baseline. For real applications, pretrained models (CLIP, DINO) with linear probing often outperform specialized few-shot methods.

---

### Question 36
**Explain zero-shot learning in image recognition.**

**Answer:**

Zero-shot learning recognizes classes never seen during training by leveraging auxiliary information (text descriptions, attributes, or embeddings). Models learn a shared embedding space where images and class descriptions can be compared. CLIP exemplifies this—matching images to text descriptions enables classification without class-specific training.

**Core Concepts:**

**Problem Setting:**
- **Seen classes:** Used for training
- **Unseen classes:** Never seen, must recognize at test time
- **Auxiliary info:** Text, attributes, or semantic embeddings describe classes

**How It Works:**
```
Training: Learn to map images → shared embedding space
          Learn to map text/attributes → same space

Inference: Image → embedding
           Compare with all class embeddings (including unseen)
           Predict nearest class
```

**Approaches:**

| Method | Auxiliary Information |
|--------|----------------------|
| Attribute-based | Binary attributes per class |
| Text-based (CLIP) | Natural language descriptions |
| Semantic embeddings | Word2Vec, class hierarchies |

**CLIP Architecture:**
```
Image → Image Encoder → Image Embedding
Text  → Text Encoder  → Text Embedding
                            ↓
              Contrastive matching (cosine similarity)
```

**Python Example - CLIP Zero-Shot:**
```python
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load image
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Define classes (including unseen)
text = clip.tokenize(["a dog", "a cat", "a bird", "a car"]).to(device)

# Zero-shot classification
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Compute similarity
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    predicted_class = similarity.argmax().item()
```

**Challenges:**
- Unseen classes must be describable in auxiliary space
- Domain gap between training and test classes
- Bias toward seen classes in generalized zero-shot learning

**Interview Tip:** CLIP represents a paradigm shift—massive web-scale training enables strong zero-shot transfer. Combine with prompting for best results.

---

### Question 37
**How can reinforcement learning be applied to problems in computer vision?**

**Answer:**

RL in CV uses visual input as state and learns policies for sequential decision-making tasks. Applications include active perception (where to look next), visual navigation, robotic manipulation, attention mechanisms (where to attend), and adversarial training. The agent learns to take actions that maximize cumulative visual task reward.

**Core Concepts:**

**RL Framework Applied to Vision:**
```
State: Visual input (image/video)
Action: Where to look, how to move, what to attend
Reward: Task-specific (navigation success, recognition accuracy)
Policy: CNN/ViT that maps images to actions
```

**Applications:**

| Application | State | Action | Reward |
|-------------|-------|--------|--------|
| Visual Navigation | Camera view | Move/turn | Reach goal |
| Active Object Recognition | Current view | Move camera | Correct classification |
| Hard Attention | Image | Select region | Recognition accuracy |
| Image Editing | Current image | Apply edit | User preference |
| Video Summarization | Video frame | Include/skip | Summary quality |

**Key Architectures:**

**1. Visual Navigation (PointGoal):**
```
RGB + Depth → CNN → Policy (move forward, turn left/right, stop)
```

**2. Visual Attention (Hard):**
- Agent decides where to look (glimpse location)
- Recurrent Attention Model (RAM)

**3. Neural Architecture Search:**
- RL agent designs CNN architecture
- Reward = validation accuracy

**Challenges:**
- Sample inefficiency (vision models are expensive)
- Sparse rewards in visual tasks
- Sim-to-real gap for robotics

**Python Example - Simple Visual RL:**
```python
import gym
import torch
import torch.nn as nn

class VisualPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # CNN encoder for visual input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 9 * 9, 256)
        self.policy = nn.Linear(256, action_dim)
    
    def forward(self, state):
        # state: [B, C, H, W] image
        x = self.cnn(state)
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.policy(x), dim=-1)
        return action_probs
```

**Interview Tip:** RL in CV is computationally expensive but powerful for active perception and embodied AI. Simulation (Habitat, AI2-THOR) is crucial for training.

---

### Question 38
**What are Siamese networks and where are they applicable?**

**Answer:**

Siamese networks use two identical subnetworks (shared weights) to process two inputs and compare their embeddings. They learn similarity metrics rather than class labels, making them ideal for verification tasks (same/different), one-shot learning, and tracking. The network outputs a similarity score between input pairs.

**Core Concepts:**

**Architecture:**
```
Input A → Encoder → Embedding A ─┐
                                  ├→ Distance/Similarity → Same/Different
Input B → Encoder → Embedding B ─┘
    (shared weights)
```

**Key Properties:**
- **Weight Sharing:** Same network for both inputs
- **Metric Learning:** Learns to compare, not classify
- **Generalization:** Can handle unseen classes

**Loss Functions:**

**Contrastive Loss:**
$$L = (1-y) \frac{1}{2} D^2 + y \frac{1}{2} \max(0, m - D)^2$$
Where $y=1$ for different, $D$ = distance, $m$ = margin

**Triplet Loss:**
$$L = \max(0, D(a, p) - D(a, n) + m)$$
Push positive pairs closer, negatives apart

**Applications:**

| Application | Input Pair | Output |
|-------------|-----------|--------|
| Face verification | Two face images | Same person? |
| Signature verification | Two signatures | Match? |
| One-shot recognition | Query, support image | Same class? |
| Visual tracking | Current frame, target | Similarity map |
| Duplicate detection | Two images | Duplicate? |

**Python Example:**
```python
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256)
        )
    
    def forward_one(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = torch.pairwise_distance(emb1, emb2)
    loss = (1 - label) * dist.pow(2) + \
           label * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()
```

**Interview Tip:** Siamese networks are great when you need to compare inputs rather than classify them, especially with limited training examples per class.

---

## Model Optimization & Evaluation

### Question 39
**How do you handle overfitting in a computer vision model?**

**Answer:**

Handle overfitting through: data augmentation (most effective in CV), regularization (dropout, weight decay), early stopping, transfer learning from pretrained models, reducing model complexity, and using more training data. For CV specifically, augmentation is crucial since images have natural transformations that preserve semantics.

**Core Strategies:**

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Data Augmentation** | Create varied training samples | Always, first choice |
| **Dropout** | Random neuron deactivation | FC layers |
| **Weight Decay** | L2 regularization | Always |
| **Early Stopping** | Stop when val loss increases | Always |
| **Transfer Learning** | Use pretrained weights | Limited data |
| **Reduce Model Size** | Fewer parameters | Clear overfitting |
| **Batch Normalization** | Regularization effect | Standard practice |

**Data Augmentation (Most Important for CV):**
- Geometric: flip, rotate, crop, scale
- Photometric: brightness, contrast, color
- Advanced: Mixup, CutMix, AutoAugment

**Regularization Techniques:**

$$L_{total} = L_{task} + \lambda ||w||_2^2 \quad \text{(Weight Decay)}$$

**Signs of Overfitting:**
- Training accuracy >> Validation accuracy
- Training loss decreasing, validation loss increasing
- Gap widens with more epochs

**Python Example:**
```python
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Data Augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# 2. Model with Dropout
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout before FC
            nn.Linear(128 * 54 * 54, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

# 3. Weight Decay in Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. Early Stopping
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    val_loss = validate(model)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Interview Tip:** In CV, data augmentation is often more effective than regularization. Combine augmentation with transfer learning for small datasets.

---

### Question 40
**What are some common metrics to evaluate a computer vision system's performance?**

**Answer:**

Metrics depend on the task: Classification uses accuracy, precision, recall, F1-score, and AUC-ROC. Detection uses mAP (mean Average Precision) and IoU. Segmentation uses IoU/Dice coefficient and pixel accuracy. Generation uses FID and Inception Score. Choose metrics aligned with business requirements.

**Metrics by Task:**

**Classification:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN) / Total | Balanced classes |
| Precision | TP / (TP+FP) | Minimize false positives |
| Recall | TP / (TP+FN) | Minimize false negatives |
| F1-Score | 2·P·R / (P+R) | Balance P and R |
| AUC-ROC | Area under ROC | Binary classification |

**Object Detection:**

| Metric | Description |
|--------|-------------|
| IoU | Intersection over Union of boxes |
| mAP | Mean AP across classes |
| mAP@0.5 | AP at IoU threshold 0.5 |
| mAP@0.5:0.95 | Average across IoU thresholds |

**Segmentation:**

| Metric | Formula |
|--------|---------|
| Pixel Accuracy | Correct pixels / Total pixels |
| IoU (Jaccard) | Intersection / Union |
| Dice Coefficient | 2·Intersection / (Pred + GT) |
| Boundary F1 | F1 score on boundary pixels |

**Image Generation:**

| Metric | Description |
|--------|-------------|
| FID | Fréchet Inception Distance (lower = better) |
| IS | Inception Score (higher = better) |
| LPIPS | Perceptual similarity |

**Python Example:**
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Classification metrics
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# IoU for segmentation
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

# Dice coefficient
def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum())
```

**Interview Tip:** Always clarify what metric matters for the business. High recall for medical diagnosis (don't miss disease), high precision for spam (don't block good emails).

---

### Question 41
**Discuss the importance of cross-validation for a vision model.**

**Answer:**

Cross-validation provides robust performance estimates by training and evaluating on multiple data splits, reducing variance from a single train-test split. For vision models with limited data, it maximizes use of available samples. K-fold CV is standard; stratified splits maintain class balance. However, it's expensive for large models.

**Core Concepts:**

**Why Cross-Validation:**
- Single train-test split can be lucky/unlucky
- Provides confidence intervals for performance
- Uses all data for both training and validation
- Detects overfitting to specific data splits

**Types of Cross-Validation:**

| Type | Description | Use Case |
|------|-------------|----------|
| K-Fold | Split into K parts, rotate test set | Standard |
| Stratified K-Fold | Maintain class proportions | Imbalanced classes |
| Leave-One-Out | K = N (expensive) | Very small datasets |
| Group K-Fold | Keep groups together | Patient data, video frames |

**K-Fold Process:**
```
Data: [Fold1, Fold2, Fold3, Fold4, Fold5]

Iteration 1: Train on [2,3,4,5], Test on [1]
Iteration 2: Train on [1,3,4,5], Test on [2]
...
Final: Average metrics across all folds
```

**CV for Vision Models:**

**When to Use:**
- Small datasets (< 10,000 images)
- Model selection and hyperparameter tuning
- Comparing architectures fairly

**When to Skip:**
- Large datasets (ImageNet-scale)
- Expensive models (training time prohibitive)
- Use single validation set instead

**Python Example:**
```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Stratified K-Fold (maintains class balance)
X = np.array(image_paths)
y = np.array(labels)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model on this fold
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    val_acc = model.evaluate(X_val, y_val)
    fold_results.append(val_acc)
    print(f"Fold {fold+1}: {val_acc:.4f}")

print(f"Mean: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
```

**Interview Tip:** For deep learning, full K-fold is often impractical. Use single validation split for training, K-fold for final model comparison or small datasets.

---

### Question 42
**Explain how Intersection over Union (IoU) works for object detection models.**

**Answer:**

IoU measures overlap between predicted and ground-truth bounding boxes by computing intersection area divided by union area. It ranges from 0 (no overlap) to 1 (perfect match). IoU is used to determine true/false positives at various thresholds (typically 0.5) and to calculate mAP. It's also used as a loss function variant.

**Mathematical Formulation:**

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Intersection Area}}{\text{Union Area}}$$

**Computing IoU:**
```
Box A: (x1_a, y1_a, x2_a, y2_a)
Box B: (x1_b, y1_b, x2_b, y2_b)

Intersection:
  x1_i = max(x1_a, x1_b)
  y1_i = max(y1_a, y1_b)
  x2_i = min(x2_a, x2_b)
  y2_i = min(y2_a, y2_b)
  area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

Union:
  area_a = (x2_a - x1_a) * (y2_a - y1_a)
  area_b = (x2_b - x1_b) * (y2_b - y1_b)
  area_u = area_a + area_b - area_i

IoU = area_i / area_u
```

**IoU Thresholds:**

| Threshold | Meaning |
|-----------|---------|
| IoU > 0.5 | PASCAL VOC standard |
| IoU > 0.75 | Stricter matching |
| mAP@0.5:0.95 | Average across 0.5, 0.55, ..., 0.95 (COCO) |

**IoU Loss Variants:**

| Loss | Improvement |
|------|-------------|
| IoU Loss | $L = 1 - IoU$ |
| GIoU | Handles non-overlapping boxes |
| DIoU | Considers center distance |
| CIoU | Adds aspect ratio penalty |

**Python Example:**
```python
import numpy as np

def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Example
pred_box = [100, 100, 200, 200]
gt_box = [120, 120, 220, 220]
print(f"IoU: {compute_iou(pred_box, gt_box):.3f}")
```

**Interview Tip:** IoU = 0.5 is "good enough" match; IoU > 0.75 is "precise" match. COCO mAP averages across thresholds for comprehensive evaluation.

---

## Real-World Applications

### Question 43
**How would you design a vision system for automatic license plate recognition?**

**Answer:**

ALPR pipeline: (1) Detect vehicle → (2) Localize license plate region → (3) Segment characters → (4) Recognize characters via OCR. Use YOLO/SSD for detection, perspective correction for tilted plates, and CNN+CTC or transformer OCR. Handle variable plate formats, lighting conditions, and motion blur.

**System Design:**

```
Input Image → Vehicle Detection → Plate Localization → Preprocessing 
                                                            ↓
                              Output ← Post-processing ← OCR Recognition
```

**Pipeline Components:**

| Stage | Method | Output |
|-------|--------|--------|
| Vehicle Detection | YOLO, Faster R-CNN | Vehicle bounding boxes |
| Plate Localization | YOLO, edge detection | Plate region |
| Preprocessing | Deskew, binarization | Clean plate image |
| OCR | CRNN+CTC, TrOCR | Character sequence |
| Post-processing | Regex, format validation | Validated plate number |

**Key Challenges:**

| Challenge | Solution |
|-----------|----------|
| Variable plate sizes | Multi-scale detection |
| Perspective distortion | Perspective transform, spatial transformer |
| Low light / night | IR cameras, image enhancement |
| Motion blur | Deblurring, high frame rate |
| Different formats | Region-specific training data |
| Occlusion | Multiple frames, tracking |

**Design Considerations:**
- **Speed:** Real-time processing for traffic systems
- **Accuracy:** >95% character accuracy typical requirement
- **Edge deployment:** Optimize for embedded hardware

**Python Example - Simplified Pipeline:**
```python
import cv2
from ultralytics import YOLO
import easyocr

class LicensePlateRecognizer:
    def __init__(self):
        self.plate_detector = YOLO('plate_detector.pt')
        self.ocr = easyocr.Reader(['en'])
    
    def preprocess_plate(self, plate_img):
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def recognize(self, image):
        # Detect plate
        results = self.plate_detector(image)
        plates = []
        
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]
            
            # Preprocess
            processed = self.preprocess_plate(plate_img)
            
            # OCR
            text = self.ocr.readtext(processed, detail=0)
            plates.append(''.join(text).upper())
        
        return plates
```

**Interview Tip:** Discuss real-world constraints: camera placement, lighting, speed limits, privacy regulations. Mention tracking for temporal voting across frames.

---

### Question 44
**Outline the computer vision technologies involved in autonomous vehicle navigation.**

**Answer:**

Autonomous vehicles use multi-sensor fusion: cameras for object detection/lane recognition, LiDAR for 3D mapping, radar for velocity estimation. CV tasks include object detection (vehicles, pedestrians), lane detection, traffic sign recognition, depth estimation, SLAM, and motion prediction. Redundancy across sensors ensures safety.

**Sensor Suite:**

| Sensor | Purpose | Strengths |
|--------|---------|-----------|
| Cameras | Object detection, lane lines, signs | Rich semantic info, low cost |
| LiDAR | 3D point cloud, distance | Accurate depth, weather robust |
| Radar | Velocity, distance | Works in fog/rain |
| Ultrasonic | Short-range obstacles | Parking assistance |
| GPS/IMU | Localization | Global positioning |

**Computer Vision Tasks:**

| Task | Method | Output |
|------|--------|--------|
| Object Detection | YOLO, Faster R-CNN | Bounding boxes + classes |
| Semantic Segmentation | DeepLab, SegFormer | Pixel-wise road/lane/object |
| Lane Detection | LaneNet, SCNN | Lane polynomials |
| Depth Estimation | Stereo, monocular DL | Depth map |
| 3D Object Detection | PointPillars, BEVFusion | 3D bounding boxes |
| Traffic Sign Recognition | CNN classifier | Sign type |
| Motion Prediction | Trajectron++, VectorNet | Future trajectories |
| SLAM | ORB-SLAM, visual odometry | Localization + mapping |

**System Architecture:**
```
Sensors → Perception → Sensor Fusion → Planning → Control → Actuators
           ↓
      Detection, Segmentation, Tracking
           ↓
      3D Scene Understanding
```

**Key Challenges:**

| Challenge | Solution |
|-----------|----------|
| Adverse weather | Multi-sensor fusion, radar |
| Edge cases | Extensive simulation, rare event training |
| Real-time processing | Efficient models, TensorRT |
| Safety | Redundancy, uncertainty estimation |

**Python Example - Bird's Eye View Perception:**
```python
import torch
import torch.nn as nn

class BEVPerception(nn.Module):
    """Simplified Bird's Eye View perception"""
    def __init__(self):
        super().__init__()
        # Image backbone (multi-camera)
        self.img_backbone = ResNet50()
        
        # Transform to BEV
        self.img_to_bev = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            # View transformation (simplified)
        )
        
        # LiDAR backbone
        self.lidar_backbone = PointPillars()
        
        # Fusion
        self.fusion = nn.Conv2d(512, 256, 3, padding=1)
        
        # Detection head
        self.detection_head = nn.Conv2d(256, 10, 1)  # Classes + boxes
    
    def forward(self, images, lidar):
        img_feat = self.img_backbone(images)
        img_bev = self.img_to_bev(img_feat)
        
        lidar_bev = self.lidar_backbone(lidar)
        
        fused = self.fusion(torch.cat([img_bev, lidar_bev], dim=1))
        return self.detection_head(fused)
```

**Interview Tip:** Emphasize sensor fusion and redundancy for safety. Mention specific datasets (nuScenes, KITTI, Waymo) and the importance of handling edge cases.

---

### Question 45
**Propose an approach for medical image analysis using computer vision.**

**Answer:**

Medical image analysis uses specialized CV models for diagnosis assistance. Approach: (1) Domain-specific preprocessing (DICOM handling, windowing), (2) Pretrained models fine-tuned on medical data, (3) U-Net variants for segmentation, (4) Explainability via Grad-CAM. Address class imbalance, require regulatory compliance, and provide uncertainty estimates.

**System Design:**

```
DICOM Images → Preprocessing → Augmentation → Model → Post-processing → Clinical Output
                    ↓                             ↓
              Normalization                  Explainability
              Windowing                      Uncertainty
```

**Medical CV Tasks:**

| Task | Application | Methods |
|------|-------------|---------|
| Classification | Disease diagnosis | CNN, ViT with attention |
| Segmentation | Tumor delineation | U-Net, nnU-Net |
| Detection | Lesion finding | Faster R-CNN, YOLO |
| Registration | Image alignment | VoxelMorph, ANTs |
| Reconstruction | MRI acceleration | Deep unfolding, diffusion |

**Key Considerations:**

| Aspect | Requirement |
|--------|-------------|
| Data privacy | HIPAA compliance, federated learning |
| Class imbalance | Focal loss, oversampling |
| Explainability | Grad-CAM, attention maps |
| Uncertainty | Dropout inference, ensembles |
| Validation | Multi-site testing, clinical trials |
| Regulatory | FDA approval, CE marking |

**Architecture Choice:**
- **nnU-Net:** Self-configuring, SOTA for segmentation
- **DenseNet:** Good for classification, fewer parameters
- **3D Models:** For volumetric data (CT, MRI)

**Python Example - Medical Image Classification:**
```python
import torch
import torch.nn as nn
from torchvision import models
import pydicom

class MedicalImageClassifier:
    def __init__(self, num_classes):
        # Pretrained backbone
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, num_classes)
        
    def preprocess_dicom(self, dicom_path):
        # Load DICOM
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(float)
        
        # Windowing (for CT)
        if hasattr(dcm, 'WindowCenter'):
            center = dcm.WindowCenter
            width = dcm.WindowWidth
            img = np.clip(img, center - width/2, center + width/2)
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to 3-channel
        img = np.stack([img, img, img], axis=0)
        return torch.tensor(img, dtype=torch.float32)
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """Monte Carlo Dropout for uncertainty"""
        self.model.train()  # Enable dropout
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = torch.softmax(self.model(x), dim=1)
                predictions.append(pred)
        
        mean_pred = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).std(dim=0)
        return mean_pred, uncertainty
```

**Interview Tip:** Medical AI requires explainability (clinicians need to trust), uncertainty (know when to refer to human), and rigorous validation (multiple hospitals, diverse populations).

---

### Question 46
**Discuss the use of computer vision in retail for product recognition and tracking.**

**Answer:**

Retail CV enables automated checkout, inventory management, and analytics. Systems use object detection for product recognition from shelf images, barcode/label reading, customer tracking for behavior analysis, and planogram compliance checking. Challenges include high SKU counts, similar products, and varying lighting conditions.

**Retail CV Applications:**

| Application | CV Task | Business Value |
|-------------|---------|----------------|
| Smart checkout | Product detection | Reduced wait times |
| Inventory counting | Object detection + counting | Stock accuracy |
| Planogram compliance | Object detection + matching | Shelf optimization |
| Customer analytics | Person tracking, pose | Store layout improvement |
| Loss prevention | Anomaly detection | Shrinkage reduction |
| Price tag validation | OCR | Pricing accuracy |

**System Architecture:**
```
Cameras → Edge Processing → Detection/Recognition → Cloud Analytics → Actions
              ↓
        Real-time inference
        Low latency required
```

**Technical Challenges:**

| Challenge | Solution |
|-----------|----------|
| High SKU count (10,000+) | Hierarchical classification, retrieval |
| Similar products | Fine-grained recognition |
| Varying lighting | Robust augmentation, domain adaptation |
| Occlusion on shelves | Multi-view cameras |
| Real-time requirements | Edge AI, quantized models |
| Privacy concerns | On-device processing, no face storage |

**Key Technologies:**
- **Product Detection:** YOLO, EfficientDet
- **Product Identification:** Image retrieval, embedding matching
- **Barcode/OCR:** Specialized readers, TrOCR
- **People Tracking:** DeepSORT, ByteTrack

**Python Example - Product Recognition System:**
```python
from ultralytics import YOLO
import faiss
import numpy as np

class ProductRecognitionSystem:
    def __init__(self):
        self.detector = YOLO('product_detector.pt')
        self.embedder = load_embedding_model()
        
        # Product database (embeddings + metadata)
        self.product_db = self.load_product_database()
        self.index = self.build_faiss_index()
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        embeddings = np.array(self.product_db['embeddings'])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def recognize_products(self, shelf_image):
        # Detect products
        detections = self.detector(shelf_image)
        
        products = []
        for box in detections[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            product_crop = shelf_image[y1:y2, x1:x2]
            
            # Get embedding
            embedding = self.embedder(product_crop)
            
            # Search in database
            D, I = self.index.search(embedding.reshape(1, -1), k=1)
            
            if D[0][0] < threshold:
                product_id = self.product_db['ids'][I[0][0]]
                products.append({
                    'bbox': [x1, y1, x2, y2],
                    'product_id': product_id,
                    'confidence': 1 - D[0][0]
                })
        
        return products
```

**Interview Tip:** Discuss the scale challenge (thousands of SKUs) and the need for image retrieval rather than pure classification. Mention edge deployment for privacy and latency.

---

### Question 47
**How might AR applications benefit from advances in computer vision?**

**Answer:**

AR relies heavily on CV for understanding the real world: SLAM for device localization, plane detection for surface anchoring, object recognition for context, hand/face tracking for interaction, and depth estimation for occlusion. Advances in real-time CV enable more immersive, stable, and interactive AR experiences.

**CV Technologies Enabling AR:**

| CV Technology | AR Application |
|---------------|----------------|
| SLAM | Device tracking, world mapping |
| Plane Detection | Placing virtual objects on surfaces |
| Depth Estimation | Realistic occlusion handling |
| Object Detection | Object-anchored AR content |
| Hand Tracking | Gesture-based interaction |
| Face Tracking | AR filters, face effects |
| Semantic Segmentation | Environment understanding |
| Pose Estimation | Body-based AR effects |

**How CV Advances Benefit AR:**

| Advance | AR Benefit |
|---------|------------|
| Real-time processing | Smoother, lower latency |
| Better depth estimation | More realistic occlusion |
| Improved SLAM | More stable tracking |
| Neural rendering (NeRF) | Photorealistic virtual objects |
| Efficient models | On-device AR, longer battery |
| 3D reconstruction | Persistent AR worlds |

**AR Pipeline:**
```
Camera → Tracking/SLAM → Scene Understanding → Rendering → Display
             ↓                   ↓
        Camera pose        Planes, objects, depth
             ↓                   ↓
        Place content      Handle occlusion
```

**Key AR Platforms:**
- **ARKit (Apple):** LiDAR + CV for depth, plane detection
- **ARCore (Google):** Motion tracking, environmental understanding
- **Meta Quest:** Hand tracking, spatial understanding
- **HoloLens:** Full 3D mapping, spatial anchors

**Python Example - Simple AR Marker Detection:**
```python
import cv2
import numpy as np

class SimpleARSystem:
    def __init__(self):
        # ArUco marker dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Camera calibration (intrinsics)
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros(5)
    
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        return corners, ids
    
    def get_marker_pose(self, corners, marker_size=0.05):
        """Get rotation and translation of marker"""
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.camera_matrix, self.dist_coeffs
        )
        return rvecs, tvecs
    
    def render_cube(self, frame, rvec, tvec, size=0.05):
        """Render virtual cube on marker"""
        # Define cube vertices
        cube_points = np.float32([
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, -size], [size, 0, -size], [size, size, -size], [0, size, -size]
        ])
        
        # Project 3D points to 2D
        img_points, _ = cv2.projectPoints(
            cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        # Draw cube edges
        img_points = img_points.reshape(-1, 2).astype(int)
        # ... draw lines between points
        return frame
```

**Interview Tip:** AR success depends on robust tracking (SLAM) and understanding scene geometry (depth, planes). Discuss trade-offs between quality and real-time performance on mobile devices.

---

