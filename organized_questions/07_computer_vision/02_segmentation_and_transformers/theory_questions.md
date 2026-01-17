# Segmentation & Vision Transformers - Interview Questions

## Semantic Segmentation

### Question 1
**Explain U-Net's encoder-decoder architecture with skip connections. Why are skip connections crucial for segmentation?**

**Answer:**

U-Net uses an encoder (contracting path) to capture context and a decoder (expanding path) to enable precise localization. Skip connections directly connect encoder layers to decoder layers at the same resolution, preserving fine-grained spatial information that would otherwise be lost during downsampling.

**Architecture:**

```
Encoder (Contracting)         Decoder (Expanding)
    [Input 572×572]
         ↓ Conv
    [64×568×568] ─────────→ [Concat] → [64×388×388]
         ↓ Pool                ↑
    [128×280×280] ────────→ [Concat] → [128×196×196]
         ↓ Pool                ↑
    [256×136×136] ────────→ [Concat] → [256×100×100]
         ↓ Pool                ↑
    [512×64×64] ──────────→ [Concat] → [512×52×52]
         ↓ Pool                ↑
    [1024×28×28] ─────────────→ UpConv
         (Bottleneck)
```

**Why Skip Connections Matter:**

| Without Skip Connections | With Skip Connections |
|-------------------------|----------------------|
| Only semantic info from bottleneck | Semantic + spatial info |
| Blurry boundaries | Sharp boundaries |
| Poor small object recovery | Better fine details |
| Localization errors | Precise localization |

**Key Insight:**
- Encoder: "What" (high-level features, semantic meaning)
- Decoder: "Where" (precise localization)
- Skip connections: Combine both for accurate segmentation

**Python Implementation:**
```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(self._double_conv(feature * 2, feature))
        
        # Final output
        self.final = nn.Conv2d(features[0], out_channels, 1)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder with skip connections
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)  # Skip connection
            x = self.decoder[idx + 1](x)  # Conv
        
        return self.final(x)
```

**Interview Tip:** Skip connections solve the information bottleneck problem—high-level features alone can't recover precise boundaries. The concatenation (not addition) doubles channels, giving the decoder rich information from multiple scales.

---

### Question 2
**What are the key innovations in DeepLabv3+ (atrous convolutions, ASPP, encoder-decoder) for boundary delineation?**

**Answer:**

DeepLabv3+ combines three key innovations: atrous (dilated) convolutions for multi-scale context without resolution loss, ASPP (Atrous Spatial Pyramid Pooling) for capturing objects at multiple scales, and an encoder-decoder structure for sharp boundary recovery.

**Key Innovations:**

| Component | Purpose | Benefit |
|-----------|---------|---------|
| Atrous Convolution | Expand receptive field | No resolution loss |
| ASPP | Multi-scale feature extraction | Handle varying object sizes |
| Encoder-Decoder | Low-level feature fusion | Sharp boundaries |

**Atrous Convolution:**

Standard convolution vs. atrous with rate r:
$$y[i] = \sum_k x[i + r \cdot k] \cdot w[k]$$

- Rate r=1: Standard convolution
- Rate r=2: Skips every other pixel, 2× receptive field
- Rate r=4: 4× receptive field, same parameters

**ASPP Module:**

Parallel atrous convolutions at different rates:
```
        Input Features
       /    |    |    \
   1×1   3×3   3×3   3×3   Image
   Conv  r=6  r=12  r=18   Pooling
       \    |    |    /
         Concatenate
             ↓
          1×1 Conv
             ↓
        ASPP Output
```

**DeepLabv3+ Architecture:**
```
Image → Backbone → ASPP → Decoder → Segmentation
              ↓              ↑
         Low-level     ← 1×1 Conv (reduce channels)
         Features
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        
        # 1×1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Atrous convolutions at different rates
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                         dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        # Image-level features (global context)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Multi-scale features
        features = [self.conv1(x)]
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        
        # Global features
        global_feat = self.image_pooling(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', 
                                     align_corners=False)
        features.append(global_feat)
        
        # Concatenate and project
        return self.project(torch.cat(features, dim=1))

class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with encoder-decoder"""
    def __init__(self, backbone, num_classes, low_level_channels=256):
        super().__init__()
        self.backbone = backbone
        self.aspp = ASPP(2048, 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Backbone features
        low_level, high_level = self.backbone(x)  # C2 and C5
        
        # ASPP on high-level features
        aspp_out = self.aspp(high_level)
        aspp_out = F.interpolate(aspp_out, size=low_level.shape[2:], 
                                  mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level = self.low_level_conv(low_level)
        
        # Concatenate and decode
        x = torch.cat([aspp_out, low_level], dim=1)
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        return x
```

**Comparison with Earlier Versions:**

| Version | Key Feature | Limitation Fixed |
|---------|-------------|-----------------|
| DeepLabv1 | Atrous convolution + CRF | Resolution loss |
| DeepLabv2 | ASPP | Multi-scale objects |
| DeepLabv3 | Improved ASPP + batch norm | Training stability |
| DeepLabv3+ | Encoder-decoder | Boundary sharpness |

**Interview Tip:** DeepLabv3+'s key insight is combining dense prediction (ASPP) with sharp boundaries (decoder). The low-level features provide edge information while ASPP provides semantic context. This is why it excels at boundary delineation.

---

### Question 3
**Compare Dice loss vs. Cross-entropy for segmentation. When would you use focal loss or Lovász loss?**

**Answer:**

Cross-entropy treats pixels independently; Dice loss optimizes global overlap. Use Dice for imbalanced classes, focal loss for hard examples, and Lovász loss for direct IoU optimization. Often combine multiple losses for best results.

**Loss Function Comparison:**

| Loss | Optimizes | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| Cross-Entropy | Per-pixel accuracy | Stable, well-understood | Ignores class imbalance |
| Dice | Overlap (F1 score) | Handles imbalance | Unstable for small objects |
| Focal | Hard examples | Reduces easy example weight | Hyperparameter sensitive |
| Lovász | IoU directly | Matches eval metric | Complex, slower |

**Mathematical Formulations:**

**Cross-Entropy:**
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{p}_{i,c})$$

**Dice Loss:**
$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

**Focal Loss:**
$$\mathcal{L}_{Focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

**Lovász-Softmax:**
Directly optimizes IoU using Lovász extension (submodular set function optimization).

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1.0, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight  # Class weights
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)
        
        dice_per_class = []
        for c in range(self.num_classes):
            p = pred[:, c].flatten()
            g = target_one_hot[:, c].float().flatten()
            
            intersection = (p * g).sum()
            dice = (2. * intersection + self.smooth) / \
                   (p.sum() + g.sum() + self.smooth)
            dice_per_class.append(dice)
        
        if self.weight is not None:
            dice_per_class = [d * w for d, w in zip(dice_per_class, self.weight)]
        
        return 1 - torch.stack(dice_per_class).mean()

class FocalLoss(nn.Module):
    """Focal Loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for IoU optimization"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        C = pred.shape[1]
        
        losses = []
        for c in range(C):
            fg = (target == c).float()
            errors = (fg - pred[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg.flatten()[perm]
            
            # Lovász extension
            gts = fg_sorted.sum()
            intersection = gts - fg_sorted.cumsum(0)
            union = gts + (1 - fg_sorted).cumsum(0)
            jaccard = 1 - intersection / union
            
            # Gradient of Jaccard
            jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
            losses.append((errors_sorted * jaccard).sum())
        
        return torch.stack(losses).mean()

class CombinedLoss(nn.Module):
    """Combine multiple losses"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5, focal_weight=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=21)
        self.focal = FocalLoss()
        
        self.weights = {
            'ce': ce_weight,
            'dice': dice_weight,
            'focal': focal_weight
        }
    
    def forward(self, pred, target):
        loss = 0
        if self.weights['ce'] > 0:
            loss += self.weights['ce'] * self.ce(pred, target)
        if self.weights['dice'] > 0:
            loss += self.weights['dice'] * self.dice(pred, target)
        if self.weights['focal'] > 0:
            loss += self.weights['focal'] * self.focal(pred, target)
        return loss
```

**When to Use Each:**

| Scenario | Recommended Loss |
|----------|------------------|
| Balanced classes | Cross-Entropy |
| Imbalanced (medical) | Dice or Dice + CE |
| Many small objects | Focal Loss |
| Maximize mIoU | Lovász-Softmax |
| General purpose | CE + Dice (0.5 each) |

**Interview Tip:** Dice loss directly optimizes overlap but can be unstable when target is empty. Cross-entropy is stable but ignores imbalance. Best practice: combine CE + Dice for stability and imbalance handling. Use Lovász when IoU is the evaluation metric.

---

### Question 4
**How do atrous/dilated convolutions capture multi-scale context without losing resolution?**

**Answer:**

Atrous convolutions insert gaps (zeros) between kernel weights, expanding the receptive field without increasing parameters or reducing spatial resolution. This allows capturing context at multiple scales while maintaining dense predictions.

**Standard vs. Atrous Convolution:**

| Aspect | Standard 3×3 | Atrous 3×3 (rate=2) |
|--------|-------------|---------------------|
| Receptive field | 3×3 | 5×5 |
| Parameters | 9 | 9 |
| Output resolution | Same (with padding) | Same |
| Context captured | Local | Larger |

**How It Works:**

Standard 3×3 kernel:
```
[x x x]     Receptive field: 3×3
[x x x]     Spacing: 1
[x x x]
```

Atrous 3×3 with rate=2:
```
[x . x . x]     Receptive field: 5×5
[. . . . .]     Spacing: 2
[x . x . x]     (dots are skipped pixels)
[. . . . .]
[x . x . x]
```

**Mathematical Definition:**
$$y[i] = \sum_{k=1}^{K} x[i + r \cdot k] \cdot w[k]$$

Where:
- $r$ = dilation rate
- $K$ = kernel size
- Effective kernel size = $K + (K-1) \times (r-1)$

**Multi-Scale Context:**
```
        Input Feature Map
       /        |        \
  Atrous     Atrous     Atrous
  rate=1     rate=2     rate=4
  (3×3)      (5×5)      (9×9)
       \        |        /
         Concatenate
              ↓
       Multi-scale Features
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousConv(nn.Module):
    """Single atrous convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MultiScaleAtrous(nn.Module):
    """Multi-scale context using atrous convolutions"""
    def __init__(self, in_channels, out_channels, rates=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList([
            AtrousConv(in_channels, out_channels, dilation=r) for r in rates
        ])
        self.fusion = nn.Conv2d(out_channels * len(rates), out_channels, 1)
    
    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(features, dim=1))

# Visualization of receptive fields
def visualize_receptive_field(kernel_size=3, dilation=1):
    """Show effective receptive field"""
    effective_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    
    grid = torch.zeros(effective_size, effective_size)
    center = effective_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            pos_i = i * dilation
            pos_j = j * dilation
            grid[pos_i, pos_j] = 1
    
    return grid

# Example: Compare receptive fields
for rate in [1, 2, 4]:
    rf = visualize_receptive_field(3, rate)
    print(f"Rate {rate}: {rf.shape[0]}×{rf.shape[0]} effective receptive field")
```

**Resolution Preservation:**
```python
# Standard approach: pooling reduces resolution
standard = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),  # H×W
    nn.MaxPool2d(2),                   # H/2×W/2  ← Resolution loss!
    nn.Conv2d(64, 64, 3, padding=1),
    nn.MaxPool2d(2),                   # H/4×W/4
)

# Atrous approach: maintains resolution
atrous = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, dilation=1),  # H×W
    nn.Conv2d(64, 64, 3, padding=2, dilation=2),  # H×W  ← Same resolution!
    nn.Conv2d(64, 64, 3, padding=4, dilation=4),  # H×W
)
```

**Trade-offs:**

| Advantage | Disadvantage |
|-----------|--------------|
| No resolution loss | Gridding artifacts at high rates |
| Same parameters | Memory for full resolution |
| Multi-scale context | Less effective for small objects |

**Gridding Problem Solution:**
Use hybrid rates (e.g., 6, 12, 18) rather than powers of 2 to avoid systematic gaps.

**Interview Tip:** Atrous convolutions are key to dense prediction tasks. They expand receptive field without downsampling, but beware of gridding artifacts. ASPP uses different rates to capture multi-scale context efficiently.

---

### Question 5
**Explain U-Net variants: U-Net++, Attention U-Net, and TransUNet. What problems does each solve?**

**Answer:**

U-Net variants address specific limitations: U-Net++ uses nested skip connections for better feature fusion, Attention U-Net adds attention gates to focus on relevant regions, and TransUNet incorporates transformers for global context. Each improves upon the original U-Net architecture.

**Comparison:**

| Variant | Key Innovation | Problem Solved |
|---------|----------------|----------------|
| U-Net | Skip connections | Basic encoder-decoder |
| U-Net++ | Nested/dense skip paths | Semantic gap in skips |
| Attention U-Net | Attention gates | Irrelevant feature suppression |
| TransUNet | Transformer encoder | Limited global context |

**U-Net++ (Nested U-Net):**

Problem: Direct skip connections have semantic gap between encoder and decoder.

Solution: Dense convolution blocks between encoder and decoder at each level.

```
Encoder                    Decoder
  X0,0 ─────────────────→ X0,4
   ↓      X0,1 ──→ X0,2 ──→ X0,3    ↑
  X1,0 ────→ X1,1 ──→ X1,2 ──→ X1,3 ↑
   ↓          X1,1 ──→ X1,2    ↑
  X2,0 ────────→ X2,1 ──→ X2,2 ↑
   ↓              ↑            ↑
  X3,0 ──────────→ X3,1 ───────↑
   ↓                           ↑
  X4,0 ────────────────────────↑
```

**Attention U-Net:**

Problem: Skip connections pass all features, including irrelevant ones.

Solution: Attention gates learn to suppress irrelevant regions.

Attention gate formula:
$$\alpha = \sigma(\psi^T(\sigma_1(W_x x + W_g g + b)))$$

Where:
- $x$ = skip features
- $g$ = gating signal from decoder
- $\alpha$ = attention coefficients

**TransUNet:**

Problem: CNNs have limited receptive field for global context.

Solution: Transformer encoder processes flattened CNN features.

```
Image → CNN Encoder → Flatten → Transformer → Reshape → CNN Decoder
                                   ↓
                            Global Context
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net++ Block
class NestedBlock(nn.Module):
    """Nested skip connection block for U-Net++"""
    def __init__(self, in_ch, out_ch, num_concat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * num_concat, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.conv(x)

# Attention Gate
class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection features
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear')
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi  # Weighted skip features

class AttentionUNet(nn.Module):
    """U-Net with attention gates"""
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # Encoder
        for f in features:
            self.encoder.append(self._conv_block(in_ch, f))
            in_ch = f
        
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder with attention
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.attention.append(AttentionGate(f, f, f // 2))
            self.decoder.append(self._conv_block(f * 2, f))
        
        self.final = nn.Conv2d(features[0], out_ch, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skips = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skips = skips[::-1]
        
        # Decoder with attention
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample
            skip = self.attention[i // 2](x, skips[i // 2])  # Attention
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i + 1](x)  # Conv
        
        return self.final(x)

# TransUNet simplified
class TransUNet(nn.Module):
    """U-Net with Transformer encoder"""
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            # ResNet-like encoder
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        # Transformer encoder
        self.patch_embed = nn.Conv2d(64, 768, patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=12
        )
        
        # Decoder
        self.decoder = UNetDecoder(768, num_classes)
    
    def forward(self, x):
        # CNN features
        cnn_feat = self.cnn_encoder(x)
        
        # Patch embedding
        patches = self.patch_embed(cnn_feat)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        
        # Transformer
        transformer_out = self.transformer(patches)
        
        # Reshape and decode
        transformer_out = transformer_out.transpose(1, 2).view(B, C, H, W)
        return self.decoder(transformer_out)
```

**When to Use Each:**

| Scenario | Best Choice |
|----------|-------------|
| Medical imaging (standard) | U-Net++ |
| Noisy backgrounds | Attention U-Net |
| Need global context | TransUNet |
| Resource constrained | Original U-Net |

**Interview Tip:** U-Net++ densely connects all levels (like DenseNet for skip connections). Attention U-Net adds soft attention to filter skip features. TransUNet brings transformer's global context to segmentation. Choose based on your specific needs.

---

### Question 6
**How do you handle class imbalance in segmentation when some classes occupy very few pixels?**

**Answer:**

Class imbalance in segmentation is addressed through: weighted loss functions (inverse frequency weighting), specialized losses (Dice, focal), oversampling of rare class images, class-balanced batching, and data augmentation targeting minority classes.

**Common Strategies:**

| Strategy | Level | Approach |
|----------|-------|----------|
| Loss reweighting | Loss | Higher weight for rare classes |
| Dice/Focal loss | Loss | Directly handles imbalance |
| Oversampling | Data | More samples with rare classes |
| Class-balanced batching | Training | Equal class representation |
| Copy-paste augmentation | Data | Add rare class instances |

**Weight Calculation Methods:**

| Method | Formula | When to Use |
|--------|---------|-------------|
| Inverse frequency | $w_c = \frac{1}{f_c}$ | Standard |
| Median frequency | $w_c = \frac{median(f)}{f_c}$ | Very imbalanced |
| Effective number | $w_c = \frac{1-\beta}{1-\beta^{n_c}}$ | Extreme imbalance |
| Log-smoothed | $w_c = \frac{1}{\log(1.1 + f_c)}$ | Prevent extreme weights |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_class_weights(dataset, num_classes, method='inverse'):
    """Compute class weights from dataset statistics"""
    pixel_counts = np.zeros(num_classes)
    
    for _, mask in dataset:
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()
    
    frequencies = pixel_counts / pixel_counts.sum()
    
    if method == 'inverse':
        weights = 1.0 / (frequencies + 1e-8)
    elif method == 'median':
        median_freq = np.median(frequencies[frequencies > 0])
        weights = median_freq / (frequencies + 1e-8)
    elif method == 'log':
        weights = 1.0 / np.log(1.1 + frequencies)
    
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

class WeightedCrossEntropy(nn.Module):
    """Cross-entropy with class weights"""
    def __init__(self, class_weights, ignore_index=255):
        super().__init__()
        self.weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        return F.cross_entropy(
            pred, target, 
            weight=self.weights.to(pred.device),
            ignore_index=self.ignore_index
        )

class BalancedDiceLoss(nn.Module):
    """Dice loss with class balancing"""
    def __init__(self, num_classes, class_weights=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.weights = class_weights
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)
        
        dice_per_class = []
        for c in range(self.num_classes):
            p = pred[:, c].flatten()
            g = target_one_hot[:, c].float().flatten()
            
            intersection = (p * g).sum()
            dice = (2. * intersection + self.smooth) / \
                   (p.sum() + g.sum() + self.smooth)
            dice_per_class.append(dice)
        
        dice_tensor = torch.stack(dice_per_class)
        
        if self.weights is not None:
            weights = self.weights.to(pred.device)
            return 1 - (dice_tensor * weights).sum() / weights.sum()
        
        return 1 - dice_tensor.mean()

class FocalLossWithClassBalance(nn.Module):
    """Focal loss with class balancing"""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Per-class weight
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        
        # Get alpha for each pixel based on its class
        alpha = self.alpha[target]
        
        focal = alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()

# Class-balanced sampling
class ClassBalancedSampler(torch.utils.data.Sampler):
    """Sampler that balances class representation"""
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.indices_per_class = self._group_by_class()
    
    def _group_by_class(self):
        """Group image indices by dominant class"""
        indices = {c: [] for c in range(self.num_classes)}
        
        for idx, (_, mask) in enumerate(self.dataset):
            # Find dominant class in this image
            class_counts = [(mask == c).sum() for c in range(self.num_classes)]
            dominant = np.argmax(class_counts)
            indices[dominant].append(idx)
        
        return indices
    
    def __iter__(self):
        # Sample equally from each class
        samples = []
        max_samples = max(len(v) for v in self.indices_per_class.values())
        
        for c in range(self.num_classes):
            if len(self.indices_per_class[c]) > 0:
                samples.extend(np.random.choice(
                    self.indices_per_class[c], 
                    max_samples, 
                    replace=True
                ))
        
        np.random.shuffle(samples)
        return iter(samples)
    
    def __len__(self):
        return len(self.dataset)

# Copy-paste augmentation for rare classes
class CopyPasteAugmentation:
    """Copy instances of rare classes to other images"""
    def __init__(self, rare_classes, paste_prob=0.5):
        self.rare_classes = rare_classes
        self.paste_prob = paste_prob
        self.instance_bank = {c: [] for c in rare_classes}
    
    def add_to_bank(self, image, mask):
        for c in self.rare_classes:
            if (mask == c).any():
                # Extract instance
                binary_mask = (mask == c)
                instance = image * binary_mask
                self.instance_bank[c].append((instance, binary_mask))
    
    def __call__(self, image, mask):
        if np.random.random() > self.paste_prob:
            return image, mask
        
        # Pick a rare class to paste
        c = np.random.choice(self.rare_classes)
        if len(self.instance_bank[c]) == 0:
            return image, mask
        
        # Get random instance
        instance, inst_mask = random.choice(self.instance_bank[c])
        
        # Random position
        y = np.random.randint(0, image.shape[1] - inst_mask.shape[0])
        x = np.random.randint(0, image.shape[2] - inst_mask.shape[1])
        
        # Paste
        image[:, y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]] = \
            torch.where(inst_mask, instance, 
                       image[:, y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]])
        mask[y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]] = \
            torch.where(inst_mask, c, 
                       mask[y:y+inst_mask.shape[0], x:x+inst_mask.shape[1]])
        
        return image, mask
```

**Best Practices:**

| Imbalance Ratio | Recommended Approach |
|-----------------|---------------------|
| <10:1 | Weighted CE |
| 10:1 - 100:1 | Dice + Weighted CE |
| >100:1 | Focal + Copy-paste + Oversampling |

**Interview Tip:** Combine multiple strategies for severe imbalance. Dice loss helps but fails when a class is absent. Use online hard example mining (OHEM) to focus on difficult pixels. Monitor per-class IoU, not just mIoU.

---

### Question 7
**What techniques improve segmentation performance on small or thin objects (boundary loss, deep supervision)?**

**Answer:**

Small and thin objects are challenging due to resolution loss during downsampling. Solutions include: boundary-aware losses, deep supervision at multiple scales, high-resolution feature branches, and dedicated boundary detection modules.

**Challenges:**

| Issue | Cause | Effect |
|-------|-------|--------|
| Resolution loss | Pooling/striding | Small objects disappear |
| Class imbalance | Few pixels | Model ignores them |
| Blurry boundaries | Upsampling artifacts | Imprecise edges |

**Techniques:**

| Technique | How It Helps |
|-----------|-------------|
| Boundary loss | Penalizes boundary errors directly |
| Deep supervision | Gradients at multiple resolutions |
| HRNet | Maintains high-resolution throughout |
| Edge detection branch | Explicit boundary learning |

**Boundary Loss:**

Distance-based loss using distance transform:
$$\mathcal{L}_{boundary} = \sum_p \phi(p) \cdot (y_p - \hat{y}_p)^2$$

Where $\phi(p)$ = distance to nearest boundary.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

class BoundaryLoss(nn.Module):
    """Loss that weights pixels based on distance to boundary"""
    def __init__(self, theta=10):
        super().__init__()
        self.theta = theta
    
    def compute_distance_map(self, mask):
        """Compute distance transform from boundary"""
        mask_np = mask.cpu().numpy()
        batch_dist = []
        
        for m in mask_np:
            # Find edges
            edges = ndimage.sobel(m.astype(float))
            edges = (np.abs(edges) > 0).astype(float)
            
            # Distance transform
            dist = ndimage.distance_transform_edt(1 - edges)
            batch_dist.append(dist)
        
        return torch.tensor(np.stack(batch_dist), device=mask.device)
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # Compute boundary weights
        dist_map = self.compute_distance_map(target)
        boundary_weights = torch.exp(-dist_map / self.theta)
        
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * boundary_weights
        
        return weighted_loss.mean()

class HDLoss(nn.Module):
    """Hausdorff Distance Loss for boundary accuracy"""
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]  # Binary
        
        # Distance transforms
        dist_pred = self._distance_transform(pred > 0.5)
        dist_target = self._distance_transform(target > 0)
        
        # Hausdorff-like loss
        hd_loss = (pred - target).abs() * (dist_pred + dist_target).pow(self.alpha)
        
        return hd_loss.mean()

class DeepSupervision(nn.Module):
    """Deep supervision with losses at multiple scales"""
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        
        # Auxiliary heads at different decoder stages
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, 1),  # 1/4 scale
            nn.Conv2d(128, num_classes, 1),  # 1/2 scale
        ])
        
        self.main_head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x, target=None):
        # Get intermediate features
        features = self.base.encode(x)
        
        outputs = []
        for i, (feat, head) in enumerate(zip(
            [features['1/4'], features['1/2'], features['1/1']],
            self.aux_heads + [self.main_head]
        )):
            out = head(feat)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear')
            outputs.append(out)
        
        if target is not None:
            # Weighted loss from all outputs
            loss = 0
            weights = [0.4, 0.4, 1.0]  # Higher weight for final output
            for out, w in zip(outputs, weights):
                loss += w * F.cross_entropy(out, target)
            return outputs[-1], loss
        
        return outputs[-1]

class BoundaryAwareUNet(nn.Module):
    """U-Net with explicit boundary branch"""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.encoder = UNetEncoder(in_ch)
        
        # Main segmentation decoder
        self.seg_decoder = UNetDecoder(num_classes)
        
        # Boundary detection branch
        self.boundary_decoder = UNetDecoder(1)  # Binary boundary
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes + 1, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        # Segmentation
        seg = self.seg_decoder(features)
        
        # Boundary
        boundary = self.boundary_decoder(features)
        
        # Fuse boundary with segmentation
        fused = self.fusion(torch.cat([seg, boundary], dim=1))
        
        return fused, boundary

class MultiScaleLoss(nn.Module):
    """Loss computed at multiple scales"""
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.weights = weights
    
    def forward(self, pred, target):
        total_loss = 0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # Downsample prediction and target
                h, w = int(pred.shape[2] * scale), int(pred.shape[3] * scale)
                pred_scaled = F.interpolate(pred, (h, w), mode='bilinear')
                target_scaled = F.interpolate(
                    target.unsqueeze(1).float(), (h, w), mode='nearest'
                ).squeeze(1).long()
            else:
                pred_scaled, target_scaled = pred, target
            
            loss = F.cross_entropy(pred_scaled, target_scaled)
            total_loss += weight * loss
        
        return total_loss
```

**HRNet Approach:**
```python
class HighResolutionBranch(nn.Module):
    """Maintain high resolution features throughout"""
    def __init__(self):
        super().__init__()
        # Multiple parallel branches at different resolutions
        self.branch_1x = nn.ModuleList()  # Full resolution
        self.branch_2x = nn.ModuleList()  # 1/2 resolution
        self.branch_4x = nn.ModuleList()  # 1/4 resolution
        
        # Cross-resolution fusion
        self.fuse = CrossResolutionFusion()
    
    def forward(self, x):
        x_1x = x
        x_2x = F.max_pool2d(x, 2)
        x_4x = F.max_pool2d(x_2x, 2)
        
        for b1, b2, b4 in zip(self.branch_1x, self.branch_2x, self.branch_4x):
            x_1x, x_2x, x_4x = self.fuse(b1(x_1x), b2(x_2x), b4(x_4x))
        
        return x_1x  # High-resolution output
```

**Results:**

| Technique | mIoU Gain | Boundary F1 Gain |
|-----------|-----------|------------------|
| Boundary loss | +1-2% | +5-10% |
| Deep supervision | +2-3% | +3-5% |
| HRNet | +3-5% | +8-12% |
| Combined | +5-8% | +15-20% |

**Interview Tip:** For small objects, avoid aggressive downsampling (use dilated convolutions instead). Deep supervision helps gradients flow to early layers. Boundary losses are essential for thin structures like vessels or roads.

---

### Question 8
**Explain the overlap-tile strategy for inference on large images that don't fit in memory.**

**Answer:**

Overlap-tile strategy divides large images into overlapping tiles, processes each independently, and stitches results using the overlap to avoid boundary artifacts. Essential for high-resolution medical, satellite, and pathology images.

**Why Overlap is Needed:**

| Without Overlap | With Overlap |
|-----------------|--------------|
| Edge artifacts | Smooth boundaries |
| Inconsistent predictions | Consistent results |
| Missing context at borders | Full context everywhere |

**Strategy:**

```
Original Large Image (8000×8000)
         ↓ Divide into tiles
[Tile 1][Tile 2][Tile 3]...
    ↓        ↓        ↓
  Process each tile
    ↓        ↓        ↓
[Result 1][Result 2][Result 3]
         ↓ Stitch (use center, discard edges)
Final Prediction (8000×8000)
```

**Key Parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Tile size | Processing patch size | 256-1024 px |
| Overlap | Shared pixels between tiles | 32-128 px |
| Stride | Step between tiles | tile_size - overlap |

**Python Implementation:**
```python
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class TiledInference:
    """Sliding window inference for large images"""
    
    def __init__(self, model, tile_size=512, overlap=64, batch_size=4):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.batch_size = batch_size
    
    def predict(self, image):
        """
        Process large image using overlap-tile strategy
        
        image: numpy array (H, W, C) or (H, W)
        returns: prediction array (H, W) or (H, W, num_classes)
        """
        H, W = image.shape[:2]
        
        # Pad image to fit tiles exactly
        pad_h = (self.stride - (H - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.tile_size) % self.stride) % self.stride
        
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Initialize output and count arrays
        H_pad, W_pad = padded.shape[:2]
        output = np.zeros((H_pad, W_pad), dtype=np.float32)
        counts = np.zeros((H_pad, W_pad), dtype=np.float32)
        
        # Generate tile coordinates
        tiles = []
        coords = []
        for y in range(0, H_pad - self.tile_size + 1, self.stride):
            for x in range(0, W_pad - self.tile_size + 1, self.stride):
                tile = padded[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(tile)
                coords.append((y, x))
        
        # Process in batches
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i:i+self.batch_size]
            batch_coords = coords[i:i+self.batch_size]
            
            # Prepare batch
            batch = np.stack(batch_tiles)
            batch = torch.tensor(batch).permute(0, 3, 1, 2).float() / 255.0
            
            # Inference
            with torch.no_grad():
                preds = self.model(batch.cuda())
                preds = torch.argmax(preds, dim=1).cpu().numpy()
            
            # Create weight mask (higher weight in center)
            weight = self._create_weight_mask()
            
            # Accumulate predictions
            for (y, x), pred in zip(batch_coords, preds):
                output[y:y+self.tile_size, x:x+self.tile_size] += pred * weight
                counts[y:y+self.tile_size, x:x+self.tile_size] += weight
        
        # Average overlapping regions
        output = output / (counts + 1e-8)
        
        # Remove padding
        output = output[:H, :W]
        
        return output.astype(np.uint8)
    
    def _create_weight_mask(self):
        """Create weight mask: higher in center, lower at edges"""
        x = np.linspace(0, 1, self.tile_size)
        y = np.linspace(0, 1, self.tile_size)
        xx, yy = np.meshgrid(x, y)
        
        # Sigmoid-like weights
        weight_x = np.minimum(xx, 1 - xx) * 2
        weight_y = np.minimum(yy, 1 - yy) * 2
        weight = np.minimum(weight_x, weight_y)
        
        # Smooth edges
        weight = np.clip(weight * 4, 0, 1)
        
        return weight

class SmartTiledInference(TiledInference):
    """Improved tiling with multi-scale support"""
    
    def __init__(self, model, tile_size=512, overlap=64, scales=[1.0]):
        super().__init__(model, tile_size, overlap)
        self.scales = scales
    
    def predict_multiscale(self, image):
        """Average predictions across multiple scales"""
        H, W = image.shape[:2]
        all_preds = []
        
        for scale in self.scales:
            # Resize image
            h_scaled = int(H * scale)
            w_scaled = int(W * scale)
            
            if scale != 1.0:
                image_scaled = np.array(Image.fromarray(image).resize(
                    (w_scaled, h_scaled), Image.BILINEAR
                ))
            else:
                image_scaled = image
            
            # Process at this scale
            pred = self.predict(image_scaled)
            
            # Resize prediction back
            if scale != 1.0:
                pred = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST))
            
            all_preds.append(pred)
        
        # Vote or average
        stacked = np.stack(all_preds)
        final = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=0, arr=stacked
        )
        
        return final

# Soft probability accumulation
def predict_with_soft_accumulation(model, image, tile_size, overlap, num_classes):
    """Accumulate soft probabilities instead of hard labels"""
    H, W = image.shape[:2]
    stride = tile_size - overlap
    
    # Initialize probability accumulator
    probs = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            tile_tensor = preprocess(tile).unsqueeze(0).cuda()
            
            with torch.no_grad():
                pred = F.softmax(model(tile_tensor), dim=1)
                pred = pred.squeeze().cpu().numpy()
            
            # Accumulate probabilities
            probs[:, y:y+tile_size, x:x+tile_size] += pred
            counts[y:y+tile_size, x:x+tile_size] += 1
    
    # Average
    probs = probs / (counts + 1e-8)
    
    # Argmax for final prediction
    return np.argmax(probs, axis=0)
```

**Memory-Efficient Tips:**

| Tip | Benefit |
|-----|---------|
| Process tiles in batches | GPU utilization |
| Use half precision (FP16) | 2× memory savings |
| Delete tensors after use | Free GPU memory |
| Use soft accumulation | Better boundaries |

**Interview Tip:** Overlap should be at least 2× the model's receptive field boundary effect. Weighted averaging (center > edges) gives smoother results than hard boundaries. Multi-scale can improve accuracy but increases computation.

---

### Question 9
**How do you implement data augmentation that preserves spatial relationships for segmentation?**

**Answer:**

Segmentation augmentation must apply identical geometric transforms to both image and mask. Photometric augmentations (brightness, contrast) apply only to images. Use augmentation libraries that handle paired transforms (Albumentations, torchvision.transforms.v2).

**Augmentation Categories:**

| Type | Apply to Image | Apply to Mask |
|------|----------------|---------------|
| Geometric (flip, rotate) | ✓ | ✓ |
| Photometric (color, brightness) | ✓ | ✗ |
| Noise/blur | ✓ | ✗ |
| Crop/resize | ✓ | ✓ (same coords) |

**Common Segmentation Augmentations:**

| Augmentation | Effect | Caution |
|--------------|--------|---------|
| Random flip | Invariance to orientation | May not suit directional data |
| Random crop | Scale invariance | Must keep enough context |
| Rotation | Orientation invariance | Use appropriate interpolation |
| Elastic deform | Shape variation | Can create artifacts |
| Scale jitter | Size invariance | Keep objects visible |

**Python Implementation:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision.transforms.v2 as T

# Using Albumentations (recommended)
class SegmentationTransforms:
    """Paired transforms for image and mask"""
    
    def __init__(self, train=True, img_size=256):
        if train:
            self.transform = A.Compose([
                # Geometric (applied to both)
                A.RandomResizedCrop(img_size, img_size, scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                
                # Elastic deformation (applied to both)
                A.ElasticTransform(alpha=120, sigma=6, p=0.3),
                
                # Photometric (image only)
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=7, p=0.3),
                
                # Normalize and convert
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image, mask):
        # Albumentations handles paired transforms automatically
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

# Using torchvision v2 transforms
class TorchvisionSegmentationTransforms:
    """Using torchvision.transforms.v2 for segmentation"""
    
    def __init__(self, train=True, img_size=256):
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45),
            ])
            
            # Image-only transforms
            self.image_transform = T.Compose([
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.GaussianBlur(7, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.mask_transform = T.Compose([
                T.ToTensor()
            ])
        else:
            self.transform = T.Resize((img_size, img_size))
            self.image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.mask_transform = T.ToTensor()
    
    def __call__(self, image, mask):
        # Apply paired geometric transforms
        # Stack for joint transform
        stacked = torch.cat([
            torch.tensor(image).permute(2, 0, 1),
            torch.tensor(mask).unsqueeze(0)
        ], dim=0)
        
        transformed = self.transform(stacked)
        
        image = transformed[:3]
        mask = transformed[3:]
        
        # Apply image-only transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask.long().squeeze(0)

# Custom paired transforms
class PairedTransform:
    """Manual paired transform implementation"""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, mask):
        # Use same random seed for both
        seed = np.random.randint(2147483647)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        image = self._apply_transforms(image)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        mask = self._apply_transforms(mask, is_mask=True)
        
        return image, mask
    
    def _apply_transforms(self, x, is_mask=False):
        for t in self.transforms:
            if hasattr(t, 'is_geometric') and t.is_geometric:
                x = t(x)
            elif not is_mask:
                x = t(x)
        return x

# CutMix/MixUp for segmentation
class CutMixSegmentation:
    """CutMix augmentation for segmentation"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, image1, mask1, image2, mask2):
        lam = np.random.beta(self.alpha, self.alpha)
        
        H, W = image1.shape[1:3]
        
        # Random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        
        # Cut and paste
        image1[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        
        return image1, mask1

# Copy-paste augmentation
class CopyPasteSegmentation:
    """Copy instances from one image to another"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image1, mask1, image2, mask2):
        if np.random.random() > self.p:
            return image1, mask1
        
        # Find instances in image2
        unique_ids = torch.unique(mask2)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        
        if len(unique_ids) == 0:
            return image1, mask1
        
        # Select random instance
        instance_id = unique_ids[np.random.randint(len(unique_ids))]
        instance_mask = (mask2 == instance_id)
        
        # Paste instance
        image1[:, instance_mask] = image2[:, instance_mask]
        mask1[instance_mask] = instance_id
        
        return image1, mask1
```

**Augmentation Pipeline Example:**
```python
# Complete training pipeline
train_transform = A.Compose([
    # Resize
    A.RandomResizedCrop(512, 512, scale=(0.5, 1.5)),
    
    # Flips
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    
    # Rotation
    A.RandomRotate90(p=0.5),
    
    # Color
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ], p=0.5),
    
    # Noise/blur
    A.OneOf([
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),
    
    # Normalize
    A.Normalize(),
    ToTensorV2()
])
```

**Interview Tip:** Always use the same random seed for paired transforms. Albumentations is the go-to library for segmentation augmentation. For medical imaging, elastic deformation is particularly effective. Avoid heavy augmentation that makes the task too hard.

---

### Question 10
**What approaches work for real-time semantic segmentation (BiSeNet, Fast-SCNN, EfficientPS)?**

**Answer:**

Real-time segmentation uses efficient architectures with two-branch designs (spatial + context), depthwise separable convolutions, early downsampling, and attention-based feature fusion. Target: >30 FPS on standard hardware while maintaining reasonable accuracy.

**Speed vs Accuracy Trade-off:**

| Model | mIoU (Cityscapes) | FPS (1024×2048) | Key Feature |
|-------|-------------------|-----------------|-------------|
| BiSeNet | 68.4% | 105 | Two-branch |
| Fast-SCNN | 68.0% | 123 | Learning to downsample |
| EfficientPS | 84.2% | 10 | Panoptic, accurate |
| DDRNet | 79.4% | 37 | Deep dual resolution |
| STDC | 76.8% | 97 | Short-term dense concat |

**BiSeNet Architecture:**

```
        Input Image
       /           \
Spatial Path    Context Path
(preserve       (semantic info)
 detail)        
 3 Conv3×3      Backbone (Xception)
    ↓                ↓
    ↓           Global Pooling + ARM
    ↓                ↓
      → Fusion (FFM) ←
             ↓
        Segmentation
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution (efficient)
class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))

# BiSeNet Components
class SpatialPath(nn.Module):
    """Preserve spatial information"""
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # 1/8 resolution

class ContextPath(nn.Module):
    """Extract semantic context"""
    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.backbone = self._get_backbone(backbone)
        self.arm1 = AttentionRefinementModule(256, 128)
        self.arm2 = AttentionRefinementModule(512, 128)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        
        # Global context
        global_ctx = self.global_context(feat32)
        global_ctx = F.interpolate(global_ctx, feat32.shape[2:], mode='nearest')
        
        # ARM processing
        feat32 = self.arm2(feat32) + global_ctx
        feat16 = self.arm1(feat16)
        
        # Upsample and combine
        feat32_up = F.interpolate(feat32, feat16.shape[2:], mode='bilinear')
        return torch.cat([feat16, feat32_up], dim=1)

class AttentionRefinementModule(nn.Module):
    """Channel attention for feature refinement"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        att = self.attention(x)
        return x * att

class FeatureFusionModule(nn.Module):
    """Fuse spatial and context features"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial, context):
        x = torch.cat([spatial, context], dim=1)
        x = self.conv(x)
        att = self.attention(x)
        return x + x * att

class BiSeNet(nn.Module):
    """Complete BiSeNet for real-time segmentation"""
    def __init__(self, num_classes, backbone='resnet18'):
        super().__init__()
        self.spatial_path = SpatialPath(3, 128)
        self.context_path = ContextPath(backbone)
        self.ffm = FeatureFusionModule(128 + 256, 256)
        self.head = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        
        spatial = self.spatial_path(x)
        context = self.context_path(x)
        
        # Resize context to match spatial
        context = F.interpolate(context, spatial.shape[2:], mode='bilinear')
        
        fused = self.ffm(spatial, context)
        out = self.head(fused)
        
        return F.interpolate(out, size, mode='bilinear')

# Fast-SCNN: Learning to Downsample
class LearningToDownsample(nn.Module):
    """Efficient downsampling module"""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 32, 3, stride=2, padding=1)
        self.dsconv1 = DSConv(32, 48, stride=2)
        self.dsconv2 = DSConv(48, out_ch, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x  # 1/8 resolution

class FastSCNN(nn.Module):
    """Fast Semantic Segmentation"""
    def __init__(self, num_classes):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(3, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128)
        self.feature_fusion = FeatureFusionModule(64 + 128, 128)
        self.classifier = nn.Sequential(
            DSConv(128, 128),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Learning to downsample
        lower = self.learning_to_downsample(x)
        
        # Global features
        upper = self.global_feature_extractor(lower)
        
        # Fusion
        fused = self.feature_fusion(lower, upper)
        
        # Classify
        out = self.classifier(fused)
        return F.interpolate(out, size, mode='bilinear')
```

**Optimization Techniques:**

| Technique | Speedup | Accuracy Loss |
|-----------|---------|---------------|
| Depthwise separable conv | 2-3× | <1% |
| Early downsampling | 2× | 2-3% |
| Lower resolution input | 4× | 3-5% |
| TensorRT/ONNX | 2× | 0% |

**Interview Tip:** BiSeNet's key insight is separating spatial (detail) and semantic (context) processing into two branches. For production, always optimize with TensorRT. Trade-off: use lower resolution or fewer channels for speed, but test on your specific use case.

---

### Question 11
**How do you handle segmentation of objects with fuzzy or ambiguous boundaries?**

**Answer:**

Fuzzy boundaries (smoke, hair, glass) require soft predictions instead of hard labels. Use soft labels during training, probabilistic outputs, boundary-aware losses, and uncertainty estimation. Matting techniques help for semi-transparent objects.

**Challenges:**

| Object Type | Issue | Example |
|-------------|-------|---------|
| Semi-transparent | No clear boundary | Glass, smoke |
| Fine details | Sub-pixel accuracy needed | Hair, fur |
| Motion blur | Boundaries spread | Fast-moving objects |
| Gradual transitions | No sharp edge | Clouds, fog |

**Approaches:**

| Approach | How It Helps |
|----------|-------------|
| Soft labels | Allow fractional class membership |
| Alpha matting | Predict opacity (0-1) instead of binary |
| Uncertainty output | Model predicts confidence |
| Multi-label | Pixel can belong to multiple classes |

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSegmentationModel(nn.Module):
    """Model that outputs soft probabilities"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Main segmentation head
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        logits = self.seg_head(features)
        probs = F.softmax(logits, dim=1)
        
        uncertainty = self.uncertainty_head(features)
        
        return probs, uncertainty

class SoftLabelLoss(nn.Module):
    """Loss for soft/fuzzy labels"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target, soft_target=None):
        if soft_target is not None:
            # Use provided soft labels
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(soft_target * log_probs).sum(dim=1).mean()
        else:
            # Label smoothing
            num_classes = pred.shape[1]
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(smooth_target * log_probs).sum(dim=1).mean()
        
        return loss

class AlphaMattingHead(nn.Module):
    """Predict alpha matte for transparency"""
    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, coarse_mask):
        # Combine features with coarse mask
        combined = torch.cat([features, coarse_mask], dim=1)
        alpha = self.refine(combined)
        return alpha  # Values in [0, 1]

class MattingLoss(nn.Module):
    """Loss for alpha matting"""
    def __init__(self, alpha_weight=1.0, comp_weight=1.0, grad_weight=0.5):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.comp_weight = comp_weight
        self.grad_weight = grad_weight
    
    def forward(self, pred_alpha, gt_alpha, image, fg, bg):
        # Alpha loss
        alpha_loss = F.l1_loss(pred_alpha, gt_alpha)
        
        # Composition loss
        pred_comp = pred_alpha * fg + (1 - pred_alpha) * bg
        gt_comp = gt_alpha * fg + (1 - gt_alpha) * bg
        comp_loss = F.l1_loss(pred_comp, gt_comp)
        
        # Gradient loss (preserve edges)
        pred_grad = self._gradient(pred_alpha)
        gt_grad = self._gradient(gt_alpha)
        grad_loss = F.l1_loss(pred_grad, gt_grad)
        
        return (self.alpha_weight * alpha_loss + 
                self.comp_weight * comp_loss + 
                self.grad_weight * grad_loss)
    
    def _gradient(self, x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return torch.abs(grad_x).mean() + torch.abs(grad_y).mean()

class BoundaryUncertaintyModel(nn.Module):
    """Model with boundary-aware uncertainty"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.seg_head = nn.Conv2d(256, num_classes, 1)
        
        # Boundary detection
        self.boundary_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        seg = self.seg_head(features)
        boundary = self.boundary_head(features)
        
        return seg, boundary
    
    def predict_soft(self, x):
        seg, boundary = self.forward(x)
        probs = F.softmax(seg, dim=1)
        
        # Soften predictions near boundaries
        boundary_expanded = boundary.expand_as(probs)
        soft_probs = probs * (1 - boundary_expanded * 0.3)  # Reduce confidence at boundaries
        
        return soft_probs

# Training with soft labels
def create_soft_labels(hard_mask, sigma=2):
    """Create soft labels by blurring boundaries"""
    from scipy.ndimage import gaussian_filter
    
    num_classes = hard_mask.max() + 1
    soft_labels = np.zeros((*hard_mask.shape, num_classes))
    
    for c in range(num_classes):
        binary = (hard_mask == c).astype(float)
        soft_labels[..., c] = gaussian_filter(binary, sigma)
    
    # Normalize
    soft_labels = soft_labels / soft_labels.sum(axis=-1, keepdims=True)
    
    return soft_labels

# Trimap-based matting
def generate_trimap(mask, erosion=10, dilation=10):
    """Create trimap: definite foreground, definite background, unknown"""
    import cv2
    
    kernel = np.ones((3, 3), np.uint8)
    
    fg = cv2.erode(mask.astype(np.uint8), kernel, iterations=erosion)
    bg = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation)
    
    trimap = np.zeros_like(mask)
    trimap[bg == 0] = 0  # Definite background
    trimap[fg == 1] = 255  # Definite foreground
    trimap[(bg == 1) & (fg == 0)] = 128  # Unknown
    
    return trimap
```

**Evaluation Metrics:**

| Metric | For | Formula |
|--------|-----|---------|
| Soft IoU | Soft predictions | $\frac{\sum \min(p, g)}{\sum \max(p, g)}$ |
| MSE | Alpha matting | $\frac{1}{N}\sum(p - g)^2$ |
| Gradient error | Edge quality | $\|\nabla p - \nabla g\|$ |
| SAD | Overall | $\sum|p - g|$ |

**Interview Tip:** For fuzzy boundaries, avoid hard argmax predictions. Output soft probabilities and let downstream tasks decide threshold. Alpha matting is essential for compositing applications (green screen, portrait mode).

---

### Question 12
**Explain weakly supervised segmentation using image-level labels or bounding boxes instead of pixel masks.**

**Answer:**

Weakly supervised segmentation trains with cheap annotations (image labels, bounding boxes, scribbles) instead of expensive pixel-level masks. Uses CAM (Class Activation Maps) for localization, pseudo-label generation, and iterative refinement to achieve near-fully-supervised performance.

**Supervision Levels:**

| Level | Annotation | Cost | Quality |
|-------|------------|------|---------|
| Fully supervised | Pixel masks | Very High | Best |
| Bounding boxes | Boxes | Medium | Good |
| Image-level | Class labels | Low | Moderate |
| Scribbles | Few clicks | Low | Moderate |

**Approaches:**

| Method | Input | Technique |
|--------|-------|-----------|
| CAM-based | Image labels | Grad-CAM → pseudo-mask |
| BoxSup | Bounding boxes | Box as supervision |
| ScribbleSup | Scribbles | Propagate labels |
| Self-training | Any weak | Iterative pseudo-labeling |

**CAM-Based Pipeline:**
```
Image → Classifier → Grad-CAM → Threshold → Pseudo-mask → Train Segmentation
                                    ↓
                              CRF Refinement
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassActivationMap:
    """Generate CAMs from classifier"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, class_idx):
        # Forward pass
        output = self.model(image)
        
        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Weight by gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / cam.max()
        
        # Upsample to image size
        cam = F.interpolate(cam, image.shape[2:], mode='bilinear')
        
        return cam.squeeze().cpu().numpy()

class WeaklySupervised Segmentation:
    """Weakly supervised segmentation with image-level labels"""
    
    def __init__(self, classifier, segmenter, num_classes):
        self.classifier = classifier
        self.segmenter = segmenter
        self.num_classes = num_classes
        self.cam_generator = ClassActivationMap(classifier, classifier.layer4)
    
    def generate_pseudo_masks(self, images, labels, threshold=0.4):
        """Generate pseudo-masks from CAMs"""
        pseudo_masks = []
        
        for image, label in zip(images, labels):
            mask = np.zeros(image.shape[1:], dtype=np.int64)
            
            # Generate CAM for each present class
            for class_idx in label.nonzero()[0]:
                cam = self.cam_generator.generate_cam(
                    image.unsqueeze(0), class_idx
                )
                
                # Threshold
                class_mask = cam > threshold
                mask[class_mask] = class_idx + 1  # +1 for background=0
            
            # CRF refinement
            mask = self.crf_refine(image, mask)
            pseudo_masks.append(mask)
        
        return pseudo_masks
    
    def crf_refine(self, image, mask):
        """Refine pseudo-mask with CRF"""
        import pydensecrf.densecrf as dcrf
        
        h, w = mask.shape
        d = dcrf.DenseCRF2D(w, h, self.num_classes + 1)
        
        # Unary potentials from mask
        U = np.zeros((self.num_classes + 1, h * w), dtype=np.float32)
        for c in range(self.num_classes + 1):
            U[c] = (mask.flatten() == c).astype(np.float32) * (-10) + \
                   (mask.flatten() != c).astype(np.float32) * 10
        d.setUnaryEnergy(U)
        
        # Pairwise potentials
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_np, compat=10)
        
        Q = d.inference(5)
        return np.argmax(Q, axis=0).reshape(h, w)
    
    def train_step(self, images, image_labels):
        """One training step with pseudo-labels"""
        # Generate pseudo-masks
        pseudo_masks = self.generate_pseudo_masks(images, image_labels)
        pseudo_masks = torch.tensor(pseudo_masks)
        
        # Train segmenter
        pred = self.segmenter(images)
        loss = F.cross_entropy(pred, pseudo_masks, ignore_index=255)
        
        return loss

class BoundingBoxSupervision(nn.Module):
    """Segmentation with bounding box supervision"""
    
    def __init__(self, segmenter):
        super().__init__()
        self.segmenter = segmenter
    
    def forward(self, images, boxes, box_labels):
        pred = self.segmenter(images)
        
        loss = 0
        for i, (box_list, label_list) in enumerate(zip(boxes, box_labels)):
            for box, label in zip(box_list, label_list):
                x1, y1, x2, y2 = box.int()
                
                # Loss inside box: should be the class
                inside_pred = pred[i, :, y1:y2, x1:x2]
                inside_loss = self.box_inside_loss(inside_pred, label)
                
                # Loss outside all boxes: should be background
                outside_mask = self.get_outside_mask(pred.shape[2:], box_list)
                outside_pred = pred[i, :, outside_mask]
                outside_loss = F.cross_entropy(
                    outside_pred.T.unsqueeze(0), 
                    torch.zeros(1, dtype=torch.long)
                )
                
                loss += inside_loss + outside_loss
        
        return loss
    
    def box_inside_loss(self, pred, label):
        """GrabCut-style loss: fill box with class"""
        # At least some pixels should be the class
        probs = F.softmax(pred, dim=0)
        class_prob = probs[label].mean()
        return -torch.log(class_prob + 1e-8)

class IterativePseudoLabeling:
    """Self-training with iterative pseudo-label refinement"""
    
    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.threshold = confidence_threshold
    
    def train(self, labeled_data, unlabeled_data, epochs=10, rounds=5):
        for round in range(rounds):
            print(f"Round {round + 1}/{rounds}")
            
            # Generate pseudo-labels for unlabeled data
            pseudo_labels = self.generate_pseudo_labels(unlabeled_data)
            
            # Combine with labeled data
            combined = self.combine_data(labeled_data, unlabeled_data, pseudo_labels)
            
            # Train
            for epoch in range(epochs):
                for images, masks in combined:
                    pred = self.model(images)
                    loss = F.cross_entropy(pred, masks, ignore_index=255)
                    loss.backward()
    
    def generate_pseudo_labels(self, data):
        pseudo = []
        for images in data:
            with torch.no_grad():
                pred = self.model(images)
                probs = F.softmax(pred, dim=1)
                
                max_probs, labels = probs.max(dim=1)
                
                # Only keep high-confidence predictions
                labels[max_probs < self.threshold] = 255  # Ignore
                pseudo.append(labels)
        
        return pseudo
```

**Results Comparison:**

| Supervision | VOC mIoU | Annotation Time |
|-------------|----------|-----------------|
| Fully supervised | 77% | 100× |
| Bounding boxes | 70% | 5× |
| Image labels | 62% | 1× |

**Interview Tip:** CAM-based methods are simple but coarse. Combine with CRF for better boundaries. Self-training with confidence thresholding iteratively improves pseudo-labels. Bounding box supervision with GrabCut achieves near-full supervision quality.

---

## Instance Segmentation

### Question 13
**How does Mask R-CNN's architecture balance object detection and pixel-level segmentation?**

**Answer:**

Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. Detection and mask heads share RoI features but have separate loss functions. The key is decoupling mask and classification—masks are predicted for each class independently.

**Architecture:**

```
Image → Backbone (ResNet+FPN) → RPN → RoI Proposals
                                         ↓
                                    RoIAlign
                                    /   |   \
                              Class  Box  Mask
                              Head   Head Head
                                ↓      ↓     ↓
                              Labels Boxes Masks
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| Backbone (FPN) | Multi-scale feature extraction |
| RPN | Generate object proposals |
| RoIAlign | Extract aligned features per proposal |
| Box head | Predict bounding box + class |
| Mask head | Predict binary mask per class |

**Key Insight: Decoupled Mask and Class**

- Mask head predicts K binary masks (one per class)
- Classification head selects which mask to use
- Avoids competition between classes in mask prediction

**Loss Function:**
$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{mask}$$

Where mask loss is binary cross-entropy on the GT class mask only.

**Python Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import RoIAlign

class MaskRCNN(nn.Module):
    """Simplified Mask R-CNN architecture"""
    
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        # Backbone with FPN
        self.backbone = self._build_backbone(backbone)
        
        # Region Proposal Network
        self.rpn = RegionProposalNetwork(256, 256)
        
        # RoI feature extraction
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, 
                                   sampling_ratio=2)
        self.roi_align_mask = RoIAlign(output_size=(14, 14), spatial_scale=1/16,
                                        sampling_ratio=2)
        
        # Detection heads
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.cls_predictor = nn.Linear(1024, num_classes)
        self.box_predictor = nn.Linear(1024, num_classes * 4)
        
        # Mask head (FCN)
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)  # K masks
        )
    
    def forward(self, images, targets=None):
        # Feature extraction
        features = self.backbone(images)
        
        # RPN proposals
        proposals, rpn_losses = self.rpn(features, targets)
        
        if self.training:
            proposals, matched_targets = self.match_proposals(proposals, targets)
        
        # RoI features for detection
        roi_features = self.roi_align(features['p3'], proposals)
        box_features = self.box_head(roi_features)
        
        class_logits = self.cls_predictor(box_features)
        box_regression = self.box_predictor(box_features)
        
        # RoI features for mask (higher resolution)
        mask_features = self.roi_align_mask(features['p3'], proposals)
        mask_logits = self.mask_head(mask_features)  # [N, K, 28, 28]
        
        if self.training:
            losses = self.compute_losses(
                class_logits, box_regression, mask_logits,
                matched_targets
            )
            return losses
        else:
            return self.post_process(class_logits, box_regression, mask_logits, proposals)
    
    def compute_losses(self, class_logits, box_regression, mask_logits, targets):
        # Classification loss
        cls_loss = F.cross_entropy(class_logits, targets['labels'])
        
        # Box regression loss (only for positive samples)
        box_loss = F.smooth_l1_loss(
            box_regression[targets['labels'] > 0],
            targets['boxes'][targets['labels'] > 0]
        )
        
        # Mask loss (only for positive samples, only GT class mask)
        mask_loss = 0
        for i, label in enumerate(targets['labels']):
            if label > 0:  # Positive sample
                # Get predicted mask for GT class
                pred_mask = mask_logits[i, label]  # [28, 28]
                gt_mask = targets['masks'][i]  # [28, 28]
                
                mask_loss += F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        
        mask_loss = mask_loss / max(1, (targets['labels'] > 0).sum())
        
        return {
            'loss_cls': cls_loss,
            'loss_box': box_loss,
            'loss_mask': mask_loss
        }
    
    def post_process(self, class_logits, box_regression, mask_logits, proposals):
        # Get predicted classes
        scores = F.softmax(class_logits, dim=1)
        max_scores, labels = scores.max(dim=1)
        
        # Select mask for predicted class
        masks = []
        for i, label in enumerate(labels):
            mask = torch.sigmoid(mask_logits[i, label])
            masks.append(mask)
        
        return {
            'boxes': proposals,
            'labels': labels,
            'scores': max_scores,
            'masks': torch.stack(masks)
        }

# Using torchvision (recommended)
def get_maskrcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace heads for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model
```

**Balancing Detection and Segmentation:**

| Aspect | Detection | Segmentation |
|--------|-----------|--------------|
| Features | 7×7 RoI | 14×14 RoI (finer) |
| Loss weight | 1.0 | 1.0 (equal) |
| Output | Class + box | Per-class binary mask |
| Training | Multi-task | Shared backbone |

**Interview Tip:** Mask R-CNN's key insight is predicting class-agnostic masks (one per class) and selecting based on classification. This decoupling prevents mask quality from being affected by classification uncertainty. RoIAlign (vs RoIPool) is crucial for mask quality.

---

### Question 14
**Explain the difference between ROIPool and ROIAlign. Why is ROIAlign crucial for mask quality?**

**Answer:**

ROIPool uses coarse quantization that misaligns features with input pixels, causing spatial distortions. ROIAlign uses bilinear interpolation for exact alignment, eliminating quantization errors. This precision is crucial for pixel-level mask predictions where small misalignments cause visible artifacts.

**The Problem with ROIPool:**

```
Input feature map: 7.3 → 7 (floor)
                          ↓ Quantization error
                   Misaligned features
```

**Quantization Errors in ROIPool:**
1. Rounding proposal coordinates to integers
2. Dividing into bins with floor/ceil
3. Each quantization loses sub-pixel precision

**ROIAlign Solution:**
- Uses bilinear interpolation
- Samples at exact floating-point locations
- No rounding of coordinates

**Visual Comparison:**

```
ROIPool:                          ROIAlign:
[Grid snaps to pixels]            [Grid aligns exactly]
┌───┬───┬───┐                     ┌───┬───┬───┐
│ ● │   │   │  ← Sampled at      │ ⊕ │ ⊕ │ ⊕ │  ← Sampled at
├───┼───┼───┤    integer pos     ├───┼───┼───┤    exact positions
│   │ ● │   │                     │ ⊕ │ ⊕ │ ⊕ │
└───┴───┴───┘                     └───┴───┴───┘
```

**Python Implementation:**
```python
import torch
import torch.nn as nn
from torchvision.ops import roi_align, roi_pool

class RoIPoolVsAlign:
    """Compare ROIPool and ROIAlign"""
    
    @staticmethod
    def roi_pool_manual(features, boxes, output_size):
        """
        ROIPool with quantization
        """
        B, C, H, W = features.shape
        pooled = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Quantization step 1: Round box coordinates
            x1, y1 = int(x1), int(y1)  # ← Precision loss!
            x2, y2 = int(x2), int(y2)
            
            # Extract region
            roi = features[:, :, y1:y2, x1:x2]
            
            # Quantization step 2: Divide into bins
            bin_h = (y2 - y1) // output_size[0]  # ← More precision loss!
            bin_w = (x2 - x1) // output_size[1]
            
            # Max pool each bin
            output = torch.zeros(C, *output_size)
            for i in range(output_size[0]):
                for j in range(output_size[1]):
                    bin_region = roi[:, :, 
                                     i*bin_h:(i+1)*bin_h, 
                                     j*bin_w:(j+1)*bin_w]
                    output[:, i, j] = bin_region.max()
            
            pooled.append(output)
        
        return torch.stack(pooled)
    
    @staticmethod
    def roi_align_manual(features, boxes, output_size, sampling_ratio=2):
        """
        ROIAlign with bilinear interpolation
        """
        B, C, H, W = features.shape
        aligned = []
        
        for box in boxes:
            x1, y1, x2, y2 = box  # Keep as float!
            
            roi_h = y2 - y1
            roi_w = x2 - x1
            
            # Bin size (floating point)
            bin_h = roi_h / output_size[0]
            bin_w = roi_w / output_size[1]
            
            output = torch.zeros(C, *output_size)
            
            for i in range(output_size[0]):
                for j in range(output_size[1]):
                    # Sample points within bin
                    samples = []
                    for si in range(sampling_ratio):
                        for sj in range(sampling_ratio):
                            # Exact sample location (no quantization)
                            y = y1 + bin_h * (i + (si + 0.5) / sampling_ratio)
                            x = x1 + bin_w * (j + (sj + 0.5) / sampling_ratio)
                            
                            # Bilinear interpolation
                            val = bilinear_interpolate(features[0], x, y)
                            samples.append(val)
                    
                    # Average samples
                    output[:, i, j] = torch.stack(samples).mean(dim=0)
            
            aligned.append(output)
        
        return torch.stack(aligned)

def bilinear_interpolate(features, x, y):
    """Bilinear interpolation at (x, y)"""
    C, H, W = features.shape
    
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    
    # Weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    return (wa * features[:, y0, x0] + 
            wb * features[:, y0, x1] + 
            wc * features[:, y1, x0] + 
            wd * features[:, y1, x1])

# Using torchvision (recommended)
def compare_pool_vs_align():
    features = torch.randn(1, 256, 50, 50)
    boxes = torch.tensor([[10.3, 15.7, 30.2, 40.8]])  # Non-integer coords
    
    # ROIPool (with quantization)
    pooled = roi_pool(features, [boxes], output_size=(7, 7), spatial_scale=1.0)
    
    # ROIAlign (exact)
    aligned = roi_align(features, [boxes], output_size=(7, 7), 
                        spatial_scale=1.0, sampling_ratio=2)
    
    print(f"Max difference: {(pooled - aligned).abs().max()}")
    # Significant difference due to quantization
```

**Impact on Mask Quality:**

| Metric | ROIPool | ROIAlign | Improvement |
|--------|---------|----------|-------------|
| Mask AP | 23.6 | 37.1 | +13.5 |
| Mask AP (small) | 8.2 | 17.2 | +9.0 |
| Box AP | 26.9 | 38.2 | +11.3 |

**Why Masks Need Alignment:**

| Task | Tolerance to Misalignment |
|------|---------------------------|
| Classification | High (global pooling) |
| Box regression | Medium |
| Mask prediction | Very Low (pixel-level) |

**Key Parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| output_size | RoI feature resolution | 7 (box), 14 (mask) |
| spatial_scale | Feature map / input ratio | 1/16 or 1/32 |
| sampling_ratio | Points per bin | 2 |

**Interview Tip:** ROIAlign is essential for any pixel-level prediction from region proposals. The 13.5 AP improvement shows how important sub-pixel precision is. Small objects benefit most because quantization error is proportionally larger.

---

---

### Question 15
**What are the trade-offs between two-stage (Mask R-CNN) and single-stage (YOLACT, SOLOv2) instance segmentation?**

**Answer:** _To be filled_

---

### Question 16
**How do you handle overlapping instances in dense object arrangements?**

**Answer:** _To be filled_

---

### Question 17
**Explain panoptic segmentation and how it combines semantic and instance segmentation.**

**Answer:** _To be filled_

---

### Question 18
**What techniques help with segmenting objects with complex or irregular shapes?**

**Answer:** _To be filled_

---

### Question 19
**How do you optimize Mask R-CNN for real-time applications without significant accuracy loss?**

**Answer:** _To be filled_

---

## Vision Transformers (ViT)

### Question 20
**Explain the core innovation of Vision Transformers compared to CNNs. What inductive biases does ViT lack?**

**Answer:** _To be filled_

---

### Question 21
**How are images converted into patch embeddings in ViT? Explain the linear projection layer.**

**Answer:** _To be filled_

---

### Question 22
**What is the role of the [CLS] token and positional encodings in Vision Transformers?**

**Answer:** _To be filled_

---

### Question 23
**Explain the computational complexity of ViT (O(n²) attention) and how it limits resolution scalability.**

**Answer:** _To be filled_

---

### Question 24
**What are the data requirements for training ViT from scratch vs. using pre-trained models?**

**Answer:** _To be filled_

---

### Question 25
**Explain DeiT (Data-efficient Image Transformers) and how knowledge distillation improves ViT training on smaller datasets.**

**Answer:** _To be filled_

---

### Question 26
**Describe Masked Autoencoder (MAE) pre-training for Vision Transformers. How does masking 75% of patches work?**

**Answer:** _To be filled_

---

### Question 27
**How do hybrid architectures combine CNN feature extraction with transformer attention?**

**Answer:** _To be filled_

---

### Question 28
**Explain attention visualization in ViT. How do you interpret attention patterns across layers?**

**Answer:** _To be filled_

---

### Question 29
**What are the architectural variants of ViT (ViT-B, ViT-L, ViT-H) and their trade-offs?**

**Answer:** _To be filled_

---

### Question 30
**How does ViT handle different input image resolutions during fine-tuning vs. pre-training?**

**Answer:** _To be filled_

---

## Swin Transformer

### Question 31
**Explain shifted window partitioning in Swin Transformer. How does it enable cross-window connections?**

**Answer:** _To be filled_

---

### Question 32
**How does Swin Transformer's hierarchical representation differ from ViT's flat structure?**

**Answer:** _To be filled_

---

### Question 33
**Explain the linear complexity (O(n)) of Swin vs. quadratic complexity of ViT. How is this achieved?**

**Answer:** _To be filled_

---

### Question 34
**Describe patch merging layers and how they create multi-scale feature maps.**

**Answer:** _To be filled_

---

### Question 35
**Explain relative positional bias in Swin vs. absolute positional encoding in ViT.**

**Answer:** _To be filled_

---

### Question 36
**How is Swin Transformer used as a backbone for object detection (Swin + FPN) and segmentation (Swin + UPerNet)?**

**Answer:** _To be filled_

---

### Question 37
**Compare Swin-T, Swin-S, Swin-B, and Swin-L configurations. How do you choose for your task?**

**Answer:** _To be filled_

---

### Question 38
**Explain Swin-V2 improvements: log-scaled continuous position bias, residual post-norm, and scaled cosine attention.**

**Answer:** _To be filled_

---

### Question 39
**How does Video Swin Transformer extend the architecture for temporal modeling?**

**Answer:** _To be filled_

---

### Question 40
**Compare Swin Transformer to ConvNeXt. What design principles from Swin were adopted back into CNNs?**

**Answer:** _To be filled_

---

## Segmentation Transformers

### Question 41
**Explain SETR (Segmentation Transformer) and how pure transformers handle dense prediction.**

**Answer:** _To be filled_

---

### Question 42
**Describe SegFormer architecture and its efficient self-attention mechanism for segmentation.**

**Answer:** _To be filled_

---

### Question 43
**How does Mask2Former unify semantic, instance, and panoptic segmentation with a single architecture?**

**Answer:** _To be filled_

---

### Question 44
**Explain SAM (Segment Anything Model) and its promptable segmentation capabilities.**

**Answer:** _To be filled_

---

## Practical Considerations

### Question 45
**How do you handle temporal consistency in video semantic/instance segmentation?**

**Answer:** _To be filled_

---

### Question 46
**What approaches work for domain adaptation in segmentation across different imaging modalities?**

**Answer:** _To be filled_

---

### Question 47
**Explain active learning strategies for efficient mask annotation in segmentation tasks.**

**Answer:** _To be filled_

---

### Question 48
**How do you implement uncertainty quantification in segmentation predictions?**

**Answer:** _To be filled_

---

### Question 49
**What techniques help segment objects in adverse weather or lighting conditions?**

**Answer:** _To be filled_

---

### Question 50
**How do you handle segmentation with limited computational resources or memory on edge devices?**

**Answer:** _To be filled_

---

## Evaluation & Medical Imaging

### Question 51
**Explain IoU, Dice coefficient, and boundary F1-score for segmentation evaluation. When to use each?**

**Answer:** _To be filled_

---

### Question 52
**How do you optimize U-Net architectures for medical image segmentation (3D U-Net, nnU-Net)?**

**Answer:** _To be filled_

---

### Question 53
**What are the challenges of segmentation in specialized domains like satellite imagery or microscopy?**

**Answer:** _To be filled_

---

### Question 54
**Explain federated learning for medical image segmentation across hospitals with privacy constraints.**

**Answer:** _To be filled_

---

### Question 55
**How do you handle noisy or inconsistent annotations in segmentation ground truth?**

**Answer:** _To be filled_

---

## Advanced Topics

### Question 56
**Explain few-shot segmentation in novel semantic categories without retraining.**

**Answer:** _To be filled_

---

### Question 57
**How do you implement knowledge distillation for compressing large segmentation models?**

**Answer:** _To be filled_

---

### Question 58
**What techniques help with explaining segmentation decisions for model interpretability?**

**Answer:** _To be filled_

---

### Question 59
**How do you integrate conditional random fields (CRF) as post-processing for segmentation refinement?**

**Answer:** _To be filled_

---

### Question 60
**Explain multi-task learning that combines segmentation with depth estimation or other vision tasks.**

**Answer:** _To be filled_

---
