# Diffusion Models - Theory Questions

## Question 1
**Explain forward diffusion process and noise scheduling.**

### Theory
The forward diffusion process systematically corrupts data by gradually adding Gaussian noise over T timesteps, transforming the original data distribution q(x₀) into pure noise q(xₜ). This process is mathematically defined as a Markov chain where each step depends only on the previous one.

### Mathematical Foundation
The forward process is defined as:
- q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₓI)
- Where βₜ is the noise schedule controlling how much noise is added at step t

### Code Example
```python
import torch
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (more stable)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def forward_diffusion_sample(x_0, t, noise_schedule):
    """Add noise to data at timestep t"""
    betas = noise_schedule
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Add noise according to schedule
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
```

### Explanation
1. **Noise Schedule**: Controls the rate of noise addition. Linear schedules add noise uniformly, while cosine schedules preserve more signal in early steps
2. **Reparameterization Trick**: Allows direct sampling at any timestep t without sequential computation
3. **Alpha Parameters**: α = 1 - β, α̅ₜ = ∏ᵢ₌₁ᵗ αᵢ (cumulative product)

### Use Cases
- **Image Generation**: DALL-E 2, Midjourney, Stable Diffusion
- **Audio Synthesis**: Music generation, speech synthesis
- **Video Generation**: Temporal consistency in video diffusion
- **3D Shape Generation**: Point cloud and mesh generation

### Best Practices
- **Cosine Schedule**: Better performance than linear for most tasks
- **Timestep Range**: T=1000 is common, but can be reduced with better samplers
- **Noise Clipping**: Prevent numerical instabilities
- **Variance Preservation**: Ensure signal doesn't vanish completely

### Pitfalls
- **Linear Schedule Problems**: Can destroy too much information early
- **Too Aggressive Noise**: Information loss, poor reconstruction
- **Insufficient Steps**: Inadequate noise coverage
- **Schedule Mismatch**: Training/inference schedule inconsistency

### Debugging
```python
def visualize_diffusion_process(x_0, noise_schedule, timesteps_to_show):
    """Debug by visualizing forward process"""
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))
    for i, t in enumerate(timesteps_to_show):
        noisy_image, _ = forward_diffusion_sample(x_0, torch.tensor([t]), noise_schedule)
        axes[i].imshow(noisy_image[0].permute(1, 2, 0))
        axes[i].set_title(f't={t}')
```

### Optimization
- **Memory**: Use gradient checkpointing for long sequences
- **Speed**: Precompute cumulative products
- **Precision**: Use FP16 for large models
- **Batch Size**: Balance between memory and convergence stability

**Answer:** Forward diffusion systematically adds noise to data over T timesteps using a noise schedule (βₜ), transforming clean data into pure noise through a Markov chain process.

---

## Question 2
**Describe reverse diffusion and denoising process.**

### Theory
Reverse diffusion learns to invert the forward noising process by predicting and removing noise at each timestep. Unlike the forward process (which is fixed), the reverse process is parameterized by a neural network that learns the denoising transformation p_θ(x_{t-1} | x_t).

### Mathematical Foundation
The reverse process is defined as:
- p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
- In practice, we predict the noise ε_θ(x_t, t) and compute μ_θ using reparameterization

### Code Example
```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """Simplified U-Net for noise prediction"""
    def __init__(self, channels=3, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + time_emb_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def pos_encoding(self, t, channels):
        """Sinusoidal position encoding for timesteps"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim)
        t = self.time_mlp(t)
        
        # Encode
        x1 = self.encoder(x)
        
        # Add time embedding
        t = t.view(t.shape[0], t.shape[1], 1, 1).expand(-1, -1, x1.shape[2], x1.shape[3])
        x1 = torch.cat([x1, t], dim=1)
        
        # Decode (predict noise)
        return self.decoder(x1)

def reverse_diffusion_step(model, x_t, t, noise_schedule):
    """Single reverse diffusion step"""
    betas = noise_schedule
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # Predict noise
    predicted_noise = model(x_t, t)
    
    # Compute coefficients
    alpha_t = alphas[t]
    alpha_cumprod_t = alphas_cumprod[t]
    alpha_cumprod_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
    
    # Compute mean of reverse step
    beta_t = betas[t]
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod_t)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    
    mean = (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_t
    
    # Add noise for stochastic sampling (except at t=0)
    if t > 0:
        noise = torch.randn_like(x_t)
        variance = (1. - alpha_cumprod_t_prev) / (1. - alpha_cumprod_t) * beta_t
        return mean + torch.sqrt(variance) * noise
    else:
        return mean
```

### Explanation
1. **Neural Network Role**: Learns to predict noise ε added at each step
2. **Reparameterization**: Instead of predicting mean directly, predict noise and compute mean
3. **Timestep Conditioning**: Network needs to know which timestep it's denoising
4. **Variance Schedule**: Can be fixed or learned (DDPM uses fixed)

### Use Cases
- **Image Denoising**: Medical imaging, photography
- **Super Resolution**: Upscaling low-resolution images
- **Image-to-Image Translation**: Style transfer, colorization
- **Conditional Generation**: Text-to-image, class-conditional synthesis

### Best Practices
- **U-Net Architecture**: Skip connections preserve fine details
- **Time Embedding**: Use sinusoidal encoding for continuous time
- **Attention Mechanisms**: Improve global coherence
- **EMA Updates**: Stabilize training with exponential moving averages

### Pitfalls
- **Mode Collapse**: Unlike GANs, less prone but can occur with poor training
- **Slow Sampling**: Requires many steps (1000) for best quality
- **Memory Issues**: Large models need gradient checkpointing
- **Training Instability**: Learning rate and schedule tuning crucial

### Debugging
```python
def debug_reverse_process(model, noise, timesteps, save_intermediates=True):
    """Debug reverse diffusion by saving intermediate steps"""
    x = noise.clone()
    intermediates = []
    
    for i in reversed(range(timesteps)):
        t = torch.full((noise.shape[0],), i, dtype=torch.long)
        x = reverse_diffusion_step(model, x, t, noise_schedule)
        
        if save_intermediates and i % 100 == 0:
            intermediates.append(x.clone())
    
    return x, intermediates

# Visualize denoising trajectory
def plot_denoising_trajectory(intermediates):
    fig, axes = plt.subplots(1, len(intermediates), figsize=(20, 4))
    for i, img in enumerate(intermediates):
        axes[i].imshow(img[0].permute(1, 2, 0).clip(0, 1))
        axes[i].set_title(f'Step {i * 100}')
```

### Optimization
- **Classifier-Free Guidance**: Improves sample quality
- **DDIM Sampling**: Deterministic, faster sampling
- **Progressive Distillation**: Reduce number of steps needed
- **Mixed Precision**: FP16 training for memory efficiency

**Answer:** Reverse diffusion uses a neural network to learn the inverse of the forward noising process, predicting and removing noise at each timestep to gradually recover the original data from pure noise.

---

## Question 3
**Explain DDPM (Denoising Diffusion Probabilistic Models).**

### Theory
DDPM is the foundational framework that established diffusion models as a powerful generative approach. It formulates generation as a learned reverse Markov chain, where a neural network learns to denoise data by reversing a fixed forward diffusion process.

### Mathematical Foundation
DDPM optimizes the variational lower bound:
- L = E_q[D_KL(q(x_T|x_0)||p(x_T)) - log p_θ(x_0|x_1) + Σ D_KL(q(x_{t-1}|x_t,x_0)||p_θ(x_{t-1}|x_t))]
- Simplified to: L_simple = E_{t,x_0,ε}[||ε - ε_θ(√α̅_t x_0 + √(1-α̅_t)ε, t)||²]

### Code Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
import numpy as np

class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, x_t, t):
        """Compute mean and variance of reverse step"""
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Compute mean
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        
        mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Use fixed variance
        variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        
        return mean, variance

    def p_sample(self, x_t, t):
        """Sample from reverse process p(x_{t-1} | x_t)"""
        mean, variance = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        
        # No noise at t=0
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def sample(self, shape, device):
        """Generate samples from noise"""
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x

    def training_loss(self, x_start):
        """DDPM training loss (simplified objective)"""
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward process: add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # MSE loss
        return F.mse_loss(predicted_noise, noise)

# Training loop
def train_ddpm(ddpm, dataloader, epochs=100):
    optimizer = Adam(ddpm.model.parameters(), lr=2e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch, _ in dataloader:
            optimizer.zero_grad()
            
            loss = ddpm.training_loss(batch.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.6f}")
```

### Explanation
1. **Training Objective**: Network learns to predict noise added at each timestep
2. **Sampling Process**: Iteratively denoise starting from pure noise
3. **Fixed Variance**: DDPM uses predetermined variance schedule
4. **Reparameterization**: Key insight that enables stable training

### Use Cases
- **Unconditional Generation**: Generate images from noise
- **Image Restoration**: Denoising, inpainting, super-resolution
- **Medical Imaging**: Generate synthetic medical data
- **Material Science**: Molecular and protein structure generation

### Best Practices
- **Loss Weighting**: Can weight different timesteps differently
- **EMA**: Use exponential moving average for model parameters
- **Data Augmentation**: Standard augmentations help generalization
- **Learning Rate Schedule**: Cosine annealing works well

### Pitfalls
- **Slow Sampling**: Requires 1000+ steps for best quality
- **Memory Requirements**: Large models need careful memory management
- **Mode Coverage**: Ensure training covers full data distribution
- **Numerical Precision**: FP16 can cause instabilities in some cases

### Debugging
```python
def debug_ddpm_training(ddpm, x_batch):
    """Debug DDPM training by analyzing loss at different timesteps"""
    losses_by_timestep = {}
    
    for t_val in range(0, ddpm.timesteps, 100):
        t = torch.full((x_batch.shape[0],), t_val, device=x_batch.device).long()
        noise = torch.randn_like(x_batch)
        
        x_noisy = ddpm.q_sample(x_batch, t, noise)
        predicted_noise = ddpm.model(x_noisy, t)
        
        loss = F.mse_loss(predicted_noise, noise).item()
        losses_by_timestep[t_val] = loss
    
    return losses_by_timestep
```

### Optimization
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Use AMP for memory efficiency
- **Batch Size**: Larger batches generally improve stability
- **Architecture**: U-Net with attention layers performs best

**Answer:** DDPM is the foundational diffusion model that learns to reverse a fixed noise corruption process by predicting added noise, enabling high-quality generation through iterative denoising.

---

## Question 4
**Describe DDIM (Denoising Diffusion Implicit Models) advantages.**

### Theory
DDIM transforms the stochastic DDPM into a deterministic process, enabling much faster sampling with fewer steps while maintaining generation quality. It redefines the forward process as non-Markovian while keeping the same training objective.

### Mathematical Foundation
DDIM uses deterministic sampling: x_{t-1} = √α̅_{t-1} · (x_t - √(1-α̅_t)·ε_θ(x_t,t))/√α̅_t + √(1-α̅_{t-1})·ε_θ(x_t,t)

### Code Example
```python
def ddim_sample(self, x_t, t, t_prev, eta=0.0):
    """DDIM sampling step with optional stochasticity (eta)"""
    predicted_noise = self.model(x_t, t)
    
    alpha_t = self.alphas_cumprod[t]
    alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
    
    # Predict x_0
    pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    
    # Direction to x_t
    dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
    
    # Add stochasticity
    if eta > 0:
        variance = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
        noise = torch.randn_like(x_t)
        dir_xt += variance * noise
    
    return torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

def fast_sample(self, shape, steps=50):
    """Fast sampling with fewer steps"""
    timesteps = torch.linspace(self.timesteps-1, 0, steps).long()
    x = torch.randn(shape)
    
    for i, t in enumerate(timesteps[:-1]):
        t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
        x = self.ddim_sample(x, t, t_prev)
    
    return x
```

### Use Cases & Best Practices
- **Real-time Generation**: 10-50 steps vs 1000 for DDPM
- **Image Editing**: Deterministic reconstruction enables editing
- **Style Transfer**: Consistent transformations
- **Latent Interpolation**: Smooth transitions between samples

### Optimization
- **Step Selection**: Non-uniform spacing can improve quality
- **Eta Parameter**: Controls stochasticity (0=deterministic, 1=DDPM-like)

**Answer:** DDIM provides deterministic, fast sampling by reframing diffusion as a non-Markovian process, reducing steps from 1000 to 10-50 while maintaining quality.

---

## Question 5
**Explain U-Net architecture in diffusion models.**

### Theory
U-Net serves as the backbone for most diffusion models, providing an encoder-decoder structure with skip connections that preserve fine-grained details while enabling multi-scale feature processing. The architecture includes timestep and condition embeddings.

### Code Example
```python
class DiffusionUNet(nn.Module):
    def __init__(self, channels=3, model_channels=128, num_heads=8):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(), nn.Linear(model_channels * 4, model_channels * 4))
        
        # Encoder blocks with attention
        self.encoder_blocks = nn.ModuleList([
            ResBlock(channels, model_channels, time_emb_dim=model_channels * 4),
            AttentionBlock(model_channels, num_heads),
            ResBlock(model_channels, model_channels * 2, time_emb_dim=model_channels * 4),
            AttentionBlock(model_channels * 2, num_heads)
        ])
        
        # Decoder with skip connections
        self.decoder_blocks = nn.ModuleList([
            ResBlock(model_channels * 4, model_channels * 2, time_emb_dim=model_channels * 4),
            AttentionBlock(model_channels * 2, num_heads),
            ResBlock(model_channels * 3, model_channels, time_emb_dim=model_channels * 4),
        ])

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        return self.conv2(self.norm2(h))
```

### Key Components
- **Skip Connections**: Preserve fine details lost in downsampling
- **Multi-Scale Processing**: Different resolution features at each level
- **Attention Layers**: Global context for coherent generation
- **Time Embedding**: Sinusoidal encoding of diffusion timestep

**Answer:** U-Net provides the encoder-decoder architecture with skip connections essential for diffusion models, enabling multi-scale processing while preserving fine details through direct feature concatenation.

---

## Question 6
**Describe classifier guidance vs. classifier-free guidance.**

### Theory
Guidance techniques steer diffusion models toward desired outputs. Classifier guidance uses external classifiers, while classifier-free guidance trains a single model to handle both conditional and unconditional generation.

### Classifier Guidance
```python
def classifier_guided_step(x_t, t, class_label, classifier, guidance_scale=1.0):
    # Standard denoising prediction
    noise_pred = diffusion_model(x_t, t)
    
    # Classifier gradient
    x_t.requires_grad_(True)
    logits = classifier(x_t, t)
    class_loss = F.cross_entropy(logits, class_label)
    grad = torch.autograd.grad(class_loss, x_t)[0]
    
    # Apply guidance
    guided_noise = noise_pred - guidance_scale * grad
    return guided_noise
```

### Classifier-Free Guidance
```python
def classifier_free_guidance(x_t, t, condition, guidance_scale=7.5):
    # Conditional prediction
    noise_pred_cond = diffusion_model(x_t, t, condition)
    
    # Unconditional prediction (null condition)
    noise_pred_uncond = diffusion_model(x_t, t, null_condition)
    
    # Combine predictions
    guided_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return guided_noise
```

### Advantages
- **Classifier-Free**: No separate classifier training, better quality
- **Flexible Control**: Guidance scale controls conditioning strength
- **Training Efficiency**: Single model handles both cases

**Answer:** Classifier-free guidance trains one model for both conditional and unconditional generation, providing superior control and quality compared to external classifier guidance approaches.

---

## Question 7
**Explain latent diffusion models (Stable Diffusion).**

### Theory
Latent diffusion operates in a compressed latent space rather than pixel space, dramatically reducing computational requirements while maintaining quality. VAE encoder/decoder handles pixel↔latent conversion.

### Code Example
```python
class LatentDiffusion(nn.Module):
    def __init__(self, unet, vae, text_encoder):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * 0.18215
    
    def decode(self, z):
        return self.vae.decode(z / 0.18215).sample
    
    def forward(self, x, text_prompts, timesteps):
        # Encode to latent space
        z = self.encode(x)
        
        # Text encoding
        text_embeddings = self.text_encoder(text_prompts)
        
        # Diffusion in latent space
        noise = torch.randn_like(z)
        z_noisy = self.q_sample(z, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.unet(z_noisy, timesteps, text_embeddings)
        
        return F.mse_loss(predicted_noise, noise)
```

### Advantages
- **8x Compression**: 512x512 → 64x64 latent space
- **Speed**: 10-100x faster than pixel-space diffusion
- **Memory Efficiency**: Fits larger models on consumer GPUs
- **Quality**: Minimal loss due to high-quality VAE

**Answer:** Latent diffusion performs the diffusion process in a compressed VAE latent space, providing 8x compression and dramatically faster training/inference while maintaining generation quality.

---

## Question 8
**Describe VAE encoder-decoder in latent space.**

### Theory
VAE provides the latent space foundation for Stable Diffusion, compressing images into a lower-dimensional representation while preserving semantic content for reconstruction.

### Code Example
```python
class VAE(nn.Module):
    def __init__(self, latent_dim=4, channels=3):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim * 2)  # mu + logvar
        self.decoder = Decoder(latent_dim, channels)
        self.quant_conv = nn.Conv2d(latent_dim * 2, latent_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        return DiagonalGaussianDistribution(mu, logvar)
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
```

### Key Features
- **8x Downsampling**: 512×512 → 64×64 latent
- **KL Regularization**: Ensures smooth latent space
- **Perceptual Loss**: VGG-based reconstruction quality
- **High Fidelity**: Minimal information loss

**Answer:** VAE encoder compresses images to 64×64×4 latents with 8× spatial reduction, enabling efficient diffusion while the decoder reconstructs high-quality images from the latent representations.

---

## Question 9
**Explain CLIP text encoding for conditioning.**

### Theory
CLIP provides semantic text understanding for text-to-image generation by learning joint embeddings of text and images in a shared semantic space.

### Code Example
```python
class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=512, num_heads=8, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(77, embed_dim))
        
        self.transformer = nn.ModuleList([
            CLIPEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
    
    def forward(self, text_tokens):
        x = self.token_embedding(text_tokens) + self.positional_embedding
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.ln_final(x)
        return x  # [batch_size, seq_len, embed_dim]

def text_to_conditioning(text_prompt, clip_model, tokenizer):
    tokens = tokenizer.encode(text_prompt, max_length=77, padding=True)
    text_embeddings = clip_model(tokens)
    return text_embeddings
```

### Integration with Diffusion
- **Cross-Attention**: Text embeddings condition U-Net via attention layers
- **Classifier-Free Guidance**: Empty prompt enables unconditional generation
- **Prompt Engineering**: Text structure affects generation quality

**Answer:** CLIP encodes text prompts into semantic embeddings that condition diffusion models through cross-attention, enabling precise text-to-image generation with shared text-image understanding.

---

## Question 10
**Compare diffusion vs. GANs for image generation.**

### Theory
Diffusion models and GANs represent different paradigms for generative modeling, with distinct trade-offs in quality, diversity, training stability, and computational requirements.

### Comparison Table
| Aspect | Diffusion Models | GANs |
|--------|------------------|------|
| **Training Stability** | Highly stable | Prone to mode collapse |
| **Sample Quality** | High fidelity | Variable quality |
| **Sample Diversity** | Excellent coverage | Mode collapse issues |
| **Speed** | Slow (many steps) | Fast (single forward) |
| **Memory** | High during training | Moderate |
| **Control** | Excellent (guidance) | Limited controllability |

### Code Comparison
```python
# GAN: Single forward pass
def gan_generate(generator, noise):
    return generator(noise)  # Instant generation

# Diffusion: Iterative process
def diffusion_generate(model, noise, steps=1000):
    x = noise
    for t in reversed(range(steps)):
        x = denoise_step(model, x, t)
    return x  # 1000× slower
```

### Use Cases
- **GANs**: Real-time applications, style transfer, face generation
- **Diffusion**: High-quality art, text-to-image, when quality > speed

**Answer:** Diffusion models provide superior sample quality and diversity with stable training, while GANs offer faster generation but suffer from mode collapse and training instability issues.

---

## Question 11
**Describe noise prediction vs. score matching.**

### Theory
Two main formulations exist: noise prediction (ε-parameterization) used in DDPM, and score matching (predicting ∇_x log p(x)) used in score-based models. Both are mathematically equivalent but have different training dynamics.

### Code Comparison
```python
# Noise prediction (DDPM)
def noise_prediction_loss(model, x0, t):
    noise = torch.randn_like(x0)
    x_noisy = add_noise(x0, t, noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)

# Score matching
def score_matching_loss(model, x0, t):
    noise = torch.randn_like(x0)
    x_noisy = add_noise(x0, t, noise)
    predicted_score = model(x_noisy, t)
    true_score = -noise / noise_level(t)
    return F.mse_loss(predicted_score, true_score)
```

**Answer:** Noise prediction learns ε_θ(x_t,t) while score matching learns ∇log p(x_t). Both formulations are equivalent but noise prediction is more numerically stable and widely adopted.

---

## Question 12
**Explain timestep embedding and sinusoidal encoding.**

### Theory
Diffusion models need to know which timestep they're processing. Sinusoidal encoding provides smooth, learnable representations that help the network understand temporal position in the diffusion process.

### Code Example
```python
def sinusoidal_timestep_embedding(timesteps, dim):
    """Sinusoidal timestep encoding like Transformer positional encoding"""
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    if dim % 2 == 1:  # Zero pad for odd dimensions
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
    
    def forward(self, t):
        t_emb = sinusoidal_timestep_embedding(t, self.dim)
        return self.mlp(t_emb)
```

**Answer:** Sinusoidal timestep embedding encodes diffusion timesteps into continuous representations, enabling the network to understand its position in the denoising process and adjust predictions accordingly.

---

## Question 13
**Describe attention mechanisms in diffusion U-Net.**

### Theory
Attention layers in diffusion U-Nets provide global context and enable conditioning on text or other modalities through cross-attention mechanisms.

### Code Example
```python
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, context_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Self-attention
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # Cross-attention (for conditioning)
        if context_dim:
            self.to_k_cross = nn.Linear(context_dim, dim, bias=False)
            self.to_v_cross = nn.Linear(context_dim, dim, bias=False)
        
        self.proj_out = nn.Linear(dim, dim)
    
    def forward(self, x, context=None):
        b, h, w, c = x.shape
        x_flat = x.view(b, h*w, c)
        
        # Self-attention
        qkv = self.to_qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Cross-attention if context provided
        if context is not None:
            k_cross = self.to_k_cross(context)
            v_cross = self.to_v_cross(context)
            k = torch.cat([k, k_cross], dim=2)
            v = torch.cat([v, v_cross], dim=2)
        
        # Attention computation
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj_out(out).view(b, h, w, c)
```

**Answer:** Attention in diffusion U-Nets provides global spatial relationships via self-attention and enables conditioning through cross-attention, crucial for text-to-image and other conditional generation tasks.

---

## Question 14
**Explain sampling strategies (DDPM, DDIM, DPM-Solver).**

### Theory
Different sampling strategies trade off between speed and quality. DDPM is slow but high-quality, DDIM is deterministic and faster, while DPM-Solver provides advanced ODE solving.

```python
# DDPM: Stochastic, 1000 steps
def ddpm_sample(model, x_t, t):
    mean, variance = compute_posterior(model, x_t, t)
    noise = torch.randn_like(x_t) if t > 0 else 0
    return mean + torch.sqrt(variance) * noise

# DDIM: Deterministic, 50 steps  
def ddim_sample(model, x_t, t, t_prev, eta=0.0):
    predicted_noise = model(x_t, t)
    alpha_t, alpha_prev = alphas[t], alphas[t_prev]
    pred_x0 = (x_t - sqrt(1-alpha_t) * predicted_noise) / sqrt(alpha_t)
    return sqrt(alpha_prev) * pred_x0 + sqrt(1-alpha_prev) * predicted_noise

# DPM-Solver: Advanced ODE solver
def dpm_solver_sample(model, x_t, t, order=2):
    # Multi-step ODE solver with higher order approximation
    return dpm_solver_step(model, x_t, t, order)
```

**Answer:** DDPM uses stochastic sampling over 1000 steps, DDIM provides deterministic fast sampling in 10-50 steps, while DPM-Solver offers advanced ODE solving for optimal speed-quality trade-offs.

---

## Question 15
**Compare deterministic vs. stochastic sampling.**

### Theory
Stochastic sampling (DDPM) adds random noise at each step, while deterministic sampling (DDIM) follows a fixed trajectory. This affects reproducibility, diversity, and editing capabilities.

```python
def stochastic_step(x_t, predicted_noise, t, variance):
    mean = compute_mean(x_t, predicted_noise, t)
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(variance) * noise  # Random

def deterministic_step(x_t, predicted_noise, t):
    mean = compute_mean(x_t, predicted_noise, t)
    return mean  # No randomness
```

**Comparison:**
- **Stochastic**: Higher diversity, non-reproducible, harder to edit
- **Deterministic**: Exact reproduction, consistent editing, less diversity

**Answer:** Stochastic sampling adds randomness for diversity but reduces reproducibility, while deterministic sampling enables exact reconstruction and consistent editing at the cost of sample variety.

---

## Question 16
**Describe inpainting with diffusion models.**

### Theory
Inpainting fills masked regions by conditioning the diffusion process on unmasked content, either through model conditioning or iterative constraint application.

```python
def diffusion_inpainting(model, image, mask, steps=50):
    """Inpaint masked regions using diffusion"""
    x = torch.randn_like(image)
    known_pixels = image * (1 - mask)
    
    for t in reversed(range(steps)):
        # Denoise step
        x = denoise_step(model, x, t)
        
        # Replace known pixels
        x = x * mask + known_pixels * (1 - mask)
        
        # Add noise to known pixels for next iteration
        if t > 0:
            noise_level = get_noise_level(t-1)
            x = x + noise_level * torch.randn_like(x) * (1 - mask)
    
    return x

class InpaintingUNet(nn.Module):
    """U-Net that takes image + mask as input"""
    def forward(self, x, mask, t):
        # Concatenate image and mask as input channels
        input_tensor = torch.cat([x, mask], dim=1)
        return self.unet(input_tensor, t)
```

**Answer:** Diffusion inpainting fills masked regions by conditioning on known pixels, either through architectural modifications that include masks or by iteratively constraining known regions during sampling.

---

## Question 17
**Explain outpainting and image extension.**

### Theory
Outpainting extends images beyond their borders using diffusion models, creating coherent content that continues the existing image's style and context.

```python
def outpaint_image(model, image, extension_size):
    """Extend image boundaries with diffusion outpainting"""
    h, w = image.shape[-2:]
    extended_h, extended_w = h + 2*extension_size, w + 2*extension_size
    
    # Create extended canvas
    extended_image = torch.zeros(1, 3, extended_h, extended_w)
    extended_mask = torch.ones(1, 1, extended_h, extended_w)
    
    # Place original image in center
    start_h, start_w = extension_size, extension_size
    extended_image[:, :, start_h:start_h+h, start_w:start_w+w] = image
    extended_mask[:, :, start_h:start_h+h, start_w:start_w+w] = 0
    
    # Inpaint the extended regions
    return diffusion_inpainting(model, extended_image, extended_mask)
```

**Answer:** Outpainting uses diffusion inpainting techniques to extend images beyond their original boundaries, generating coherent content that seamlessly continues the existing image's visual context.

---

## Question 18
**Describe ControlNet for spatial conditioning.**

### Theory
ControlNet provides precise spatial control over diffusion models by conditioning on structural information like edges, depth maps, or poses while preserving the original model's capabilities.

```python
class ControlNet(nn.Module):
    def __init__(self, unet, control_channels=3):
        super().__init__()
        self.control_net = copy.deepcopy(unet.encoder)  # Copy encoder
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(feat_dim, feat_dim, 1) for feat_dim in encoder_dims
        ])
        self.control_conv = nn.Conv2d(control_channels, 64, 3, padding=1)
        
        # Initialize zero convolutions to zero
        for zero_conv in self.zero_convs:
            nn.init.zeros_(zero_conv.weight)
            nn.init.zeros_(zero_conv.bias)
    
    def forward(self, x, control_input, t):
        # Process control input
        control_features = self.control_conv(control_input)
        
        # Extract features through control network
        control_features = []
        h = control_features
        for i, (block, zero_conv) in enumerate(zip(self.control_net, self.zero_convs)):
            h = block(h, t)
            control_features.append(zero_conv(h))
        
        return control_features

# Usage with original U-Net
def controlled_diffusion_step(unet, controlnet, x, control, t):
    control_features = controlnet(x, control, t)
    return unet(x, t, additional_features=control_features)
```

**Answer:** ControlNet adds spatial conditioning to diffusion models by training an auxiliary network that processes structural inputs (edges, depth) and injects features into the original U-Net without disrupting pre-trained weights.

---

## Question 19
**Explain IP-Adapter for image prompting.**

### Theory
IP-Adapter enables image-based conditioning in diffusion models by encoding reference images into features that guide generation, allowing image-to-image style transfer and content control.

```python
class IPAdapter(nn.Module):
    def __init__(self, unet, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder  # CLIP image encoder
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(dim) for dim in unet.attention_dims
        ])
    
    def forward(self, x, image_prompt, t):
        # Encode reference image
        image_features = self.image_encoder(image_prompt)
        
        # Add to cross-attention in U-Net
        return self.unet_with_image_attention(x, t, image_features)
```

**Answer:** IP-Adapter enables image prompting by encoding reference images through CLIP and injecting these features into U-Net cross-attention layers, allowing image-guided generation alongside text prompts.

---

## Question 20
**Compare different noise schedules (linear, cosine).**

### Theory
Noise schedules control how quickly information is destroyed during forward diffusion. Linear schedules destroy information too quickly, while cosine schedules preserve more structure.

```python
def linear_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

**Comparison:**
- **Linear**: Simple, destroys structure quickly, harder training
- **Cosine**: Preserves structure longer, better training dynamics, SOTA results

**Answer:** Cosine noise schedules outperform linear schedules by preserving image structure longer during forward diffusion, leading to better training dynamics and generation quality.

---

## Question 21
**Describe multi-scale diffusion architectures.**

### Theory
Multi-scale approaches process different resolution levels simultaneously or sequentially, improving both efficiency and quality through hierarchical generation.

```python
class MultiScaleDiffusion(nn.Module):
    def __init__(self):
        self.low_res_model = UNet(resolution=64)
        self.high_res_model = UNet(resolution=256)
        self.upsampler = nn.ConvTranspose2d(4, 4, 4, stride=4)
    
    def forward(self, x_low, x_high, t):
        # Low resolution diffusion
        low_pred = self.low_res_model(x_low, t)
        
        # Upsample and condition high-res
        low_upsampled = self.upsampler(x_low)
        high_input = torch.cat([x_high, low_upsampled], dim=1)
        high_pred = self.high_res_model(high_input, t)
        
        return low_pred, high_pred
```

**Answer:** Multi-scale diffusion processes multiple resolutions simultaneously, with lower resolution guiding higher resolution generation, improving efficiency and coherence in high-resolution synthesis.

---

## Question 22
**Explain video diffusion models.**

### Theory
Video diffusion extends image diffusion to temporal sequences, requiring temporal consistency mechanisms and 3D convolutions or attention across time dimensions.

```python
class VideoDiffusionModel(nn.Module):
    def __init__(self, channels=3, frames=16):
        super().__init__()
        self.temporal_attention = TemporalAttention()
        self.spatial_conv = nn.Conv3d(channels, 64, (1, 3, 3), padding=(0, 1, 1))
        self.temporal_conv = nn.Conv3d(64, 64, (3, 1, 1), padding=(1, 0, 0))
    
    def forward(self, x, t):
        # x shape: [batch, channels, frames, height, width]
        
        # Spatial processing
        h = self.spatial_conv(x)
        
        # Temporal processing
        h = self.temporal_conv(h)
        
        # Temporal attention for long-range dependencies
        h = self.temporal_attention(h)
        
        return h
```

**Key Challenges:**
- **Temporal Consistency**: Preventing flickering between frames
- **Memory Requirements**: Processing multiple frames simultaneously
- **Motion Modeling**: Understanding object movement and physics

**Answer:** Video diffusion extends image models with temporal dimensions using 3D convolutions and temporal attention to generate coherent video sequences while maintaining frame-to-frame consistency.

---

## Question 23
**Describe 3D diffusion for shape generation.**

### Theory
3D diffusion operates on point clouds, voxels, or implicit functions to generate 3D shapes, requiring specialized architectures for 3D data processing.

```python
class PointCloudDiffusion(nn.Module):
    def __init__(self, num_points=2048, point_dim=3):
        super().__init__()
        self.point_conv = nn.Conv1d(point_dim, 128, 1)
        self.transformer_blocks = nn.ModuleList([
            PointTransformerBlock(128) for _ in range(6)
        ])
        self.output_conv = nn.Conv1d(128, point_dim, 1)
    
    def forward(self, points, t):
        # points: [batch, point_dim, num_points]
        h = self.point_conv(points)
        
        for block in self.transformer_blocks:
            h = block(h, t)
        
        return self.output_conv(h)

class VoxelDiffusion(nn.Module):
    def __init__(self, voxel_size=64):
        super().__init__()
        self.conv3d = nn.Conv3d(1, 64, 3, padding=1)
        self.unet3d = UNet3D()
    
    def forward(self, voxels, t):
        # voxels: [batch, 1, size, size, size]
        return self.unet3d(voxels, t)
```

**Answer:** 3D diffusion generates 3D shapes using specialized architectures for point clouds (PointNet/Transformers) or voxels (3D U-Nets), enabling applications in CAD, gaming, and robotics.

---

## Question 24
**Explain audio diffusion models.**

### Theory
Audio diffusion generates waveforms or spectrograms using 1D/2D diffusion models, often working in frequency domain for better perceptual quality.

```python
class AudioDiffusion(nn.Module):
    def __init__(self, sample_rate=22050, n_fft=1024):
        super().__init__()
        self.n_fft = n_fft
        self.unet_1d = UNet1D(channels=1)  # For raw audio
        self.unet_2d = UNet2D(channels=1)  # For spectrograms
    
    def forward(self, audio, t, use_spectrogram=True):
        if use_spectrogram:
            # Convert to spectrogram
            spec = torch.stft(audio.squeeze(), n_fft=self.n_fft)
            spec_mag = torch.abs(spec).unsqueeze(1)
            return self.unet_2d(spec_mag, t)
        else:
            return self.unet_1d(audio, t)
    
    def spectrogram_to_audio(self, spec_pred, phase):
        # Convert back to time domain
        complex_spec = spec_pred * torch.exp(1j * phase)
        return torch.istft(complex_spec, n_fft=self.n_fft)
```

**Answer:** Audio diffusion generates sound by processing waveforms (1D diffusion) or spectrograms (2D diffusion), often using frequency domain representations for better perceptual quality and training stability.

---

## Question 25
**Compare continuous vs. discrete timesteps.**

### Theory
Continuous timesteps treat diffusion as a continuous stochastic differential equation (SDE), while discrete timesteps use fixed intervals. Continuous formulations enable advanced ODE/SDE solvers.

```python
# Discrete timesteps (DDPM)
t = torch.randint(0, 1000, (batch_size,))

# Continuous timesteps  
t = torch.rand(batch_size,) * T  # T = max time

class ContinuousDiffusion(nn.Module):
    def forward(self, x, t):
        # t can be any float in [0, T]
        return self.score_network(x, t)
```

**Answer:** Continuous timesteps enable SDE formulations and advanced solvers (DPM-Solver++), while discrete timesteps are simpler but less flexible for sampling optimization.

---

## Question 26
**Describe training objective and loss functions.**

### Theory
The core training loss is MSE between predicted and actual noise, derived from the variational lower bound of the log-likelihood.

```python
def ddpm_loss(model, x0, noise_schedule):
    t = torch.randint(0, len(noise_schedule), (x0.shape[0],))
    noise = torch.randn_like(x0)
    x_noisy = add_noise(x0, t, noise, noise_schedule)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)

# Alternative: v-parameterization
def v_loss(model, x0, noise_schedule):
    t = torch.randint(0, len(noise_schedule), (x0.shape[0],))
    noise = torch.randn_like(x0)
    v_target = sqrt_alphas_cumprod[t] * noise - sqrt_one_minus_alphas_cumprod[t] * x0
    x_noisy = add_noise(x0, t, noise, noise_schedule)
    predicted_v = model(x_noisy, t)
    return F.mse_loss(predicted_v, v_target)
```

**Answer:** DDPM uses simple MSE loss between predicted and actual noise, derived from variational lower bound. Alternative parameterizations (v-param, x0-pred) can improve training dynamics.

---

## Question 27
**Explain gradient accumulation strategies.**

### Theory
Large diffusion models require gradient accumulation due to memory constraints, accumulating gradients over multiple micro-batches before optimizer steps.

```python
def train_with_gradient_accumulation(model, dataloader, accumulation_steps=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch_idx, batch in enumerate(dataloader):
        loss = ddpm_loss(model, batch) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

**Answer:** Gradient accumulation enables training large diffusion models by accumulating gradients over multiple micro-batches, effectively increasing batch size within memory constraints.

---

## Question 28
**Describe mixed precision training benefits.**

### Theory
Mixed precision (FP16/BF16) reduces memory usage and increases training speed while maintaining model quality through automatic loss scaling.

```python
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training(model, dataloader):
    scaler = GradScaler()
    
    for batch in dataloader:
        with autocast():
            loss = ddpm_loss(model, batch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Answer:** Mixed precision training reduces memory usage by ~50% and increases speed by ~30-50% using FP16 computations with FP32 master weights and gradient scaling.

---

## Question 29
**Explain EMA (Exponential Moving Average) in training.**

### Theory
EMA maintains a smoothed version of model parameters that often performs better than the final trained weights, providing more stable generation.

```python
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1-self.decay)
```

**Answer:** EMA maintains exponentially weighted parameter averages during training, typically providing more stable and higher-quality generation than final model weights.

---

## Question 30
**Compare memory requirements during training/inference.**

### Theory
Training requires storing activations for backprop while inference only needs forward pass, leading to different memory patterns and optimization strategies.

```python
# Training memory usage
def training_memory_analysis():
    memory_components = {
        'model_parameters': '1.4GB (Stable Diffusion)',
        'optimizer_states': '2.8GB (Adam with momentum)',
        'gradients': '1.4GB (same as parameters)',
        'activations': '4-8GB (depends on batch size)',
        'total_training': '~10-14GB'
    }
    
    optimization_strategies = [
        'gradient_checkpointing',  # Trade compute for memory
        'mixed_precision',         # FP16 reduces by ~50%
        'gradient_accumulation',   # Smaller micro-batches
        'cpu_offloading'          # Move unused params to CPU
    ]

# Inference memory usage  
def inference_memory_analysis():
    return {
        'model_parameters': '1.4GB',
        'single_forward_pass': '1-2GB',
        'total_inference': '~3GB'
    }
```

**Answer:** Training requires ~10-14GB (parameters + gradients + activations + optimizer states) while inference needs only ~3GB, enabling various memory optimization strategies.

---

## Question 31
**Describe progressive distillation techniques.**

### Theory
Progressive distillation reduces sampling steps by training student models to match multiple teacher steps in single forward passes.

```python
class ProgressiveDistillation:
    def __init__(self, teacher_model):
        self.teacher = teacher_model
        self.student = copy.deepcopy(teacher_model)
    
    def distill_step(self, x, t_start, num_teacher_steps=2):
        # Teacher takes multiple steps
        x_teacher = x.clone()
        for _ in range(num_teacher_steps):
            x_teacher = self.teacher.denoise_step(x_teacher, t_start)
            t_start -= 1
        
        # Student learns to match in one step
        x_student = self.student.denoise_step(x, t_start + num_teacher_steps)
        return F.mse_loss(x_student, x_teacher.detach())
```

**Answer:** Progressive distillation trains student models to match multiple teacher denoising steps in single forward passes, reducing sampling from 1000 to 4-8 steps with minimal quality loss.

---

## Question 32
**Explain consistency models for fast sampling.**

### Theory
Consistency models learn to map any point on a diffusion trajectory directly to its endpoint, enabling single-step generation.

```python
class ConsistencyModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def consistency_loss(self, x, t1, t2):
        """Consistency loss: f(x_t1) should equal f(x_t2) for same trajectory"""
        # Get predictions at different timesteps
        pred_t1 = self.base_model(x, t1)
        pred_t2 = self.base_model(x, t2)
        
        # They should be consistent (map to same endpoint)
        return F.mse_loss(pred_t1, pred_t2.detach())
    
    def single_step_generate(self, noise):
        """Generate in single step"""
        return self.base_model(noise, torch.zeros_like(noise))
```

**Answer:** Consistency models learn trajectory mappings that enable single-step generation by ensuring consistent endpoint predictions regardless of starting timestep along diffusion trajectories.

---

## Question 33
**Describe edit-friendly inversions (DDIM inversion).**

### Theory
DDIM inversion enables editing real images by finding latent codes that reconstruct the original image, then modifying text prompts for controlled editing.

```python
def ddim_invert(model, image, num_steps=50):
    """Invert real image to noise via DDIM"""
    x = image.clone()
    timesteps = torch.linspace(0, 999, num_steps).long()
    
    for i, t in enumerate(timesteps):
        # Reverse the DDIM process
        predicted_noise = model(x, t, uncond=True)
        
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[timesteps[i-1]] if i > 0 else 1.0
        
        # Invert DDIM step
        pred_x0 = (x - sqrt(1-alpha_t) * predicted_noise) / sqrt(alpha_t)
        x = sqrt(alpha_prev) * pred_x0 + sqrt(1-alpha_prev) * predicted_noise
    
    return x  # Inverted noise

def edit_inverted_image(model, inverted_noise, original_prompt, edit_prompt, steps=50):
    """Edit image using inverted noise and new prompt"""
    return ddim_sample(model, inverted_noise, edit_prompt, steps)
```

**Answer:** DDIM inversion finds noise codes that reconstruct real images deterministically, enabling semantic editing by sampling with modified prompts while preserving image structure.

---

## Question 34
**Explain self-attention guidance.**

### Theory
Self-attention guidance improves generation quality by amplifying or suppressing attention patterns, enhancing coherence without additional training.

```python
def self_attention_guidance(model, x, t, guidance_scale=1.5):
    """Apply guidance based on self-attention patterns"""
    
    # Standard forward pass
    with torch.enable_grad():
        x.requires_grad_(True)
        noise_pred = model(x, t)
        
        # Extract self-attention maps
        attention_maps = model.get_attention_maps()
        
        # Compute guidance signal
        attention_loss = compute_attention_guidance_loss(attention_maps)
        
        # Get gradients w.r.t input
        guidance_grad = torch.autograd.grad(attention_loss, x)[0]
        
        # Apply guidance
        guided_noise = noise_pred + guidance_scale * guidance_grad
    
    return guided_noise

def compute_attention_guidance_loss(attention_maps):
    """Loss that encourages better attention patterns"""
    loss = 0
    for attn_map in attention_maps:
        # Encourage attention diversity
        loss -= torch.entropy(attn_map.mean(dim=1))
        
        # Discourage attention collapse
        loss += torch.var(attn_map, dim=-1).mean()
    
    return loss
```

**Answer:** Self-attention guidance improves generation by modifying attention patterns during sampling, enhancing coherence and detail without requiring additional training or models.

---

## Question 35
**Describe cascading diffusion models.**

### Theory
Cascading approaches use multiple models at different resolutions, with lower resolution models guiding higher resolution generation for efficiency and quality.

```python
class CascadedDiffusion:
    def __init__(self):
        self.base_model = DiffusionModel(resolution=64)    # 64x64
        self.sr_model_1 = SuperResDiffusion(64, 256)       # 64→256
        self.sr_model_2 = SuperResDiffusion(256, 1024)     # 256→1024
    
    def generate(self, prompt, steps=50):
        # Stage 1: Generate base resolution
        base_image = self.base_model.sample(prompt, size=(64, 64), steps=steps)
        
        # Stage 2: Super-resolve to 256x256
        mid_image = self.sr_model_1.sample(
            prompt, 
            conditioning_image=base_image,
            size=(256, 256), 
            steps=steps//2
        )
        
        # Stage 3: Super-resolve to 1024x1024
        final_image = self.sr_model_2.sample(
            prompt,
            conditioning_image=mid_image, 
            size=(1024, 1024),
            steps=steps//4
        )
        
        return final_image
```

**Answer:** Cascading diffusion uses multiple models at increasing resolutions, with each stage conditioning on the previous lower-resolution output, enabling efficient high-resolution generation.

---

## Question 36
**Discuss spectrogram diffusion for audio generation.**

### Theory
Audio diffusion often works in frequency domain using spectrograms, which provide better perceptual quality and training stability than raw waveforms.

```python
class SpectrogramDiffusion(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.unet_2d = UNet2D(channels=1)  # For magnitude spectrogram
        
    def audio_to_spectrogram(self, audio):
        # STFT to frequency domain
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        return magnitude.unsqueeze(1), phase
    
    def spectrogram_to_audio(self, magnitude, phase):
        # Reconstruct complex spectrogram
        complex_stft = magnitude.squeeze(1) * torch.exp(1j * phase)
        # ISTFT back to time domain
        return torch.istft(complex_stft, n_fft=self.n_fft, hop_length=self.hop_length)
    
    def forward(self, audio, t):
        magnitude, phase = self.audio_to_spectrogram(audio)
        
        # Diffusion on magnitude spectrogram
        predicted_noise = self.unet_2d(magnitude, t)
        
        return predicted_noise, phase  # Phase is preserved
```

**Answer:** Spectrogram diffusion processes audio in frequency domain using 2D U-Nets on magnitude spectrograms, preserving phase information for high-quality audio reconstruction.

---

## Question 37
**Explain safe completions via policy guidance.**

### Theory
Policy guidance steers diffusion models toward safe, appropriate content by incorporating reward models or safety classifiers during generation.

```python
class SafetyGuidedDiffusion:
    def __init__(self, diffusion_model, safety_classifier):
        self.diffusion_model = diffusion_model
        self.safety_classifier = safety_classifier
    
    def safe_sampling_step(self, x, t, prompt, safety_scale=2.0):
        """Sample with safety guidance"""
        x.requires_grad_(True)
        
        # Standard diffusion prediction
        noise_pred = self.diffusion_model(x, t, prompt)
        
        # Safety scoring
        safety_score = self.safety_classifier(x)
        safety_loss = -torch.log(safety_score + 1e-8)  # Higher score = lower loss
        
        # Compute safety gradient
        safety_grad = torch.autograd.grad(safety_loss, x)[0]
        
        # Apply safety guidance
        guided_noise = noise_pred - safety_scale * safety_grad
        
        return guided_noise
    
    def generate_safe_content(self, prompt, safety_threshold=0.8):
        """Generate content with safety filtering"""
        x = torch.randn(1, 3, 512, 512)
        
        for t in reversed(range(1000)):
            x = self.safe_sampling_step(x, t, prompt)
            
            # Check safety periodically
            if t % 100 == 0:
                safety_score = self.safety_classifier(x)
                if safety_score < safety_threshold:
                    # Restart or apply stronger guidance
                    safety_scale *= 1.5
        
        return x
```

**Answer:** Policy guidance incorporates safety classifiers or reward models during sampling to steer diffusion models away from harmful content while maintaining generation quality and diversity.

---

## Question 38
**Describe hardware acceleration (FP8) for diffusion.**

### Theory
Next-generation hardware optimizations like FP8 precision and specialized tensor units dramatically accelerate diffusion model training and inference.

```python
# FP8 training (theoretical implementation)
def fp8_diffusion_training(model, dataloader):
    """FP8 precision training for maximum efficiency"""
    
    # Configure FP8 settings
    fp8_config = {
        'compute_precision': 'FP8_E4M3',    # 8-bit with 4 exp, 3 mantissa
        'gradient_precision': 'FP8_E5M2',   # 8-bit with 5 exp, 2 mantissa
        'master_weights': 'FP16'            # Keep master in FP16
    }
    
    for batch in dataloader:
        with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
            loss = ddpm_loss(model, batch)
        
        # FP8 gradient computation
        loss.backward()
        
        # Update with FP16 master weights
        optimizer.step()

# Hardware-specific optimizations
def optimize_for_h100():
    """Optimizations for NVIDIA H100 tensor cores"""
    optimizations = {
        'tensor_parallel': 'Use multiple GPUs efficiently',
        'flash_attention': 'Memory-efficient attention computation', 
        'kernel_fusion': 'Fuse operations to reduce memory bandwidth',
        'fp8_transformer': 'Native FP8 transformer blocks'
    }
    
    return optimizations
```

**Answer:** FP8 precision and specialized tensor cores (H100) provide 2-4x speedup for diffusion models by reducing memory bandwidth and increasing compute throughput with minimal accuracy loss.

---

## Question 39
**Explain mixture of experts diffusion.**

### Theory
Mixture of Experts (MoE) scales diffusion models by using specialized sub-networks for different aspects of generation, activated conditionally.

```python
class MoEDiffusionBlock(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            FeedForward(dim) for _ in range(num_experts)
        ])
        
    def forward(self, x, t):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Router decides which experts to use
        router_logits = self.router(x_flat)  # [B*L, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Expert computation
        expert_outputs = []
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                expert_outputs.append((expert_mask, expert_output))
        
        # Combine expert outputs
        final_output = torch.zeros_like(x_flat)
        for mask, output in expert_outputs:
            final_output[mask] += output
        
        return final_output.view(batch_size, seq_len, dim)
```

**Answer:** MoE diffusion uses multiple expert networks with learned routing, allowing models to scale parameters while keeping computation constant by activating only relevant experts per input.

---

## Question 40
**Discuss evaluation metrics (CLIP-FID).**

### Theory
Diffusion models require specialized metrics that capture both visual quality and semantic coherence, particularly for text-conditioned generation.

```python
import clip
from scipy import linalg

def compute_fid(real_features, generated_features):
    """Fréchet Inception Distance"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def compute_clip_score(images, texts, clip_model):
    """CLIP Score for text-image alignment"""
    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    clip_score = (image_features * text_features).sum(dim=-1)
    return clip_score.mean()

class DiffusionEvaluator:
    def __init__(self):
        self.clip_model = clip.load("ViT-B/32")[0]
        self.inception_model = InceptionV3()
    
    def evaluate_model(self, diffusion_model, test_prompts, num_samples=1000):
        generated_images = []
        
        for prompt in test_prompts:
            for _ in range(num_samples // len(test_prompts)):
                image = diffusion_model.sample(prompt)
                generated_images.append(image)
        
        metrics = {
            'fid': self.compute_fid(generated_images),
            'clip_score': self.compute_clip_score(generated_images, test_prompts),
            'inception_score': self.compute_inception_score(generated_images)
        }
        
        return metrics
```

**Answer:** CLIP-FID combines visual quality (FID) with text-image alignment (CLIP score), providing comprehensive evaluation for text-to-image diffusion models beyond traditional perceptual metrics.

---

## Question 41
**Explain diffusion vs. GANs advantages.**

### Theory
Diffusion models and GANs have complementary strengths, making each suitable for different applications and requirements.

**Diffusion Advantages:**
- **Training Stability**: No adversarial dynamics, more robust training
- **Sample Diversity**: Better mode coverage, less mode collapse
- **Controllability**: Natural conditioning through guidance mechanisms
- **Quality Consistency**: More predictable generation quality

**GAN Advantages:**
- **Speed**: Single forward pass vs. iterative denoising
- **Real-time Applications**: Suitable for interactive applications
- **Architectural Flexibility**: Various discriminator designs
- **Established Techniques**: Mature optimization strategies

```python
# Speed comparison
def generation_speed_test():
    # GAN: ~10ms per image
    with torch.no_grad():
        noise = torch.randn(1, 100)
        gan_image = generator(noise)  # Single step
    
    # Diffusion: ~1000ms per image (DDPM)
    with torch.no_grad():
        noise = torch.randn(1, 3, 512, 512)
        diffusion_image = diffusion_sample(noise, steps=50)  # 50 steps
```

**Answer:** Diffusion models excel in training stability, sample diversity, and controllability, while GANs provide faster generation and real-time capability, making choice application-dependent.

---

## Question 42
**Describe computational cost mitigation.**

### Theory
Diffusion models' high computational cost requires various optimization strategies for practical deployment.

```python
class EfficientDiffusion:
    def __init__(self):
        self.optimizations = {
            'distillation': self.progressive_distillation,
            'pruning': self.model_pruning, 
            'quantization': self.fp16_inference,
            'caching': self.attention_caching,
            'parallel': self.tensor_parallelism
        }
    
    def progressive_distillation(self, teacher_model):
        """Reduce sampling steps via distillation"""
        student = copy.deepcopy(teacher_model)
        
        # Train student to match 2 teacher steps in 1 step
        for batch in dataloader:
            teacher_output = teacher_model.multi_step_sample(batch, steps=2)
            student_output = student.single_step_sample(batch)
            loss = F.mse_loss(student_output, teacher_output)
            
    def model_pruning(self, model, sparsity=0.5):
        """Remove less important parameters"""
        for name, param in model.named_parameters():
            if 'attention' not in name:  # Preserve attention weights
                mask = torch.abs(param) > torch.quantile(torch.abs(param), sparsity)
                param.data *= mask.float()
    
    def attention_caching(self, model):
        """Cache attention computations for similar inputs"""
        self.attention_cache = {}
        
        def cached_attention(query, key, value):
            cache_key = hash((query.shape, key.shape))
            if cache_key in self.attention_cache:
                return self.attention_cache[cache_key]
            
            result = torch.scaled_dot_product_attention(query, key, value)
            self.attention_cache[cache_key] = result
            return result
            
        # Replace attention function
        model.attention_fn = cached_attention
```

**Performance Improvements:**
- **Distillation**: 10-50x speedup with minimal quality loss
- **Pruning**: 2-4x speedup with structured or unstructured pruning  
- **Quantization**: 2x speedup and memory reduction
- **Parallelization**: Linear scaling with multiple GPUs

**Answer:** Computational cost mitigation uses progressive distillation (10-50x speedup), model pruning, quantization, attention caching, and parallelization to make diffusion models practical for deployment.

---

## Question 43
**Discuss legal considerations of dataset copyright.**

### Theory
Training diffusion models on copyrighted content raises complex legal questions about fair use, derivative works, and artist rights.

**Key Legal Issues:**
- **Copyright Infringement**: Training on copyrighted images without permission
- **Fair Use**: Transformative use vs. commercial harm
- **Derivative Works**: Whether generated content infringes original works
- **Artist Rights**: Moral rights and attribution concerns

**Mitigation Strategies:**
```python
class EthicalDatasetBuilder:
    def __init__(self):
        self.filters = {
            'copyright_detector': self.detect_copyrighted_content,
            'opt_out_checker': self.check_opt_out_lists,
            'license_validator': self.validate_licenses,
            'synthetic_generator': self.generate_synthetic_data
        }
    
    def build_ethical_dataset(self, raw_dataset):
        filtered_dataset = []
        
        for image, metadata in raw_dataset:
            # Check copyright status
            if self.detect_copyrighted_content(image, metadata):
                continue
                
            # Check opt-out lists (artist requests)
            if self.check_opt_out_lists(metadata.get('artist')):
                continue
                
            # Verify permissive license
            if self.validate_licenses(metadata.get('license')):
                filtered_dataset.append((image, metadata))
        
        # Augment with synthetic data if needed
        synthetic_data = self.generate_synthetic_data(len(filtered_dataset))
        
        return filtered_dataset + synthetic_data
    
    def detect_copyrighted_content(self, image, metadata):
        """Detect potentially copyrighted content"""
        # Check against known copyrighted works
        # Look for watermarks, signatures
        # Check metadata for copyright claims
        return False  # Simplified
```

**Answer:** Legal considerations include copyright infringement, fair use boundaries, and artist rights. Mitigation involves ethical dataset curation, opt-out mechanisms, and proper attribution systems.

---

## Question 44
**Explain multi-modal diffusion (image+depth).**

### Theory
Multi-modal diffusion processes multiple data types simultaneously, enabling richer control and more coherent generation across modalities.

```python
class MultiModalDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate encoders for each modality
        self.rgb_encoder = nn.Conv2d(3, 64, 3, padding=1)
        self.depth_encoder = nn.Conv2d(1, 64, 3, padding=1) 
        self.normal_encoder = nn.Conv2d(3, 64, 3, padding=1)
        
        # Fusion network
        self.fusion_conv = nn.Conv2d(64*3, 128, 1)
        
        # Shared U-Net backbone
        self.unet = UNet(input_channels=128)
        
        # Separate decoders for each modality
        self.rgb_decoder = nn.Conv2d(128, 3, 3, padding=1)
        self.depth_decoder = nn.Conv2d(128, 1, 3, padding=1)
        self.normal_decoder = nn.Conv2d(128, 3, 3, padding=1)
    
    def forward(self, rgb, depth, normals, t):
        # Encode each modality
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        normal_features = self.normal_encoder(normals)
        
        # Fuse modalities
        fused = torch.cat([rgb_features, depth_features, normal_features], dim=1)
        fused = self.fusion_conv(fused)
        
        # Shared processing
        shared_features = self.unet(fused, t)
        
        # Decode each modality
        rgb_pred = self.rgb_decoder(shared_features)
        depth_pred = self.depth_decoder(shared_features)
        normal_pred = self.normal_decoder(shared_features)
        
        return rgb_pred, depth_pred, normal_pred
    
    def multi_modal_loss(self, pred_rgb, pred_depth, pred_normals, 
                        true_rgb, true_depth, true_normals):
        """Weighted loss across modalities"""
        rgb_loss = F.mse_loss(pred_rgb, true_rgb)
        depth_loss = F.mse_loss(pred_depth, true_depth)
        normal_loss = F.mse_loss(pred_normals, true_normals)
        
        # Consistency losses
        consistency_loss = self.compute_cross_modal_consistency(
            pred_rgb, pred_depth, pred_normals
        )
        
        return rgb_loss + depth_loss + normal_loss + 0.1 * consistency_loss
```

**Answer:** Multi-modal diffusion processes RGB, depth, and surface normals jointly through shared representations, enabling geometrically consistent generation across visual modalities.

---

## Question 45
**Describe timeline of diffusion research.**

### Theory
Diffusion model development represents a rapid evolution from theoretical foundations to practical breakthroughs in generative modeling.

**Timeline:**
- **2015**: Score-based generative models (Song & Ermon)
- **2020**: DDPM establishes diffusion framework (Ho et al.)
- **2021**: DDIM enables fast sampling (Song et al.)
- **2021**: Score-based SDEs unify framework (Song et al.)
- **2022**: Latent Diffusion/Stable Diffusion (Rombach et al.)
- **2022**: DALL-E 2 demonstrates text-to-image quality (Ramesh et al.)
- **2022**: Imagen shows photorealistic generation (Saharia et al.)
- **2023**: ControlNet enables spatial control (Zhang et al.)
- **2023**: Consistency Models for fast sampling (Song et al.)
- **2024**: Advanced architectures and efficiency improvements

```python
class DiffusionEvolution:
    def __init__(self):
        self.milestones = {
            2020: {
                'ddpm': 'Foundation of diffusion models',
                'key_insight': 'Noise prediction parameterization',
                'impact': 'Established training stability'
            },
            2021: {
                'ddim': 'Fast deterministic sampling',
                'score_sde': 'Continuous formulation',
                'impact': 'Practical deployment enabled'
            },
            2022: {
                'stable_diffusion': 'Latent space efficiency',
                'dalle2': 'Text-to-image breakthrough', 
                'imagen': 'Photorealistic quality',
                'impact': 'Mainstream adoption'
            },
            2023: {
                'controlnet': 'Precise spatial control',
                'consistency_models': 'Single-step generation',
                'impact': 'Enhanced controllability'
            }
        }
```

**Answer:** Diffusion research evolved from 2015 score-based foundations through 2020 DDPM breakthrough, 2021 sampling improvements, 2022 latent space efficiency, to 2023+ advanced control and speed optimizations.

---

## Question 46
**Explain diffusion for super-resolution.**

### Theory
Diffusion super-resolution generates high-resolution images from low-resolution inputs by conditioning the diffusion process on upsampled inputs.

```python
class SuperResDiffusion(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Low-res encoder
        self.lr_encoder = nn.Conv2d(3, 64, 3, padding=1)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, scale_factor, stride=scale_factor),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        
        # Diffusion U-Net
        self.unet = UNet(input_channels=3+64)  # HR image + LR features
        
    def forward(self, hr_image, lr_image, t):
        # Encode low-res image
        lr_features = self.lr_encoder(lr_image)
        
        # Upsample to high-res
        lr_upsampled = self.upsample(lr_features)
        
        # Concatenate with noisy HR image
        combined_input = torch.cat([hr_image, lr_upsampled], dim=1)
        
        # Predict noise
        return self.unet(combined_input, t)
    
    def super_resolve(self, lr_image, steps=50):
        """Generate HR image from LR input"""
        batch_size = lr_image.shape[0]
        hr_shape = (batch_size, 3, 
                   lr_image.shape[2] * self.scale_factor,
                   lr_image.shape[3] * self.scale_factor)
        
        # Start from noise
        hr_image = torch.randn(hr_shape)
        
        # Iterative denoising conditioned on LR image
        for t in reversed(range(steps)):
            hr_image = self.denoise_step(hr_image, lr_image, t)
        
        return hr_image
```

**Answer:** Diffusion super-resolution conditions high-resolution generation on low-resolution inputs through feature concatenation, enabling realistic detail hallucination beyond simple upsampling methods.

---

## Question 47-50: [Final Questions Batch]

**Question 47: Slot diffusion for object compositionality**
**Answer:** Slot attention decomposes scenes into object-centric representations, enabling diffusion models to generate compositional scenes with controllable object placement and attributes.

**Question 48: Zero-shot human motion diffusion**
**Answer:** Human motion diffusion generates realistic poses and movements by learning from motion capture data, enabling zero-shot animation from text descriptions or pose constraints.

**Question 49: Guided diffusion in RL policy sampling**
**Answer:** Diffusion models can generate diverse policy behaviors in reinforcement learning by treating action sequences as generated samples, with reward guidance shaping policy distributions.

**Question 50: Future of diffusion in content creation**
**Answer:** Future developments will focus on real-time generation, better controllability, multi-modal integration, and specialized domain applications (video, 3D, audio) with improved efficiency and quality.

---
