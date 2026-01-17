# Generative Models & CV Applications - Interview Questions

## GANs Fundamentals

### Question 1
**Explain the generator and discriminator adversarial training dynamics. What is mode collapse and how do you prevent it?**

**Answer:** _To be filled_

---

### Question 2
**Compare GAN loss functions: vanilla GAN, WGAN, WGAN-GP, and hinge loss. When would you use each?**

**Answer:** _To be filled_

---

### Question 3
**Explain progressive growing in GANs and how it enables high-resolution image generation.**

**Answer:** _To be filled_

---

### Question 4
**What are FID (Fréchet Inception Distance) and Inception Score? How do you evaluate generative model quality?**

**Answer:** _To be filled_

---

## StyleGAN Family

### Question 5
**Explain StyleGAN's mapping network and how the W latent space differs from Z space.**

**Answer:** _To be filled_

---

### Question 6
**Describe AdaIN (Adaptive Instance Normalization) and how it injects style at each layer.**

**Answer:** _To be filled_

---

### Question 7
**Explain the separation of coarse, middle, and fine styles in StyleGAN layers.**

**Answer:** _To be filled_

---

### Question 8
**What are StyleGAN2's key improvements: weight demodulation, path-length regularization, no progressive growing?**

**Answer:** _To be filled_

---

### Question 9
**Explain StyleGAN3's alias-free design and how it eliminates texture sticking artifacts.**

**Answer:** _To be filled_

---

### Question 10
**Describe the truncation trick in StyleGAN. How does ψ parameter trade diversity for quality?**

**Answer:** _To be filled_

---

### Question 11
**Explain StyleGAN latent spaces: Z, W, W+, and S space. Which is best for image editing?**

**Answer:** _To be filled_

---

### Question 12
**How do GAN inversion methods (e4e, pSp, optimization-based) project real images to latent space?**

**Answer:** _To be filled_

---

### Question 13
**Explain semantic editing in StyleGAN latent space (InterFaceGAN, GANSpace, StyleCLIP).**

**Answer:** _To be filled_

---

## CycleGAN & Unpaired Translation

### Question 14
**Explain image-to-image translation without paired data. How does cycle-consistency loss work?**

**Answer:** _To be filled_

---

### Question 15
**What is the identity loss in CycleGAN and when is it necessary?**

**Answer:** _To be filled_

---

### Question 16
**Explain the PatchGAN discriminator and why patch-level discrimination works better than image-level.**

**Answer:** _To be filled_

---

### Question 17
**What are CycleGAN's limitations (geometry changes, semantic consistency) and how does CUT improve on them?**

**Answer:** _To be filled_

---

### Question 18
**Explain multi-domain translation with StarGAN vs. training separate CycleGAN models.**

**Answer:** _To be filled_

---

## Pix2Pix & Paired Translation

### Question 19
**Explain conditional GAN architecture in Pix2Pix and the role of L1 reconstruction loss.**

**Answer:** _To be filled_

---

### Question 20
**Describe the U-Net generator in Pix2Pix and why skip connections help paired translation.**

**Answer:** _To be filled_

---

### Question 21
**How does Pix2PixHD achieve high-resolution translation with multi-scale discriminators?**

**Answer:** _To be filled_

---

### Question 22
**Compare applications: edges-to-photo, semantic-to-photo, day-to-night. What determines task difficulty?**

**Answer:** _To be filled_

---

## Style Transfer

### Question 23
**Explain neural style transfer: content loss, style loss (Gram matrices), and the optimization process.**

**Answer:** _To be filled_

---

### Question 24
**Compare optimization-based vs. feed-forward style transfer. What are the trade-offs?**

**Answer:** _To be filled_

---

### Question 25
**How do you implement real-time arbitrary style transfer (AdaIN, WCT)?**

**Answer:** _To be filled_

---

### Question 26
**What techniques maintain temporal consistency in video style transfer?**

**Answer:** _To be filled_

---

### Question 27
**How do you balance content preservation vs. style adoption? What controls this trade-off?**

**Answer:** _To be filled_

---

## 3D Reconstruction (NeRF, Gaussian Splatting)

### Question 28
**Explain NeRF's core idea: representing scenes as neural radiance fields with MLPs.**

**Answer:** _To be filled_

---

### Question 29
**Describe volumetric rendering in NeRF. How are colors and densities integrated along rays?**

**Answer:** _To be filled_

---

### Question 30
**What are positional encodings in NeRF and why do they help capture high-frequency details?**

**Answer:** _To be filled_

---

### Question 31
**How does 3D Gaussian Splatting achieve real-time rendering compared to NeRF's slow inference?**

**Answer:** _To be filled_

---

### Question 32
**Explain spherical harmonics for view-dependent color in Gaussian Splatting.**

**Answer:** _To be filled_

---

### Question 33
**Compare NeRF vs. Gaussian Splatting for quality, speed, and memory requirements.**

**Answer:** _To be filled_

---

### Question 34
**How do you handle dynamic scenes in NeRF (D-NeRF, Nerfies)?**

**Answer:** _To be filled_

---

### Question 35
**What techniques reduce NeRF training time (Instant-NGP, TensoRF, Plenoxels)?**

**Answer:** _To be filled_

---

### Question 36
**How do you handle sparse or unevenly distributed camera viewpoints in 3D reconstruction?**

**Answer:** _To be filled_

---

## OCR (Optical Character Recognition)

### Question 37
**Compare traditional OCR pipeline (detection + recognition) vs. end-to-end approaches.**

**Answer:** _To be filled_

---

### Question 38
**How do transformer-based OCR models (TrOCR) improve upon CNN-RNN-CTC approaches?**

**Answer:** _To be filled_

---

### Question 39
**What are the key challenges in handwritten vs. printed text recognition?**

**Answer:** _To be filled_

---

### Question 40
**How do you handle OCR for documents with complex layouts, tables, and mixed content?**

**Answer:** _To be filled_

---

### Question 41
**Explain preprocessing steps (binarization, deskewing, denoising) for improving OCR accuracy.**

**Answer:** _To be filled_

---

### Question 42
**How do you implement OCR post-processing with language models to correct errors?**

**Answer:** _To be filled_

---

### Question 43
**What techniques work for multilingual OCR with different scripts and writing directions?**

**Answer:** _To be filled_

---

## Super-Resolution

### Question 44
**Compare PSNR-oriented vs. perceptual quality-oriented super-resolution models.**

**Answer:** _To be filled_

---

### Question 45
**Explain SRGAN and ESRGAN. How do perceptual and adversarial losses improve visual quality?**

**Answer:** _To be filled_

---

### Question 46
**How do you handle real-world degradation (blur, noise, compression) vs. simple bicubic downsampling?**

**Answer:** _To be filled_

---

### Question 47
**What techniques preserve fine details and textures during upscaling?**

**Answer:** _To be filled_

---

### Question 48
**How do you implement efficient super-resolution for real-time video streaming?**

**Answer:** _To be filled_

---

## Facial Recognition

### Question 49
**Explain face embedding networks (FaceNet, ArcFace, CosFace) and metric learning losses.**

**Answer:** _To be filled_

---

### Question 50
**How do you handle face recognition across different ethnicities with fairness considerations?**

**Answer:** _To be filled_

---

### Question 51
**What techniques work for face recognition with masks, glasses, or partial occlusion?**

**Answer:** _To be filled_

---

### Question 52
**Explain liveness detection and anti-spoofing techniques for face recognition systems.**

**Answer:** _To be filled_

---

### Question 53
**How do you design face recognition systems robust to aging and appearance changes?**

**Answer:** _To be filled_

---

### Question 54
**What are the privacy considerations when deploying facial recognition in public spaces?**

**Answer:** _To be filled_

---

## Video Tracking

### Question 55
**Compare detection-based (tracking-by-detection) vs. correlation-based tracking approaches.**

**Answer:** _To be filled_

---

### Question 56
**Explain multi-object tracking (MOT) and the data association problem (Hungarian algorithm, DeepSORT).**

**Answer:** _To be filled_

---

### Question 57
**How do you handle tracking through occlusions and temporary object disappearances?**

**Answer:** _To be filled_

---

### Question 58
**What techniques maintain identity consistency across long video sequences?**

**Answer:** _To be filled_

---

### Question 59
**Explain re-identification (ReID) features and their role in multi-camera tracking.**

**Answer:** _To be filled_

---

### Question 60
**How do you implement real-time tracking with computational efficiency constraints?**

**Answer:** _To be filled_

---

## Diffusion Models (Bonus)

### Question 61
**Explain the forward and reverse diffusion process. How do diffusion models generate images?**

**Answer:** _To be filled_

---

### Question 62
**Compare diffusion models vs. GANs for image generation quality, diversity, and training stability.**

**Answer:** _To be filled_

---

### Question 63
**How do classifier-free guidance and text conditioning work in Stable Diffusion?**

**Answer:** _To be filled_

---

### Question 64
**Explain ControlNet and how it adds spatial control to diffusion models.**

**Answer:** _To be filled_

---

### Question 65
**What techniques speed up diffusion model inference (DDIM, DPM-Solver, distillation)?**

**Answer:** _To be filled_

---
