# evaluate_teacher_fid.py
"""
Calculate FID score for DiT teacher model and generate sample images.
Usage:
    python evaluate_teacher_fid.py \
        --ckpt results/003/checkpoints/ckpt_step0310000_ep0794.pt \
        --train-module train_dit_teacher \
        --num-samples 100 \
        --outdir teacher_evaluation
"""
# working 
import os
import math
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import linalg


# ============================================================================
# DIFFUSION UTILITIES
# ============================================================================

def make_betas(schedule: str, diffusion_steps: int):
    schedule = schedule.lower()
    if schedule == "linear":
        scale = 1000.0 / diffusion_steps
        beta_start = scale * 1e-4
        beta_end = scale * 2e-2
        return np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
    elif schedule == "cosine":
        t = np.linspace(0, diffusion_steps, diffusion_steps + 1, dtype=np.float64) / diffusion_steps
        s = 0.008
        acp = np.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError(f"Unknown noise schedule: {schedule}")


def to_t(arr, t, shape, device):
    """Extract values from numpy array at timesteps t"""
    x = torch.from_numpy(arr).to(device)[t].float()
    while len(x.shape) < len(shape):
        x = x[..., None]
    return x.expand(shape)


def x0_from_v(x_t, t, sqrt_acp, sqrt_1m_acp, v):
    """Reconstruct x0 from v-prediction"""
    a = to_t(sqrt_acp, t, x_t.shape, x_t.device)
    b = to_t(sqrt_1m_acp, t, x_t.shape, x_t.device)
    return a * x_t - b * v


def x0_from_eps(x_t, t, sqrt_recip_acp, sqrt_recipm1_acp, eps):
    """Reconstruct x0 from epsilon-prediction"""
    return to_t(sqrt_recip_acp, t, x_t.shape, x_t.device) * x_t - \
           to_t(sqrt_recipm1_acp, t, x_t.shape, x_t.device) * eps


def q_posterior_mean(x0, x_t, t, coef1, coef2):
    """Calculate posterior mean for reverse process"""
    c1 = to_t(coef1, t, x_t.shape, x_t.device)
    c2 = to_t(coef2, t, x_t.shape, x_t.device)
    return c1 * x0 + c2 * x_t


# ============================================================================
# SAMPLING
# ============================================================================

@torch.no_grad()
def sample_images(
    model: nn.Module,
    num_samples: int,
    num_classes: int,
    image_size: int,
    device: torch.device,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    pred_type: str = "v",
    cfg_scale: float = 1.5,
    batch_size: int = 100,
    show_progress: bool = True,
):
    """
    Generate images from the model using DDPM sampling.
    """
    model.eval()

    # Precompute diffusion schedule
    betas = make_betas(noise_schedule, diffusion_steps)
    alphas = 1.0 - betas
    acp = np.cumprod(alphas, axis=0)
    acp_prev = np.append(1.0, acp[:-1])
    
    sqrt_acp = np.sqrt(acp)
    sqrt_1m_acp = np.sqrt(1.0 - acp)
    sqrt_recip_acp = np.sqrt(1.0 / acp)
    sqrt_recipm1_acp = np.sqrt(1.0 / acp - 1.0)
    
    posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)
    posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
    posterior_mean_coef1 = betas * np.sqrt(acp_prev) / (1.0 - acp)
    posterior_mean_coef2 = ((1.0 - acp_prev) * np.sqrt(alphas)) / (1.0 - acp)

    outputs = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\nTotal batches: {num_batches}")
    print(f"Steps per batch: {diffusion_steps}")
    print(f"Estimated time: ~{num_batches * 4:.0f} minutes\n")
    
    for batch_idx in range(num_batches):
        n = min(batch_size, num_samples - batch_idx * batch_size)
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches} ({n} images)")
        print(f"{'='*60}")
        
        # Sample random classes for this batch
        y_cond = torch.randint(0, num_classes, (n,), device=device)
        y_uncond = torch.full((n,), num_classes, device=device)  # null token
        
        # Start from Gaussian noise
        x = torch.randn(n, 3, image_size, image_size, device=device)
        
        # Reverse diffusion process - ALWAYS show progress
        for i in tqdm(range(diffusion_steps - 1, -1, -1), 
                     desc=f"Denoising batch {batch_idx+1}/{num_batches}",
                     ncols=80):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            
            # CFG: forward both conditional and unconditional
            out_uncond = model(x, t, y_uncond)
            out_cond = model(x, t, y_cond)
            
            C = x.shape[1]
            
            # Handle learned variance if present
            if out_cond.shape[1] == C * 2:
                pred_u, var_u = torch.split(out_uncond, C, dim=1)
                pred_c, var_c = torch.split(out_cond, C, dim=1)
                pred = pred_u + cfg_scale * (pred_c - pred_u)
                
                # Interpolate log-variance
                min_log = to_t(posterior_log_variance_clipped, t, x.shape, device)
                max_log = to_t(np.log(betas), t, x.shape, device)
                frac = (var_u + cfg_scale * (var_c - var_u) + 1) / 2
                logvar = frac * max_log + (1 - frac) * min_log
            else:
                pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                logvar = to_t(posterior_log_variance_clipped, t, x.shape, device)
            
            # Reconstruct x0
            if pred_type.lower() == "v":
                x0 = x0_from_v(x, t, sqrt_acp, sqrt_1m_acp, pred)
            else:
                x0 = x0_from_eps(x, t, sqrt_recip_acp, sqrt_recipm1_acp, pred)
            
            x0 = x0.clamp(-1, 1)
            
            # Calculate posterior mean
            mean = q_posterior_mean(x0, x, t, posterior_mean_coef1, posterior_mean_coef2)
            
            # Add noise (except at final step)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + (0.5 * logvar).exp() * noise
            else:
                x = mean
        
        outputs.append(x.cpu())
        print(f"✓ Batch {batch_idx + 1} complete")
    
    imgs = torch.cat(outputs, dim=0)
    return (imgs + 1) / 2  # Convert from [-1,1] to [0,1]

# ============================================================================
# INCEPTION V3 FOR FID
# ============================================================================

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network for FID calculation"""
    
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }
    
    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        
        from torchvision import models
        inception = models.inception_v3(pretrained=True)
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        
        # Block 0: input to maxpool1
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 1: maxpool1 to maxpool2
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Block 2: maxpool2 to aux classifier
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        
        # Block 3: aux classifier to final avgpool
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor in range [0, 1]
        Returns:
            List of feature tensors
        """
        output = []
        
        # Resize if necessary
        if self.resize_input:
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Normalize to [-1, 1]
        if self.normalize_input:
            x = 2 * x - 1
        
        # Block 0
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        if 0 in self.output_blocks:
            output.append(x)
        
        # Block 1
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        if 1 in self.output_blocks:
            output.append(x)
        
        # Block 2
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if 2 in self.output_blocks:
            output.append(x)
        
        # Block 3
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if 3 in self.output_blocks:
            output.append(x)
        
        return output


# ============================================================================
# FID CALCULATION
# ============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

@torch.no_grad()
def calculate_activation_statistics(images, inception_model, batch_size=50, device='cuda'):
    """
    Calculate mean and covariance of Inception features.
    """
    inception_model.eval()
    
    activations = []
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Computing Inception features"):
        batch = images[i * batch_size:(i + 1) * batch_size].to(device)
        
        # Get inception features
        pred = inception_model(batch)[0]  # First (and only) output block
        
        # Flatten to [batch_size, features]
        if len(pred.shape) == 4:  # [B, C, H, W]
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(-1).squeeze(-1)
        elif len(pred.shape) == 3:  # [B, C, 1]
            pred = pred.squeeze(-1)
        # If already [B, C], do nothing
        
        activations.append(pred.cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    
    # Calculate statistics
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma


def calculate_fid(real_images, fake_images, batch_size=50, device='cuda'):
    """
    Calculate FID score between real and generated images.
    """
    print("\n" + "="*70)
    print("FID CALCULATION")
    print("="*70)
    
    # Load Inception model
    print("Loading InceptionV3 model...")
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
    inception.eval()
    
    print(f"Inception model loaded. Output block: {InceptionV3.BLOCK_INDEX_BY_DIM[2048]}")
    
    # Calculate statistics for real images
    print(f"\nCalculating statistics for {len(real_images)} real images...")
    m1, s1 = calculate_activation_statistics(real_images, inception, batch_size, device)
    print(f"Real images stats: mu shape={m1.shape}, sigma shape={s1.shape}")
    
    # Calculate statistics for fake images
    print(f"\nCalculating statistics for {len(fake_images)} generated images...")
    m2, s2 = calculate_activation_statistics(fake_images, inception, batch_size, device)
    print(f"Generated images stats: mu shape={m2.shape}, sigma shape={s2.shape}")
    
    # Calculate FID
    print("\nCalculating Frechet distance...")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    print("="*70)
    print(f"FID Score: {fid_value:.2f}")
    print("="*70 + "\n")
    
    return fid_value

# ============================================================================
# MODEL LOADING
# ============================================================================

def build_model_from_train(train_module_name: str, model_name: str, 
                          image_size: int, num_classes: int, device):
    """
    Dynamically load model from training module.
    
    Args:
        train_module_name: Name of training module (without .py)
        model_name: Model configuration name (e.g., 'DiT-S/2')
        image_size: Input image size
        num_classes: Number of classes
        device: torch device
    
    Returns:
        model: DiT model
    """
    print(f"Loading model definition from module: {train_module_name}")
    trainmod = importlib.import_module(train_module_name)
    DiT = getattr(trainmod, "DiT")
    DiT_models = getattr(trainmod, "DiT_models")
    
    cfg = DiT_models[model_name]
    model = DiT(
        input_size=image_size,
        patch_size=cfg["patch_size"],
        in_channels=3,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        num_classes=num_classes,
        learn_sigma=True,
    ).to(device)
    
    model.eval()
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    """
    Load checkpoint weights into model.
    
    Args:
        model: DiT model
        ckpt_path: Path to checkpoint file
        device: torch device
        use_ema: Whether to use EMA weights
    """
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if use_ema and "ema" in ckpt:
        state_dict = ckpt["ema"]
        print("Using EMA weights")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        print("Using model weights")
    else:
        raise ValueError("Checkpoint missing 'ema' or 'model' state_dict")
    
    model.load_state_dict(state_dict, strict=True)
    
    # Print checkpoint info
    if "step" in ckpt:
        print(f"Checkpoint step: {ckpt['step']}")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_images(data_path, num_samples, image_size, batch_size=256):
    """
    Load real images from CIFAR-100 dataset.
    
    Args:
        data_path: Path to data directory
        num_samples: Number of images to load
        image_size: Image size
        batch_size: Batch size for loading
    
    Returns:
        Tensor of real images in range [0, 1]
    """
    print(f"\nLoading {num_samples} real CIFAR-100 images...")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        transform=transform,
        download=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    real_images = []
    collected = 0
    
    for images, _ in tqdm(loader, desc="Loading real images"):
        real_images.append(images)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    print(f"Loaded {len(real_images)} real images")
    return real_images


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate DiT teacher with FID score")
    
    # Model and checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to teacher checkpoint")
    parser.add_argument("--train-module", type=str, default="train_dit_teacher",
                       help="Training module name (without .py)")
    parser.add_argument("--model", type=str, default="DiT-S/2",
                       choices=["DiT-XL/2", "DiT-L/2", "DiT-B/2", "DiT-S/2"])
    parser.add_argument("--use-ema", action="store_true", default=True,
                       help="Use EMA weights")
    
    # Data
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to CIFAR-100 data")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=100)
    
    # Sampling
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of images to generate for FID")
    parser.add_argument("--sample-batch-size", type=int, default=100,
                       help="Batch size for sampling")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                       help="Classifier-free guidance scale")
    
    # Diffusion
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", type=str, default="cosine",
                       choices=["linear", "cosine"])
    parser.add_argument("--pred-type", type=str, default="v",
                       choices=["epsilon", "v"])
    
    # Output
    parser.add_argument("--outdir", type=str, default="teacher_evaluation",
                       help="Output directory")
    parser.add_argument("--save-samples", action="store_true", default=True,
                       help="Save generated samples")
    parser.add_argument("--fid-batch-size", type=int, default=50,
                       help="Batch size for FID Inception forward passes")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build and load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    model = build_model_from_train(
        args.train_module,
        args.model,
        args.image_size,
        args.num_classes,
        device
    )
    
    model = load_checkpoint(model, args.ckpt, device, args.use_ema)
    
    # Generate samples
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)
    print(f"Generating {args.num_samples} images...")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Prediction type: {args.pred_type}")
    print(f"Noise schedule: {args.noise_schedule}")
    
    fake_images = sample_images(
        model=model,
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=device,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        pred_type=args.pred_type,
        cfg_scale=args.cfg_scale,
        batch_size=args.sample_batch_size,
        show_progress=True,
    )
    
    print(f"Generated {len(fake_images)} images")
    
    # Save samples
    if args.save_samples:
        print(f"\nSaving samples to {args.outdir}/")
        
        # Save grid
        grid_size = int(np.ceil(np.sqrt(args.num_samples)))
        save_image(
            fake_images,
            os.path.join(args.outdir, "generated_grid.png"),
            nrow=grid_size,
            padding=2
        )
        print(f"Saved grid: generated_grid.png")
        
        # Save individual images
        samples_dir = os.path.join(args.outdir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        for i in range(min(100, len(fake_images))):
            save_image(
                fake_images[i],
                os.path.join(samples_dir, f"sample_{i:04d}.png")
            )
        print(f"Saved {min(100, len(fake_images))} individual samples to samples/")
    
    # Load real images
    real_images = load_real_images(
        args.data_path,
        args.num_samples,
        args.image_size
    )
    
    # Calculate FID
    fid_score = calculate_fid(
        real_images,
        fake_images,
        batch_size=args.fid_batch_size,
        device=device
    )
    
    # Save results
    results_file = os.path.join(args.outdir, "fid_results.txt")
    with open(results_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("Teacher Model FID Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"CFG scale: {args.cfg_scale}\n")
        f.write(f"Diffusion steps: {args.diffusion_steps}\n")
        f.write(f"Noise schedule: {args.noise_schedule}\n")
        f.write(f"Prediction type: {args.pred_type}\n")
        f.write(f"\n")
        f.write(f"FID Score: {fid_score:.2f}\n")
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\nEvaluation complete!")
    
    return fid_score


if __name__ == "__main__":
    main()