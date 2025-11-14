import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
import logging
import os
from time import time
from collections import OrderedDict
from copy import deepcopy
import math
import numpy as np
from PIL import Image


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                                    modulate(self.norm1(x), shift_msa, scale_msa),
                                                    modulate(self.norm1(x), shift_msa, scale_msa))[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=3, hidden_size=384,
                 depth=12, num_heads=6, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=100, learn_sigma=True):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_patches = (input_size // patch_size) ** 2

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x

    def forward(self, x, t, y):
        x = self.patchify(x)
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


DiT_models = {
    'DiT-XL/2': {'depth': 28, 'hidden_size': 1152, 'patch_size': 2, 'num_heads': 16},
    'DiT-L/2':  {'depth': 24, 'hidden_size': 1024, 'patch_size': 2, 'num_heads': 16},
    'DiT-B/2':  {'depth': 12, 'hidden_size': 768,  'patch_size': 2, 'num_heads': 12},
    'DiT-S/2':  {'depth': 12, 'hidden_size': 384,  'patch_size': 2, 'num_heads': 6},
}


# ============================================================================
# DIFFUSION
# ============================================================================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    def __init__(self, betas, pred_type="epsilon"):
        self.pred_type = pred_type
        
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_losses(self, model, x_start, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        model_output = model(x_t, t, **model_kwargs)
        
        if model.learn_sigma:
            model_output, model_var_values = torch.split(model_output, x_start.shape[1], dim=1)
        
        if self.pred_type == "epsilon":
            target = noise
        elif self.pred_type == "v":
            target = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
            )
        else:
            raise NotImplementedError(f"unknown pred_type: {self.pred_type}")

        loss = (target - model_output) ** 2
        loss = loss.mean()
        
        return loss

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        model_output = model(x, t, **model_kwargs)
        
        if model.learn_sigma:
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = _extract_into_tensor(self.posterior_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        
        if self.pred_type == "epsilon":
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        elif self.pred_type == "v":
            pred_xstart = self._predict_xstart_from_v(x, t, v=model_output)
        else:
            raise NotImplementedError(f"unknown pred_type: {self.pred_type}")
        
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        
        model_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * pred_xstart
            + _extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        return model_mean, model_variance, model_log_variance, pred_xstart

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(self, x_t, t, v):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, 
                     model_kwargs=None, device=None, progress=False):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                img = self.p_sample(
                    model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
                )
        return img

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_mean, _, _, pred_xstart = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        
        eps = self._predict_eps_from_xstart(x, t, pred_xstart)
        
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        
        noise = torch.randn_like(x)
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True,
                        model_kwargs=None, device=None, progress=False, eta=0.0):
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        
        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                img = self.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised, 
                    model_kwargs=model_kwargs, eta=eta
                )
        return img


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


# ============================================================================
# DISTRIBUTED UTILITIES
# ============================================================================

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup(use_ddp):
    if use_ddp:
        dist.destroy_process_group()


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# TRAINING
# ============================================================================

def main(args):
    use_ddp = int(os.environ.get("RANK", -1)) != -1
    if use_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup experiment
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len([d for d in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, d))])
        exp_dir = f"{args.results_dir}/{experiment_index:03d}"
        ckpt_dir = f"{exp_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory: {exp_dir}")
    else:
        logger = None

    # Create model
    model = DiT(
        input_size=args.image_size,
        num_classes=args.num_classes,
        learn_sigma=bool(args.learn_sigma),
        **DiT_models[args.model]
    ).to(device)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if rank == 0:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    # DDP
    model_wo = model
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
        model_wo = model.module

    # Diffusion
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion = GaussianDiffusion(betas, pred_type=args.pred_type)

    # Data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.CIFAR100(root=args.data_path, train=True, transform=transform, download=True)
    
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(loader) // args.accum_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(step / args.warmup_steps, 1.0))

    # AMP
    if args.amp == "bf16":
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_dtype = torch.float16
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        amp_dtype = torch.float32

    # Training loop
    if rank == 0:
        logger.info(f"Starting training for {args.epochs} epochs...")
    
    train_steps = 0
    running_loss = 0.0
    t0 = time()
    
    model.train()
    for epoch in range(args.epochs):
        if use_ddp:
            sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=(args.amp != "off")):
                loss = diffusion.training_losses(model, x, t, model_kwargs={"y": y})
                loss = loss / args.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
                
                ema.update(model_wo)
                train_steps += 1
            
            running_loss += loss.item() * args.accum_steps
            
            # Logging
            if train_steps % args.log_every == 0 and (step + 1) % args.accum_steps == 0:
                t1 = time()
                samples_per_sec = (args.log_every * args.global_batch_size) / (t1 - t0)
                
                avg_loss = running_loss / args.log_every
                if use_ddp:
                    loss_tensor = torch.tensor(avg_loss, device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = loss_tensor.item() / world_size
                
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                              f"Samples/sec: {samples_per_sec:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                
                running_loss = 0.0
                t0 = time()
            
            # Checkpointing
            if train_steps % args.ckpt_every == 0 and (step + 1) % args.accum_steps == 0 and rank == 0:
                checkpoint = {
                    "model": model_wo.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": args,
                    "step": train_steps,
                    "epoch": epoch,
                }
                ckpt_path = f"{ckpt_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

    model.eval()
    if rank == 0:
        logger.info("Training complete!")
    
    cleanup(use_ddp)


# ============================================================================
# SAMPLING
# ============================================================================

def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    saved_args = checkpoint["args"]
    
    # Create model
    model = DiT(
        input_size=saved_args.image_size,
        num_classes=saved_args.num_classes,
        learn_sigma=bool(saved_args.learn_sigma),
        **DiT_models[saved_args.model]
    ).to(device)
    
    # Load weights
    if args.use_ema:
        model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model"])
        print("Loaded model weights")
    
    model.eval()
    
    # Create diffusion
    betas = get_named_beta_schedule(saved_args.noise_schedule, saved_args.diffusion_steps)
    diffusion = GaussianDiffusion(betas, pred_type=saved_args.pred_type)
    
    # Prepare class labels
    if args.class_labels is not None:
        class_labels = torch.tensor(args.class_labels, device=device)
    else:
        class_labels = torch.randint(0, saved_args.num_classes, (args.num_samples,), device=device)
    
    # Prepare for CFG
    if args.cfg_scale > 1.0:
        # Duplicate for classifier-free guidance
        class_labels = class_labels.repeat(2)
        model_kwargs = {"y": class_labels}
        
        def cfg_model_fn(x, t, **kwargs):
            half = x.shape[0] // 2
            combined = model(x, t, **kwargs)
            eps_cond, eps_uncond = combined[:half], combined[half:]
            eps = eps_uncond + args.cfg_scale * (eps_cond - eps_uncond)
            return torch.cat([eps, eps], dim=0)
        
        sample_fn = cfg_model_fn
        shape = (args.num_samples * 2, 3, saved_args.image_size, saved_args.image_size)
    else:
        model_kwargs = {"y": class_labels}
        sample_fn = model
        shape = (args.num_samples, 3, saved_args.image_size, saved_args.image_size)
    
    # Sample
    print(f"Sampling {args.num_samples} images...")
    
    if args.sampling_method == "ddpm":
        samples = diffusion.p_sample_loop(
            sample_fn,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            device=device,
            progress=True
        )
    elif args.sampling_method == "ddim":
        samples = diffusion.ddim_sample_loop(
            sample_fn,
            shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            device=device,
            progress=True,
            eta=args.ddim_eta
        )
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    # Handle CFG
    if args.cfg_scale > 1.0:
        samples = samples[:args.num_samples]
    
    # Denormalize
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    # Save samples
    os.makedirs(args.sample_dir, exist_ok=True)
    
    if args.save_grid:
        grid_size = int(np.ceil(np.sqrt(args.num_samples)))
        save_image(samples, f"{args.sample_dir}/samples_grid.png", nrow=grid_size, padding=2)
        print(f"Saved grid to {args.sample_dir}/samples_grid.png")
    
    if args.save_individual:
        for i, sample in enumerate(samples):
            save_image(sample, f"{args.sample_dir}/sample_{i:04d}_class_{class_labels[i].item():03d}.png")
        print(f"Saved {args.num_samples} individual images to {args.sample_dir}")
    
    print("Sampling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", help="train or sample")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-path", type=str, default="./data")
    train_parser.add_argument("--results-dir", type=str, default="results")
    train_parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    train_parser.add_argument("--image-size", type=int, choices=[32, 64, 256], default=32)
    train_parser.add_argument("--num-classes", type=int, default=100)
    train_parser.add_argument("--learn-sigma", type=int, choices=[0, 1], default=1)
    train_parser.add_argument("--diffusion-steps", type=int, default=1000)
    train_parser.add_argument("--noise-schedule", type=str, choices=["linear", "cosine"], default="cosine")
    train_parser.add_argument("--pred-type", type=str, choices=["epsilon", "v"], default="v")
    train_parser.add_argument("--epochs", type=int, default=800)
    train_parser.add_argument("--global-batch-size", type=int, default=256)
    train_parser.add_argument("--global-seed", type=int, default=0)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--warmup-steps", type=int, default=2000)
    train_parser.add_argument("--ema-decay", type=float, default=0.9999)
    train_parser.add_argument("--ckpt-every", type=int, default=10_000)
    train_parser.add_argument("--log-every", type=int, default=200)
    train_parser.add_argument("--amp", type=str, choices=["off", "bf16", "fp16"], default="bf16")
    train_parser.add_argument("--accum-steps", type=int, default=1)
    train_parser.add_argument("--grad-checkpoint", action="store_true")
    
    # Sampling arguments
    sample_parser = subparsers.add_parser("sample", help="Sample from trained model")
    sample_parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    sample_parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate")
    sample_parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to save samples")
    sample_parser.add_argument("--sampling-method", type=str, choices=["ddpm", "ddim"], default="ddpm")
    sample_parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM eta parameter")
    sample_parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    sample_parser.add_argument("--class-labels", type=int, nargs="+", default=None, help="Specific class labels to generate")
    sample_parser.add_argument("--use-ema", action="store_true", help="Use EMA weights")
    sample_parser.add_argument("--save-grid", action="store_true", help="Save samples as grid")
    sample_parser.add_argument("--save-individual", action="store_true", help="Save individual sample images")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main(args)
    elif args.mode == "sample":
        sample(args)
    else:
        parser.print_help()
