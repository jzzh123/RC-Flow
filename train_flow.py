import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import math
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
import random

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.optim.lr_scheduler import LambdaLR
from loaders import *
from torch.utils.data import DataLoader
from typing import Union

logger = get_logger(__name__, log_level="INFO")

# ==========================================
# U-Net Architecture
# ==========================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    def forward(self, x): return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        actual_groups = min(groups, dim_out)
        self.norm = nn.GroupNorm(actual_groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, model_channels=64, channel_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        ch = model_channels
        ds_chs = [ch]
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(ch, out_ch, time_emb_dim=time_dim))
                ch = out_ch
                ds_chs.append(ch)
            if i != len(channel_mult) - 1:
                self.downs.append(Downsample(ch))
                ds_chs.append(ch)

        self.mid_block1 = ResnetBlock(ch, ch, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(ch, ch, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResnetBlock(ch + ds_chs.pop(), out_ch, time_emb_dim=time_dim))
                ch = out_ch
            if i != len(channel_mult) - 1:
                self.ups.append(Upsample(ch))

        self.final_res_block = ResnetBlock(ch, ch, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        if t.dim() > 1: t = t.flatten()
        t_emb = self.time_mlp(t * 1000)
        x = self.init_conv(x)
        h_list = [x]
        for module in self.downs:
            if isinstance(module, ResnetBlock):
                x = module(x, t_emb)
                h_list.append(x)
            else:
                x = module(x)
                h_list.append(x)
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        for module in self.ups:
            if isinstance(module, ResnetBlock):
                skip = h_list.pop()
                if x.shape[2:] != skip.shape[2:]:
                     x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
                x = torch.cat((x, skip), dim=1)
                x = module(x, t_emb)
            else:
                x = module(x)
        x = self.final_res_block(x, t_emb)
        return self.final_conv(x)

# ==========================================
# Flow Matching Loss Logic
# ==========================================

class FlowMatchingLoss:
    def __init__(self, snr_min_db, snr_max_db, snr_step_db, time_mu=-0.4, time_sigma=1.0, adaptive_p=1.0):
        self.snr_list_db = np.arange(snr_min_db, snr_max_db + snr_step_db, snr_step_db)
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.adaptive_p = adaptive_p  # 新增：控制权重强度

    def sample_time(self, batch_size, device):
        normal_samples = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        return torch.sigmoid(normal_samples)

    def __call__(self, model, batch):
        device = next(model.parameters()).device
        H_true_cplx = batch['H'].to(device)
        batch_size = H_true_cplx.shape[0]
        
        snr_db = np.random.choice(self.snr_list_db)
        snr_linear = 10**(snr_db/10)
        noise = (1 / snr_linear)**0.5 * (torch.randn_like(H_true_cplx) + 1j * torch.randn_like(H_true_cplx)) / 2**0.5
        
        x0, x1 = H_true_cplx, H_true_cplx + noise
        t = self.sample_time(batch_size, device)
        t_b = t.view(-1, 1, 1)
        
        H_t_cplx = (1 - t_b) * x0 + t_b * x1
        v_target_cplx = x1 - x0 
        H_t_real = torch.view_as_real(H_t_cplx).permute(0, 3, 1, 2).contiguous()
        v_target_real = torch.view_as_real(v_target_cplx).permute(0, 3, 1, 2).contiguous()
        
        v_pred_real = model(H_t_real, t)
        
        # --- 核心修改：改为代码二的聚合与加权逻辑 ---
        error = v_pred_real - v_target_real
        # 计算每个样本的误差范数 (聚合所有像素)
        error_norm = torch.norm(error.reshape(batch_size, -1), dim=1)
        unweighted_mse = torch.mean(error_norm ** 2)

        if self.adaptive_p > 0:
            # 计算自适应权重：1 / (误差平方 + epsilon)
            weights = 1.0 / (error_norm.detach() ** 2 + 1e-3).pow(self.adaptive_p)
            loss = (weights * error_norm ** 2).mean()
        else:
            loss = unweighted_mse
            
        return {'loss': loss, 'unweighted_mse': unweighted_mse}

# ==========================================
# Main Training Script
# ==========================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    # Set Paths Automatically
    args.output_dir = f"./results/unet_flowmt_flower_CDL_{args.dataset}_{args.scale}_Nt{args.nt}_Nr{args.nr}_UPA0.5"
    train_file_path = f'./bin/CDL-{args.dataset}_Nt{args.nt}_Nr{args.nr}_UPA0.50_train.npy'

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("unet_fm_denoising")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed + accelerator.process_index)

    model = UNet(
        model_channels=args.model_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks
    )
    ema = deepcopy(model)
    for p in ema.parameters(): p.requires_grad = False
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = FlowMatchingLoss(
        snr_min_db=args.snr_min_db, snr_max_db=args.snr_max_db, snr_step_db=args.snr_step_db,
        time_mu=args.time_mu, time_sigma=args.time_sigma
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    config = argparse.Namespace(image_size=[args.nr, args.nt], num_pilots=args.np, snr_db=1)
    train_dataset = NpyChannelDataset1(file_path=train_file_path, config=config)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, 
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed)
    ) 

    model, ema, optimizer, train_dataloader = accelerator.prepare(model, ema, optimizer, train_dataloader)
    
    global_step = 0
    progress_bar = tqdm(range(args.epochs * len(train_dataloader)), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss_dict = loss_fn(model, batch)  # 获取字典
                loss = loss_dict['loss']           # 提取加权后的 Loss 用于反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                for ema_p, model_p in zip(ema.parameters(), model.parameters()):
                    ema_p.mul_(0.999).add_(model_p.detach(), alpha=1 - 0.999)
                
                if accelerator.is_main_process:
                    # 记录 unweighted_mse 更有利于观察真实误差水平
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "mse": f"{loss_dict['unweighted_mse'].item():.4f}",
                        "epoch": epoch
                    })
                    accelerator.log({
                        "train_loss": loss.item(),
                        "train_mse": loss_dict['unweighted_mse'].item()
                    }, step=global_step)

        # Epoch-based Checkpointing (Every 5 Epochs, Save EMA Only)
        if (epoch + 1) % 10 == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"ema_unet_epoch_{epoch+1}.pt")
                unwrapped_ema = accelerator.unwrap_model(ema)
                torch.save(unwrapped_ema.state_dict(), save_path)
                logger.info(f"Saved EMA model at epoch {epoch+1}")

    accelerator.end_training()

def parse_args():
    parser = argparse.ArgumentParser()
    # Path & Dataset Config
    parser.add_argument("--dataset", type=str, default='C', help="Dataset type (e.g., C, D, Mixed)")
    parser.add_argument("--scale", type=str, default='3.88M', help="Model scale for path naming")
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=256)
    parser.add_argument("--np", type=int, default=38)

    # Model Config
    parser.add_argument("--model_channels", type=int, default=32)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    
    # Training Config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2400)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--snr_min_db", type=float, default=-10.0)
    parser.add_argument("--snr_max_db", type=float, default=30.0)
    parser.add_argument("--snr_step_db", type=float, default=5.0)
    parser.add_argument("--time_mu", type=float, default=0.0)
    parser.add_argument("--time_sigma", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)