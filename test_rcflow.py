import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import math
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from loaders import NpyChannelDataset1

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
        return self.act(x)

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
            scale_shift = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        return self.block2(h) + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, model_channels=32, channel_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(model_channels), nn.Linear(model_channels, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
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
            if i != len(channel_mult) - 1: self.ups.append(Upsample(ch))
        self.final_res_block = ResnetBlock(ch, ch, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        if t.dim() > 1: t = t.flatten()
        t_emb = self.time_mlp(t * 1000)
        x = self.init_conv(x)
        h_list = [x]
        for module in self.downs:
            if isinstance(module, ResnetBlock): x = module(x, t_emb); h_list.append(x)
            else: x = module(x); h_list.append(x)
        x = self.mid_block2(self.mid_block1(x, t_emb), t_emb)
        for module in self.ups:
            if isinstance(module, ResnetBlock):
                skip = h_list.pop()
                if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
                x = module(torch.cat((x, skip), dim=1), t_emb)
            else: x = module(x)
        return self.final_conv(self.final_res_block(x, t_emb))

# ==========================================
# Inference Logic
# ==========================================

def get_dynamic_nfe(noise_sigma, nt):
    sigma_min = (nt/1000)**0.5
    sigma_max = (nt*10)**0.5
    ratio = ((math.log10(sigma_max / noise_sigma)) / (math.log10(sigma_max / sigma_min)))**2
    N_max=50
    N_min=3
    nfe2 = int(N_min + ratio * (N_max - N_min))
    return 100, max(nfe2, 2)

def perform_inference(model, Y_cplx, P_cplx, device, gamma, noise_scale, H_true, snr_db, args):
    batch_size, Nr, Np = Y_cplx.shape
    Nt = P_cplx.shape[1]
    nfe1, nfe2 = get_dynamic_nfe(noise_scale[1], Nt)
    
    P_H = P_cplx.mH
    Y_P_H = torch.bmm(Y_cplx, P_H)
    P_P_H = torch.bmm(P_cplx, P_H)
    I_Nt = torch.eye(Nt, device=device, dtype=Y_P_H.dtype).expand(batch_size, Nt, Nt)
    noise_scale = noise_scale.view(batch_size, 1, 1)
    
    dt = 1.0 / (nfe2)
    H_t = (torch.randn_like(Y_P_H) + 1j * torch.randn_like(Y_P_H)) / 2**0.5
    nmse_list = []

    for k in tqdm(range(nfe1)):
        epsilon = H_t
        
        err = torch.mean(torch.sum(torch.abs(H_t - H_true)**2, dim=(-1, -2)) / torch.sum(torch.abs(H_true)**2, dim=(-1, -2)))
        nmse_list.append(10 * math.log10(err.item()))
        
        for i in range(nfe2):
            t_curr = (1.0 - i * dt)**args.lamda 
            t_batch = torch.full((batch_size,), t_curr, device=device)
            
            # 1. Flow Prediction
            H_t_real = torch.view_as_real(H_t).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                u = model(H_t_real, t_batch)
                H0_real = H_t_real - t_curr * u
            H0 = torch.view_as_complex(H0_real.permute(0, 2, 3, 1).contiguous())
            

            
            # 2. Physics Constraint (Likelihood)
            vt_sq = t_curr**2 / (t_curr**2 + (1 - t_curr)**2 + 1e-6)
            rhs = Y_P_H / noise_scale**2 + H0 / (vt_sq + 1e-6) 
            lhs = P_P_H / noise_scale**2 + I_Nt / (vt_sq + 1e-6)
            lhs_inv = torch.linalg.solve(lhs, I_Nt)
            
            H0_y = torch.bmm(rhs, lhs_inv) #+ args.gamma * kappa_t
            

            
            # 3. Step
            t_next = (1.0 - (i + 1)*dt)**args.beta
            if k == 0: 
                epsilon = (torch.randn_like(H_t) + 1j * torch.randn_like(H_t)) / 2**0.5
            H_t = t_next * epsilon + (1 - t_next) * H0_y 
            # H_t = H0_y
            
            # err = torch.mean(torch.sum(torch.abs(H_t - H_true)**2, dim=(-1, -2)) / torch.sum(torch.abs(H_true)**2, dim=(-1, -2)))
            # nmse_list.append(10 * math.log10(err.item()))

        # print(min(nmse_list))
    # Plotting
    min_nmse = min(nmse_list)

    min_idx = nmse_list.index(min_nmse)

    save_dir = Path(f"./fig/eval_CDL{args.dataset}_to_CDL{args.test_dataset}_{args.scale}_Nt{args.nt}_Nr{args.nr}")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir/f"beta_{args.beta}_snr_{snr_db}.npy", nmse_list)
    plt.figure()
    plt.plot(nmse_list)
    plt.text(min_idx, min_nmse,            
        f"({min_idx+1},{min_nmse:.4f})",    
        fontsize=10)
    plt.title(f"SNR {snr_db}dB")
    plt.savefig(save_dir / f"snr_{snr_db}dB_Np{args.np}_epoch{args.epoch}.pdf")
    plt.close()

    return H_t, nfe1, nfe2

def main(args):
    # Auto Path Config
    args.checkpoint_path = f"./results/unet_flowmt_flower_CDL_{args.dataset}_{args.scale}_Nt{args.nt}_Nr{args.nr}/ema_unet_epoch_{args.epoch}.pt"
    args.test_file_path = f"./bin/CDL-{args.test_dataset}_Nt{args.nt}_Nr{args.nr}_ULA0.50_test.npy"
    
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    model = UNet(model_channels=args.model_channels, channel_mult=tuple(args.channel_mult), num_res_blocks=args.num_res_blocks).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    snr_range = np.arange(args.snr_min_db, args.snr_max_db + 1, args.snr_step_db)
    print(f"Testing CDL-{args.dataset} model on CDL-{args.test_dataset} data...")

    for snr in snr_range:
        loader = DataLoader(NpyChannelDataset1(args.test_file_path, argparse.Namespace(image_size=[args.nr, args.nt], num_pilots=args.np, snr_db=snr)), 
                            batch_size=args.batch_size, shuffle=False)
        
        all_nmse = []
        for batch in tqdm(loader, desc=f"SNR {snr}dB"):
            H_true = batch['H'].to(device)
            H_est, n1, n2 = perform_inference(model, batch['Y'].to(device), batch['P'].to(device), device, args.gamma, batch['noise_scale'].to(device), H_true, snr, args)
            
            nmse = torch.sum(torch.abs(H_est - H_true)**2, dim=(-1, -2)) / torch.sum(torch.abs(H_true)**2, dim=(-1, -2))
            all_nmse.extend(nmse.cpu().numpy())
            
        print(f"SNR: {snr}dB | NMSE: {10*math.log10(np.mean(all_nmse)):.2f}dB | NFE: {n1}x{n2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='C', help="Training dataset type")
    parser.add_argument("--test_dataset", type=str, default='C', help="Testing dataset type")
    parser.add_argument("--scale", type=str, default='3.88M')
    parser.add_argument("--epoch", type=int, default=1600)
    parser.add_argument("--nr", type=int, default=16)
    parser.add_argument("--nt", type=int, default=64)
    parser.add_argument("--np", type=int, default=7)
    parser.add_argument("--snr_min_db", type=float, default=10)
    parser.add_argument("--snr_max_db", type=float, default=10)
    parser.add_argument("--snr_step_db", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model_channels", type=int, default=32)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lamda", type=int, default=2)
    parser.add_argument("--beta", type=int, default=2)
    main(parser.parse_args())
