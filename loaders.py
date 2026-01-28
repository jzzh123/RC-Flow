import torch, hdf5storage
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Union
from sympy import nextprime
    
class NpyChannelDataset1(Dataset):

    def __init__(self, file_path: str, config: object):

        self.filename = file_path
        
        print(f"正在从以下路径加载数据: {self.filename}")
        # try:
        self.channels = np.load(self.filename).astype(np.complex64)

        n_rx = config.image_size[0]
        n_tx = config.image_size[1]
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.snr_db = config.snr_db

        if not(self.channels.ndim == 3):
            print("数据不是 3D 形状，重整形 (样本数, Nr, Nt)...")
            self.channels = np.reshape(self.channels[:,0,:,:],
               (-1, n_rx, n_tx)) 

        print(f"数据加载完成。形状: {self.channels.shape}")
        # channel_power = np.mean(np.abs(self.channels)**2)
        # self.channel_scale = np.sqrt(channel_power)
        self.channel_scale = np.std(self.channels)
        # #self.channel_scale=0.36329567
        
        num_samples = self.channels.shape[0]
        num_pilots = config.num_pilots

        #QPSK导频
        pilot_bits_real = 2 * np.random.binomial(1, 0.5, size=(num_samples, n_tx, num_pilots)) - 1
        pilot_bits_imag = 2 * np.random.binomial(1, 0.5, size=(num_samples, n_tx, num_pilots)) - 1
        self.pilots = (pilot_bits_real + 1j * pilot_bits_imag) / np.sqrt(2.0)
        
        snr_linear = 10**(self.snr_db /10)
        
        self.noise_power = self.n_tx / snr_linear
        self.noise_sigma = np.sqrt(self.noise_power)

    def __len__(self) -> int:
        return len(self.channels)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, int, float]]:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        H = self.channels[idx] / self.channel_scale
        P = self.pilots[idx]
        Y = np.matmul(H, P)
        
        noise = self.noise_sigma * (np.random.normal(size=Y.shape) + 1j * np.random.normal(size=Y.shape)) / 2**0.5
        
        Y_noisy = Y + noise
        # print(noise[:1],'\n', Y[:1], '\n', H[:1])
        sample = {
            'H': H.astype(np.complex64),   
            'P': P.astype(np.complex64),
            'Y': Y_noisy.astype(np.complex64),
            'channel_scale': np.float32(self.channel_scale),
            'noise_scale': np.float32(self.noise_sigma),

        }
        return sample
