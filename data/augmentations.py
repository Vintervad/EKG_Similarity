from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, resample_poly


def _ensure_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        return x.unsqueeze(0), True
    if x.ndim != 3:
        raise ValueError(
            f"Expected ECG tensor with shape [batch, leads, time] or [leads, time], got {tuple(x.shape)}."
        )
    return x, False


class Compose:
    def __init__(self, transforms: Iterable[object]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


@dataclass
class RandomAmplitudeScale:
    p: float = 0.5
    min_scale: float = 0.85
    max_scale: float = 1.15
    per_lead: bool = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        x_batched, squeezed = _ensure_batched(x)
        shape = (x_batched.size(0), x_batched.size(1), 1) if self.per_lead else (x_batched.size(0), 1, 1)
        scales = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(self.min_scale, self.max_scale)
        out = x_batched * scales
        return out.squeeze(0) if squeezed else out


@dataclass
class GaussianNoise:
    p: float = 0.5
    std: float = 0.01

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise


@dataclass
class RandomTimeShift:
    p: float = 0.5
    max_shift_fraction: float = 0.02

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        x_batched, squeezed = _ensure_batched(x)
        max_shift = max(1, int(x_batched.size(-1) * self.max_shift_fraction))
        out = torch.empty_like(x_batched)
        for index in range(x_batched.size(0)):
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
            out[index] = torch.roll(x_batched[index], shifts=shift, dims=-1)
        return out.squeeze(0) if squeezed else out


@dataclass
class RandomTimeMask:
    p: float = 0.3
    max_mask_fraction: float = 0.08

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        x_batched, squeezed = _ensure_batched(x)
        seq_len = x_batched.size(-1)
        max_mask = max(1, int(seq_len * self.max_mask_fraction))
        out = x_batched.clone()
        for index in range(out.size(0)):
            mask_len = int(torch.randint(1, max_mask + 1, (1,), device=x.device).item())
            start_max = max(1, seq_len - mask_len + 1)
            start = int(torch.randint(0, start_max, (1,), device=x.device).item())
            out[index, :, start : start + mask_len] = 0
        return out.squeeze(0) if squeezed else out


@dataclass
class RandomLeadDropout:
    p: float = 0.2
    max_drop_fraction: float = 0.25

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        x_batched, squeezed = _ensure_batched(x)
        num_leads = x_batched.size(1)
        max_drop = max(1, int(num_leads * self.max_drop_fraction))
        out = x_batched.clone()
        for index in range(out.size(0)):
            drop_count = int(torch.randint(1, max_drop + 1, (1,), device=x.device).item())
            dropped = torch.randperm(num_leads, device=x.device)[:drop_count]
            out[index, dropped] = 0
        return out.squeeze(0) if squeezed else out


@dataclass
class BaselineWander:
    p: float = 0.3
    max_amplitude: float = 0.05
    frequency_range: Sequence[float] = (0.05, 0.5)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        x_batched, squeezed = _ensure_batched(x)
        batch, leads, seq_len = x_batched.shape
        timeline = torch.linspace(0, 1, seq_len, device=x.device, dtype=x.dtype)
        out = x_batched.clone()
        min_freq, max_freq = self.frequency_range
        for index in range(batch):
            frequency = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(min_freq, max_freq).item()
            amplitude = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(0, self.max_amplitude).item()
            phase = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(0, 2 * torch.pi).item()
            drift = amplitude * torch.sin(2 * torch.pi * frequency * timeline + phase)
            out[index] = out[index] + drift.view(1, seq_len).expand(leads, -1)
        return out.squeeze(0) if squeezed else out


@dataclass
class ClinicalBandpassFilter:
    fs: float = 500.0
    lowcut: float = 0.016
    highcut: float = 150.0
    order: int = 3

    def __post_init__(self) -> None:
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.b, self.a = butter(self.order, [low, high], btype="band")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        filtered = filtfilt(self.b, self.a, x_np, axis=-1)
        return torch.from_numpy(filtered.copy()).to(device=x.device, dtype=x.dtype)


class PhysioNetNoise:
    def __init__(
        self,
        noise_dir: str = "physionet_data",
        target_snr_db: float = 5.0,
        fs: float = 500.0,
    ) -> None:
        self.noise_dir = noise_dir
        self.target_snr_db = target_snr_db
        self.fs = fs
        self.snr_linear = 10 ** (target_snr_db / 10.0)
        self.noise_banks = self._load_noise_banks()

    def _load_noise_banks(self) -> dict[str, np.ndarray]:
        noise_banks = {}
        original_fs = 360
        for noise_type in ["ma", "bw", "em"]:
            path = os.path.join(self.noise_dir, noise_type)
            if not (os.path.exists(f"{path}.dat") and os.path.exists(f"{path}.hea")):
                raise FileNotFoundError(
                    f"PhysioNet noise record {noise_type!r} not found in {self.noise_dir}. "
                    "Run preproc/setup_noise.py first or specify --physionet-noise-dir."
                )
            record = wfdb.rdrecord(path)
            raw_noise = record.p_signal[:, 0].astype(np.float32)
            resampled_noise = resample_poly(raw_noise, up=int(self.fs), down=original_fs)
            noise_banks[noise_type] = resampled_noise.astype(np.float32)
        return noise_banks

    def _get_scaled_noise(self, clean_lead: np.ndarray) -> np.ndarray:
        ecg_length = clean_lead.shape[-1]
        composite_noise = np.zeros(ecg_length, dtype=np.float32)
        for noise_type in ["ma", "bw", "em"]:
            bank = self.noise_banks[noise_type]
            if len(bank) <= ecg_length:
                noise_slice = np.resize(bank, ecg_length)
            else:
                start_idx = np.random.randint(0, len(bank) - ecg_length)
                noise_slice = bank[start_idx : start_idx + ecg_length]
            composite_noise += noise_slice
        
        sig_power = np.mean(clean_lead**2)
        noise_power = np.mean(composite_noise**2)
        if noise_power > 0:
            target_noise_power = sig_power / self.snr_linear
            scaling_factor = np.sqrt(target_noise_power / noise_power)
            composite_noise *= scaling_factor
        return composite_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_batched, squeezed = _ensure_batched(x)
        batch_size, num_leads, seq_len = x_batched.shape
        x_np = x_batched.detach().cpu().numpy()
        out_np = np.zeros_like(x_np)
        
        for b in range(batch_size):
            for l in range(num_leads):
                noise = self._get_scaled_noise(x_np[b, l])
                out_np[b, l] = x_np[b, l] + noise
        
        out = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)
        return out.squeeze(0) if squeezed else out


class PhysioNetTwoViewAugmentor:
    def __init__(
        self,
        noise_dir: str = "physionet_data",
        target_snr_db: float = 5.0,
        fs: float = 500.0,
    ) -> None:
        self.filter = ClinicalBandpassFilter(fs=fs)
        self.noise = PhysioNetNoise(noise_dir=noise_dir, target_snr_db=target_snr_db, fs=fs)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = self.filter(x)
        noisy = self.noise(filtered)
        return filtered, noisy


class TwoViewECGAugmentor:
    def __init__(self, transform: Compose | None = None) -> None:
        self.transform = transform or Compose(
            [
                RandomAmplitudeScale(),
                GaussianNoise(),
                RandomTimeShift(),
                RandomTimeMask(),
                # RandomLeadDropout(),
                BaselineWander(),
            ]
        )

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x.clone())

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.augment(x), self.augment(x)
