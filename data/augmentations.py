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
        shape = (
            (x_batched.size(0), x_batched.size(1), 1)
            if self.per_lead
            else (x_batched.size(0), 1, 1)
        )
        scales = torch.empty(shape, device=x.device, dtype=x.dtype).uniform_(
            self.min_scale, self.max_scale
        )
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
            drop_count = int(
                torch.randint(1, max_drop + 1, (1,), device=x.device).item()
            )
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
            frequency = (
                torch.empty(1, device=x.device, dtype=x.dtype)
                .uniform_(min_freq, max_freq)
                .item()
            )
            amplitude = (
                torch.empty(1, device=x.device, dtype=x.dtype)
                .uniform_(0, self.max_amplitude)
                .item()
            )
            phase = (
                torch.empty(1, device=x.device, dtype=x.dtype)
                .uniform_(0, 2 * torch.pi)
                .item()
            )
            drift = amplitude * torch.sin(2 * torch.pi * frequency * timeline + phase)
            out[index] = out[index] + drift.view(1, seq_len).expand(leads, -1)
        return out.squeeze(0) if squeezed else out


@dataclass
class ClinicalBandpassFilter:
    fs: float = 500.0  # Sampling frequency in Hz
    lowcut: float = 0.016  # Hz
    highcut: float = 150.0  # Hz
    order: int = 3  # Butterworth filter order 3

    def __post_init__(self) -> None:  # Initialize Butterworth filter coefficients
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist  # Normalize lowcut frequency
        high = self.highcut / nyquist  # Normalize highcut frequency
        self.b, self.a = butter(
            self.order, [low, high], btype="band"
        )  # Compute Butterworth filter coefficients

    def __call__(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Apply Butterworth filter to input tensor
        x_np = x.detach().cpu().numpy()  # Convert input tensor to numpy array
        filtered = filtfilt(self.b, self.a, x_np, axis=-1)
        return torch.from_numpy(filtered.copy()).to(device=x.device, dtype=x.dtype)


class PhysioNetNoise:  # Load and apply PhysioNet noise
    def __init__(
        self,
        noise_dir: str = "physionet_data",
        target_snr_db: float = 5.0,  # Target signal-to-noise ratio in dB
        fs: float = 500.0,  # Sampling frequency in Hz
    ) -> None:
        self.noise_dir = noise_dir  # Directory containing PhysioNet noise records
        self.target_snr_db = target_snr_db  # Target signal-to-noise ratio in dB
        self.fs = fs  # Sampling frequency in Hz
        self.snr_linear = 10 ** (target_snr_db / 10.0)  # Convert dB to linear scale
        self.noise_banks = (
            self._load_noise_banks()
        )  # Load noise banks from PhysioNet records

    def _load_noise_banks(
        self,
    ) -> dict[str, np.ndarray]:  # Load noise banks from PhysioNet records
        noise_banks = {}
        original_fs = 360  # Original sampling frequency in Hz
        for noise_type in [
            "ma",
            "bw",
            "em",
        ]:  # ma = muscle artifact, bw = baseline wander, em = electrode motion
            path = os.path.join(self.noise_dir, noise_type)  # Path to the noise record
            if not (
                os.path.exists(f"{path}.dat") and os.path.exists(f"{path}.hea")
            ):  # Check if the noise record exists
                raise FileNotFoundError(  # Raise an error if the noise record is not found
                    f"PhysioNet noise record {noise_type!r} not found in {self.noise_dir}. "
                    "Run preproc/setup_noise.py first or specify --physionet-noise-dir."
                )
            record = wfdb.rdrecord(path)  # Read the noise record
            raw_noise = record.p_signal[:, 0].astype(
                np.float32
            )  # Extract the noise signal
            resampled_noise = resample_poly(  # Resample the noise signal to match the ECG sampling frequency
                raw_noise, up=int(self.fs), down=original_fs
            )
            noise_banks[noise_type] = resampled_noise.astype(
                np.float32
            )  # Store the resampled noise in the noise bank
        return noise_banks

    def _get_scaled_noise(
        self, clean_lead: np.ndarray
    ) -> np.ndarray:  # Generate scaled noise for the ECG signal
        ecg_length = clean_lead.shape[-1]  # Get the length of the ECG signal
        composite_noise = np.zeros(
            ecg_length, dtype=np.float32
        )  # Initialize the composite noise signal
        for noise_type in ["ma", "bw", "em"]:  # Iterate over the noise types
            bank = self.noise_banks[
                noise_type
            ]  # Get the noise bank for the current noise type
            if (
                len(bank) <= ecg_length
            ):  # If the noise bank is shorter than the ECG signal, resize it
                noise_slice = np.resize(
                    bank, ecg_length
                )  # Resize the noise bank to match the ECG signal length
            else:  # If the noise bank is longer than the ECG signal, sample a random slice
                start_idx = np.random.randint(
                    0, len(bank) - ecg_length
                )  # Sample a random start index
                noise_slice = bank[
                    start_idx : start_idx + ecg_length
                ]  # Sample a random slice from the noise bank
            composite_noise += noise_slice  # Add the noise slice to the composite noise

        sig_power = np.mean(clean_lead**2)  # Compute the signal power
        noise_power = np.mean(composite_noise**2)  # Compute the noise power
        if (
            noise_power > 0
        ):  # If the noise power is greater than 0, scale the composite noise
            target_noise_power = (
                sig_power / self.snr_linear
            )  # Compute the target noise power
            scaling_factor = np.sqrt(
                target_noise_power / noise_power
            )  # Compute the scaling factor
            composite_noise *= scaling_factor  # Scale the composite noise
        return composite_noise

    def __call__(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Apply the PhysioNet noise augmentation to the input tensor
        x_batched, squeezed = _ensure_batched(x)  # Ensure the input tensor is batched
        batch_size, num_leads, seq_len = (
            x_batched.shape
        )  # Get the batch size, number of leads, and sequence length
        x_np = (
            x_batched.detach().cpu().numpy()
        )  # Convert the batched tensor to a NumPy array
        out_np = np.zeros_like(x_np)  # Initialize the output array

        for b in range(batch_size):  # Iterate over each batch
            for l in range(num_leads):  # Iterate over each lead in the batch
                noise = self._get_scaled_noise(
                    x_np[b, l]
                )  # Get the scaled noise for the ECG signal
                out_np[b, l] = (
                    x_np[b, l] + noise
                )  # Add the scaled noise to the ECG signal

        out = torch.from_numpy(out_np).to(
            device=x.device, dtype=x.dtype
        )  # Convert the output array to a PyTorch tensor
        return (
            out.squeeze(0) if squeezed else out
        )  # Return the augmented tensor, optionally squeezed


class PhysioNetTwoViewAugmentor:  # Augment an ECG signal into two views: filtered and noisy
    def __init__(  # Initialize the augmentor with the given noise directory, target SNR, and sampling frequency
        self,
        noise_dir: str = "physionet_data",  # Directory containing the noise files
        target_snr_db: float = 5.0,  # Target SNR in decibels
        fs: float = 500.0,  # Sampling frequency in Hz
    ) -> None:
        self.filter = ClinicalBandpassFilter(
            fs=fs
        )  # Initialize the clinical bandpass filter
        self.noise = PhysioNetNoise(  # Initialize the PhysioNet noise generator
            noise_dir=noise_dir,
            target_snr_db=target_snr_db,
            fs=fs,  # Pass the noise directory, target SNR, and sampling frequency to the PhysioNet noise generator
        )

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Apply the augmentor to the input tensor
        filtered = self.filter(
            x
        )  # Apply the clinical bandpass filter to the input tensor
        noisy = self.noise(
            filtered
        )  # Generate scaled noise for the filtered ECG signal
        return filtered, noisy


class TwoViewECGAugmentor:  # Augment an ECG signal into two views: filtered and noisy
    def __init__(
        self, transform: Compose | None = None
    ) -> None:  # Initialize the TwoViewECGAugmentor with an optional transform
        self.transform = (
            transform
            or Compose(  # Compose the transforms to apply to the input tensor
                [
                    RandomAmplitudeScale(),  # Randomly scale the amplitude of the input tensor
                    GaussianNoise(),  # Add Gaussian noise to the input tensor
                    # RandomLeadDropout(),
                    BaselineWander(),  # Add baseline wander to the input tensor
                ]
            )
        )

    def augment(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Augment the input tensor using the composed transforms
        return self.transform(x.clone())  # Return the augmented tensor

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # Return two augmented views of the input tensor
        return self.augment(x), self.augment(
            x
        )  # Return two augmented views of the input tensor


class TemporalSplitTwoViewAugmentor:  # Augment an ECG signal into two views: filtered and noisy by splitting the signal into two parts
    def __init__(  # Initialize the augmentor with a transform and split length
        self,
        transform: Compose | None = None,
        split_length: int = 2500,  # Length of the split in samples
    ) -> None:
        self.transform = (
            transform
            or Compose(  # Compose the transforms to apply to the input tensor
                [
                    RandomAmplitudeScale(),  # Randomly scale the amplitude of the signal
                    GaussianNoise(),  # Add Gaussian noise to the signal
                    BaselineWander(),  # Add baseline wander to the signal
                ]
            )
        )
        self.split_length = split_length  # Length of the split in samples

    def augment(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Augment the input tensor using the composed transforms
        return self.transform(x.clone())

    def __call__(  # Split the input tensor into two views and augment each view separately
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is expected to be [..., 2 * split_length] (e.g., 5000 samples for 10s at 500Hz)
        # We split into two consecutive, non-overlapping segments of split_length
        if x.size(-1) < 2 * self.split_length:
            # Fallback if signal is too short: return two augmented versions of the same (padded) segment
            x_view = x[..., : self.split_length]
            return self.augment(x_view), self.augment(x_view), x_view, x_view

        x1 = x[..., : self.split_length]  # First split of the signal
        x2 = x[
            ..., self.split_length : 2 * self.split_length
        ]  # Second split of the signal
        return self.augment(x1), self.augment(x2), x1, x2


class PhysioNetTemporalSplitTwoViewAugmentor:  # Augment an ECG signal into two views: filtered and noisy by splitting the signal into two parts
    def __init__(
        self,
        noise_dir: str = "physionet_data",
        target_snr_db: float = 5.0,  # Target signal-to-noise ratio in decibels
        fs: float = 500.0,  # Sampling frequency in Hz
        split_length: int = 2500,  # Length of each split in samples
    ) -> None:
        self.filter = ClinicalBandpassFilter(
            fs=fs
        )  # Clinical bandpass filter for ECG signal
        self.noise = PhysioNetNoise(  # PhysioNet noise augmentation
            noise_dir=noise_dir, target_snr_db=target_snr_db, fs=fs
        )
        self.split_length = split_length

    def __call__(  # Augment an ECG signal into two views: filtered and noisy by splitting the signal into two parts
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            x.size(-1) < 2 * self.split_length
        ):  # If the signal is shorter than 2 * split_length, pad it to split_length
            x_view = x[..., : self.split_length]  # Take the first split_length samples
            filtered = self.filter(x_view)  # Apply clinical bandpass filter to the view
            return (
                filtered,
                self.noise(filtered),
                x_view,
                x_view,
            )  # Return the filtered and noisy views, and the original view

        x1 = x[..., : self.split_length]  # Take the first split_length samples
        x2 = x[
            ..., self.split_length : 2 * self.split_length
        ]  # Take the second split_length samples

        # Apply clinical filter to both views first (Morphology preservation constraint)
        v1_clean = self.filter(x1)
        v2_clean = self.filter(x2)

        # View 1 is the "clean" denoised view (Diagnostic fidelity constraint)
        # View 2 is the "noisy" augmented view (Noise invariance constraint)
        v2_noisy = self.noise(v2_clean)

        return v1_clean, v2_noisy, x1, x2
