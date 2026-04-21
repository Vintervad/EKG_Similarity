import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import DataLoader, Dataset

# The Problem: Because peak times and amplitudes are inherently dependent on the specific lead, the physical principles governing noise distribution are similarly connected.
# A patient breathing (baseline wander) or shifting their body (muscle artifact) does not affect Lead V1 in isolation;
# it creates a spatially correlated artifact across the entire 3D volume of the chest.
# If the model sees 12 completely independent, uncorrelated noises, it learns a physical impossibility.

# The Fix: Pull a single random noise slice per View.
# Then, generate a random vector of 12 weights (to simulate the electrical axis of the noise) and multiply your single noise slice by those weights before adding it to the 12 leads.
# This maintains realistic spatial correlation.


class ECGMultiNoiseDataset2(Dataset):
    def __init__(
        self, ecg_paths, noise_dir=".", target_snr_db=5.0, ecg_length=5000, fs=500
    ):
        self.ecg_paths = ecg_paths
        self.target_snr_db = target_snr_db
        self.ecg_length = ecg_length
        self.fs = fs
        self.snr_linear = 10 ** (self.target_snr_db / 10.0)

        # 1. Initialize the Butterworth Bandpass Filter (0.05 to 150 Hz)
        nyquist = 0.5 * self.fs  # Nyquist frequency is half the sampling rate
        low = 0.016 / nyquist  # Low cutoff frequency in Hz
        high = 150.0 / nyquist  # High cutoff frequency in Hz
        self.b, self.a = butter(
            3, [low, high], btype="band"
        )  # 3rd-order Butterworth bandpass filter

        # 2. Load and Resample the NSTDB Noise Banks
        print(f"Loading and Resampling NSTDB Noise from 360Hz to {self.fs}Hz...")
        self.noise_banks = {}  # Dictionary to store the resampled noise banks

        original_fs = 360  # Original sampling rate of the NSTDB noise files

        for noise_type in ["ma", "bw", "em"]:  # Iterate over the noise types
            try:  # Try to load and resample the noise file
                record = wfdb.rdrecord(
                    os.path.join(noise_dir, noise_type)
                )  # Load the noise file
                raw_noise = record.p_signal[:, 0].astype(
                    np.float32
                )  # Extract the first channel and convert to float32

                resampled_noise = resample_poly(
                    raw_noise, up=self.fs, down=original_fs
                )  # Resample the noise to the ECG sampling rate

                self.noise_banks[noise_type] = resampled_noise.astype(
                    np.float32
                )  # Store the resampled noise in the noise bank

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing {noise_type}.dat or .hea in {noise_dir}"
                )

        self.max_noise_start = len(self.noise_banks["ma"]) - self.ecg_length

    def _apply_bandpass(self, signal):
        """Applies zero-phase 0.05-150Hz filter to protect morphology."""
        return filtfilt(
            self.b, self.a, signal
        )  # Apply the bandpass filter to the signal using filtfilt

    # [FIX: SPATIAL DISCONNECT] - Function now accepts pre-determined noise indices
    def _get_spatially_coherent_noise(self, clean_ecg_signal, indices):
        """Combines all three noise types using shared indices across all leads."""
        composite_noise = np.zeros(self.ecg_length, dtype=np.float32)

        for i, noise_type in enumerate(["ma", "bw", "em"]):
            active_bank = self.noise_banks[
                noise_type
            ]  # Get the noise bank for this type
            start_idx = indices[i]  # Use the shared index for this specific patient
            noise_slice = active_bank[start_idx : start_idx + self.ecg_length]
            composite_noise += noise_slice  # Add this specific noise to the composite

        # C. Calculate Power and Scale on the COMPOSITE noise
        sig_power = np.mean(clean_ecg_signal**2)  # Calculate the signal power
        noise_power = np.mean(composite_noise**2)  # Calculate the combined noise power

        if noise_power == 0:
            return composite_noise

        target_noise_power = sig_power / self.snr_linear
        scaling_factor = np.sqrt(target_noise_power / noise_power)

        return composite_noise * scaling_factor

    def __getitem__(self, idx):
        # 1. Load the actual Parquet file
        file_path = self.ecg_paths[idx]
        df = pd.read_parquet(file_path)

        # 2. Extract the 12 leads into a NumPy array
        lead_columns = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        raw_data = df[lead_columns].values
        clean_ecg = raw_data.T.astype(np.float32)

        # Convert microvolts (uV) to millivolts (mV)
        clean_ecg = clean_ecg / 1000.0

        # 3. Handle Length Discrepancies
        actual_length = clean_ecg.shape[1]
        if actual_length < self.ecg_length:
            padding = np.zeros((12, self.ecg_length - actual_length), dtype=np.float32)
            clean_ecg = np.hstack((clean_ecg, padding))
        elif actual_length > self.ecg_length:
            clean_ecg = clean_ecg[:, : self.ecg_length]

        filtered_ecg = np.zeros_like(clean_ecg)
        noisy_ecg = np.zeros_like(clean_ecg)

        # [FIX: SPATIAL DISCONNECT] - Generate ONE set of temporal indices and spatial weights per patient
        shared_indices = [np.random.randint(0, self.max_noise_start) for _ in range(3)]
        spatial_weights = np.random.uniform(0.5, 1.5, size=12)

        # 4. Process each of the 12 leads
        for lead in range(12):
            # A: Apply clinical filter FIRST
            filtered_lead = self._apply_bandpass(clean_ecg[lead])
            filtered_ecg[lead] = filtered_lead

            # B: Get exact scaled noise using SHARED indices, then multiply by spatial weight
            scaled_noise = self._get_spatially_coherent_noise(
                filtered_lead, shared_indices
            )
            noisy_ecg[lead] = filtered_lead + (scaled_noise * spatial_weights[lead])

        # View 1: Clean, Filtered ECG (X^1)
        x1 = torch.tensor(filtered_ecg, dtype=torch.float32)

        # View 2: Noisy Augmented ECG (X^2)
        x2 = torch.tensor(noisy_ecg, dtype=torch.float32)

        return x1, x2

    def __len__(self):
        return len(self.ecg_paths)


# # --- EXECUTION TEST ---
# if __name__ == "__main__":
#     # Point this to your real parquet file!
#     test_files = [".parquet"]

#     # Initialize your dataset (Make sure target_snr_db is set, e.g., 5.0)
#     dataset = ECGMultiNoiseDataset(ecg_paths=test_files, target_snr_db=5.0)

#     # Pull the first (and only) item
#     v1_clean, v2_noisy = dataset[0]

#     # Let's plot Lead I (Index 0) to visually confirm it works
#     time_axis = np.arange(5000) / 500.0  # 10 seconds at 500Hz

#     plt.figure(figsize=(12, 6))
#     plt.plot(
#         time_axis, v1_clean[0].numpy(), label="View 1: Filtered Clean", linewidth=2
#     )
#     plt.plot(
#         time_axis, v2_noisy[0].numpy(), label="View 2: Noisy Augmentation", alpha=0.7
#     )
#     plt.title("Single Patient Pipeline Test: Lead I", fontweight="bold")
#     plt.xlabel("Time (Seconds)")
#     plt.ylabel("Amplitude (mV)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
