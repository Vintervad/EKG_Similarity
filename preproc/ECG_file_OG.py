import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import DataLoader, Dataset


class ECGMultiNoiseDataset_OG(Dataset):
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

                # Apply Polyphase Resampling
                # resample_poly automatically calculates the greatest common divisor
                # so up=500, down=360 is perfectly optimized internally.
                resampled_noise = resample_poly(
                    raw_noise, up=self.fs, down=original_fs
                )  # Resample the noise to the ECG sampling rate

                self.noise_banks[noise_type] = resampled_noise.astype(
                    np.float32
                )  # Store the resampled noise in the noise bank

            except (
                FileNotFoundError
            ):  # If the noise file is not found, raise a FileNotFoundError
                raise FileNotFoundError(
                    f"Missing {noise_type}.dat or .hea in {noise_dir}"
                )

        # Calculate max starting index based on the NEW, longer 500Hz array length
        self.max_noise_start = len(self.noise_banks["ma"]) - self.ecg_length

    def _apply_bandpass(self, signal):
        """Applies zero-phase 0.016-150Hz filter to protect morphology."""
        return filtfilt(
            self.b, self.a, signal
        )  # Apply the bandpass filter to the signal using filtfilt

    def _get_scaled_noise(self, clean_ecg_signal):
        """Combines all three noise types, pulls slices, and scales composite to SNR."""
        # Initialize an empty array to hold the combined realistic noise
        composite_noise = np.zeros(self.ecg_length, dtype=np.float32)

        # A & B. Pull a random slice from ALL THREE banks and add them together
        for noise_type in [
            "ma",
            "bw",
            "em",
        ]:  # ma = muscle artifact, bw = baseline wander, em = electrode motion
            active_bank = self.noise_banks[
                noise_type
            ]  # Get the noise bank for this type
            # Pick a unique random starting point for each noise type so they don't align artificially
            start_idx = np.random.randint(0, self.max_noise_start)
            noise_slice = active_bank[  # Slice the noise bank to match the ECG length
                start_idx : start_idx + self.ecg_length
            ]

            composite_noise += noise_slice  # Add this specific noise to the composite

        # C. Calculate Power and Scale on the COMPOSITE noise
        sig_power = np.mean(clean_ecg_signal**2)  # Calculate the signal power
        noise_power = np.mean(composite_noise**2)  # Calculate the combined noise power

        if (
            noise_power == 0
        ):  # If the noise power is 0, return the noise slice without scaling
            return composite_noise

        target_noise_power = (
            sig_power / self.snr_linear
        )  # Calculate the target noise power based on SNR
        scaling_factor = np.sqrt(
            target_noise_power / noise_power
        )  # Calculate the scaling factor based on the target noise power and actual noise power

        return (
            composite_noise * scaling_factor
        )  # Scale the composite noise slice and return it

    def __getitem__(self, idx):
        # 1. Load the actual Parquet file
        file_path = self.ecg_paths[idx]
        df = pd.read_parquet(file_path)

        # 2. Extract the 12 leads into a NumPy array
        # IMPORTANT: Change these column names if your Parquet file uses different headers!
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

        try:
            # Extract the data. Shape becomes [Time, Leads] (e.g., [5000, 12])
            raw_data = df[lead_columns].values
        except KeyError as e:
            raise KeyError(
                f"Parquet file {file_path} is missing expected lead columns: {e}"
            )

        # Extract the data and transpose to [Leads, Time]
        clean_ecg = raw_data.T.astype(np.float32)

        # --- THE FIX ---
        # Convert microvolts (uV) to millivolts (mV)
        clean_ecg = clean_ecg / 1000.0

        # 3. Handle Length Discrepancies (Safety Check)
        # If the file is shorter than 5000, pad it with zeros.
        # If it's longer than 5000, truncate it.
        actual_length = clean_ecg.shape[1]
        if actual_length < self.ecg_length:
            padding = np.zeros((12, self.ecg_length - actual_length), dtype=np.float32)
            clean_ecg = np.hstack((clean_ecg, padding))
        elif actual_length > self.ecg_length:
            clean_ecg = clean_ecg[:, : self.ecg_length]

        filtered_ecg = np.zeros_like(clean_ecg)
        noisy_ecg = np.zeros_like(clean_ecg)

        # 4. Process each of the 12 leads through the clinical filter and noise bank
        for lead in range(12):
            # A: Apply clinical filter FIRST (0.05 - 150 Hz)
            filtered_lead = self._apply_bandpass(clean_ecg[lead])
            filtered_ecg[lead] = filtered_lead

            # B: Get exact scaled noise and add it to the FILTERED signal
            scaled_noise = self._get_scaled_noise(filtered_lead)
            noisy_ecg[lead] = filtered_lead + scaled_noise

        # View 1: Clean, Filtered ECG (X^1)
        x1 = torch.tensor(filtered_ecg, dtype=torch.float32)

        # View 2: Noisy Augmented ECG (X^2)
        x2 = torch.tensor(noisy_ecg, dtype=torch.float32)

        return x1, x2

    def __len__(self):
        return len(self.ecg_paths)


# --- EXECUTION TEST ---
# if __name__ == "__main__":
#     # Point this to your real parquet file!
#     test_files = [".parquet"]

#     # Initialize your dataset (Make sure target_snr_db is set, e.g., 5.0)
#     dataset = ECGMultiNoiseDataset_OG(ecg_paths=test_files, target_snr_db=5.0)

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
