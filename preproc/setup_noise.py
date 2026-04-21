import os

import wfdb


def download_nstdb_noise(target_dir="./physionet_data"):
    """Downloads the required pure noise records from PhysioNet."""
    os.makedirs(target_dir, exist_ok=True)

    records_to_get = [
        "ma",
        "bw",
        "em",
    ]  # ma = muscle artifact, bw = baseline wander, em = electrode motion
    missing_records = []

    # Check which files are missing
    for rec in records_to_get:
        if not (
            os.path.exists(f"{target_dir}/{rec}.dat")
            and os.path.exists(f"{target_dir}/{rec}.hea")
        ):
            missing_records.append(rec)

    if not missing_records:
        print(f"All noise files already exist in {target_dir}. You are ready to train!")
        return

    print(f"Downloading missing records {missing_records} to {target_dir}...")

    # Download only the missing files directly from PhysioNet
    wfdb.dl_database("nstdb", dl_dir=target_dir, records=missing_records)
    print("Download complete. Noise bank is ready.")


if __name__ == "__main__":
    download_nstdb_noise()
