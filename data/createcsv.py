import os

import pandas as pd


def generate_csvs():
    # Get the current working directory to be sure where we are
    print(f"🔍 Current Working Directory: {os.getcwd()}")

    base_raw_path = "data/raw"
    metadata_path = "data/metadata"

    if not os.path.exists(base_raw_path):
        print(
            f"❌ ERROR: The folder '{base_raw_path}' does not exist in this directory."
        )
        return

    os.makedirs(metadata_path, exist_ok=True)

    splits = ["train", "val", "test"]
    valid_extensions = (".npy", ".pt", ".pth")

    for split in splits:
        current_dir = os.path.join(base_raw_path, split)
        print(f"--- Checking split: {split} ---")

        if not os.path.exists(current_dir):
            print(f"⚠️  Directory not found: {current_dir}")
            continue

        files = [f for f in os.listdir(current_dir) if f.endswith(valid_extensions)]
        print(f"📂 Found {len(files)} valid files in {current_dir}")

        data = []
        for filename in sorted(files):
            file_id = os.path.splitext(filename)[0]
            relative_path = os.path.join(current_dir, filename)
            data.append({"id": file_id, "path": relative_path})

        if data:
            df = pd.DataFrame(data)
            output_file = os.path.join(metadata_path, f"{split}.csv")
            df.to_csv(output_file, index=False)
            print(f"✅ SUCCESS: Created {output_file}")
        else:
            print(
                f"ℹ️  No files added for {split} (folder might be empty or wrong extension)."
            )


if __name__ == "__main__":
    generate_csvs()
