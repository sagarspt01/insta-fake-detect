import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
MERGED_FILE = RAW_DIR / "merged_dataset.csv"

def merge_json():
    # Load fake and real JSON files
    fake = pd.read_json(RAW_DIR / "fakeAccountData.json", orient="records")
    real = pd.read_json(RAW_DIR / "realAccountData.json", orient="records")

    # Add labels
    fake["label"] = "fake"
    real["label"] = "real"

    # Merge both datasets
    df = pd.concat([fake, real], ignore_index=True)

    # Save merged dataset
    df.to_csv(MERGED_FILE, index=False)

    print("✅ Merged dataset created:", MERGED_FILE)
    print("Shape:", df.shape)
    print("Class distribution:\n", df["label"].value_counts())

    return df

if __name__ == "__main__":
    merge_json()
