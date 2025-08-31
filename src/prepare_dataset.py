import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

RAW_PATH = Path("data/raw/merged_dataset.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_dataset():
    # 1. Load merged dataset
    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    # 2. Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 3. Fill missing text safely
    if "bio" in df.columns:
        df["bio"] = df["bio"].fillna("").astype(str)
    else:
        df["bio"] = ""

    if "username" in df.columns:
        df["username"] = df["username"].fillna("").astype(str)
    else:
        df["username"] = ""

    # 4. Feature engineering
    df["bio_len"] = df["bio"].str.len()
    df["username_len"] = df["username"].str.len()
    df["username_digits"] = df["username"].str.count(r"\d")

    if "following" in df.columns and "followers" in df.columns:
        df["ff_ratio"] = np.where(df["following"] != 0,
                                  df["followers"] / df["following"], 0)

    # Binary flags
    if "profile_pic" in df.columns:
        df["profile_pic"] = df["profile_pic"].map({"Yes":1, "No":0, True:1, False:0})

    if "private" in df.columns:
        df["private"] = df["private"].map({"Yes":1, "No":0, True:1, False:0})

    # 5. Features & Target
    features = ["followers", "following", "media_count",
                "bio_len", "username_len", "username_digits",
                "ff_ratio", "profile_pic", "private"]

    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df["label"].map({"fake":0, "real":1})

    # 6. Scale numeric features
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 8. Save
    X.to_csv(PROCESSED_DIR/"clean_full.csv", index=False)
    X_train.to_csv(PROCESSED_DIR/"X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR/"X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR/"y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR/"y_test.csv", index=False)

    print("✅ Processed dataset saved in data/processed/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_dataset()
