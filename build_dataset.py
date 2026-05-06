import os
import numpy as np
import pandas as pd

FEATURE_FOLDER = "features"
CSV_PATH = "../KonVid/k150kb_scores.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

X = []
y = []
used_names = []

missing = 0

for _, row in df.iterrows():
    video_name = row["video_name"]
    mos = row["mos"]

    feature_file = os.path.join(FEATURE_FOLDER, video_name + ".npy")

    if os.path.exists(feature_file):
        feat = np.load(feature_file)

        X.append(feat)
        y.append(mos)
        used_names.append(video_name)
    else:
        missing += 1

print(f"Matched videos: {len(X)}")
print(f"Missing features: {missing}")

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Save dataset (IMPORTANT)
np.save("X.npy", X)
np.save("y.npy", y)
