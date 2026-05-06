import os
import cv2
import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import time

# -----------------------
# CONFIG
# -----------------------
VIDEO_FOLDER = "../KonVid/k150kb"
OUTPUT_FOLDER = "features"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------
# MODEL (2048 features)
# -----------------------
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# FRAME EXTRACTION
# -----------------------
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * max(total // num_frames, 1))
        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    cap.release()
    return frames

# -----------------------
# FEATURE EXTRACTION
# -----------------------
def video_to_feature(video_path):
    frames = extract_frames(video_path)

    frame_features = []

    with torch.no_grad():
        for f in frames:
            img = transform(f).unsqueeze(0)
            feat = model(img)
            feat = feat.view(-1)  # 2048
            frame_features.append(feat)

    video_feature = torch.stack(frame_features).mean(dim=0)
    return video_feature.numpy()

# -----------------------
# MAIN LOOP
# -----------------------
def process_all_videos():
    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if f.endswith((".mp4", ".avi", ".mov", ".gif"))]

    print(f"Found {len(video_files)} videos")

    start_time = time.time()

    for i, video in enumerate(tqdm(video_files, desc="Processing Videos")):
        path = os.path.join(VIDEO_FOLDER, video)

        feat = video_to_feature(path)

        save_path = os.path.join(OUTPUT_FOLDER, video + ".npy")
        np.save(save_path, feat)

        # Optional: print every 50 videos
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(video_files) - (i + 1))

            print(f"\nProcessed {i+1}/{len(video_files)}")
            print(f"Avg time/video: {avg_time:.2f}s")
            print(f"Estimated remaining: {remaining/60:.2f} minutes\n")

    print("Done extracting features.")

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    process_all_videos()
