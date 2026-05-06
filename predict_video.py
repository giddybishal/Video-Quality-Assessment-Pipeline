import torch
import cv2
import numpy as np
import joblib
import torchvision.models as models
from torchvision import transforms

# -----------------------
# LOAD MODEL + SCALER
# -----------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------
# LOAD RESNET (same as before)
# -----------------------
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

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
    feats = []

    with torch.no_grad():
        for f in frames:
            img = transform(f).unsqueeze(0)
            feat = resnet(img)
            feat = feat.view(-1)
            feats.append(feat)

    return torch.stack(feats).mean(dim=0).numpy()

# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(video_path):
    feat = video_to_feature(video_path)

    feat = scaler.transform([feat])   # IMPORTANT
    score = model.predict(feat)[0]

    return score

# -----------------------
# TEST
# -----------------------
if __name__ == "__main__":
    video_path = r"C:\Users\Bishal\Downloads\6708571-uhd_2160_3840_24fps.mp4"  # change this

    score = predict(video_path)

    print(f"Predicted MOS: {score:.2f}")
    