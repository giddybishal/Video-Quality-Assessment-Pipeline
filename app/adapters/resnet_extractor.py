import torch
import cv2
import numpy as np

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from app.ports.feature_extractor import FeatureExtractor


class ResNetFeatureExtractor(FeatureExtractor):

    def __init__(self):

        weights = ResNet50_Weights.DEFAULT

        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_frames(self, video_path, num_frames=10):

        cap = cv2.VideoCapture(video_path)

        frames = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(num_frames):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                i * max(total // num_frames, 1)
            )

            ret, frame = cap.read()

            if ret:
                frames.append(frame)

        cap.release()

        return frames

    def extract(self, video_path: str) -> np.ndarray:

        frames = self.extract_frames(video_path)

        feats = []

        with torch.no_grad():

            for frame in frames:

                img = self.transform(frame).unsqueeze(0)

                feat = self.model(img)

                feat = feat.view(-1)

                feats.append(feat)

        return torch.stack(feats).mean(dim=0).numpy()
    