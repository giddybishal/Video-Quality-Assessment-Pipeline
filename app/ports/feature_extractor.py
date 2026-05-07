from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(self, video_path: str) -> np.ndarray:
        pass
     