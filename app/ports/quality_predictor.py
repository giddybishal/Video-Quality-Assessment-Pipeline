from abc import ABC, abstractmethod
import numpy as np


class QualityPredictor(ABC):

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        pass
    