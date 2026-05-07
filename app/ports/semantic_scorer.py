from abc import ABC, abstractmethod

class SemanticScorer(ABC):
    @abstractmethod
    def score(self, video_path: str, campaign_text: str) -> float:
        pass
    