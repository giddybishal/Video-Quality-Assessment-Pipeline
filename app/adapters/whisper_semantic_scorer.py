import whisper

from sentence_transformers import SentenceTransformer, util

from app.ports.semantic_scorer import SemanticScorer


class WhisperSemanticScorer(SemanticScorer):

    def __init__(self):

        self.whisper_model = whisper.load_model("base")

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def transcribe(self, video_path: str) -> str:

        result = self.whisper_model.transcribe(video_path)

        return result["text"]

    def score(
        self,
        video_path: str,
        campaign_text: str
    ) -> float:

        transcript = self.transcribe(video_path)

        transcript_embedding = self.embedder.encode(
            transcript,
            convert_to_tensor=True
        )

        campaign_embedding = self.embedder.encode(
            campaign_text,
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            transcript_embedding,
            campaign_embedding
        ).item()

        similarity = (similarity + 1) / 2

        final_score = similarity

        final_score = max(0.0, min(final_score, 1.0))

        return final_score
    