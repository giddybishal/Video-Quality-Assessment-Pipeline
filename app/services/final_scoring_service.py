class FinalScoringService:

    def __init__(self, vqa_service, semantic_scorer):
        self.vqa_service = vqa_service
        self.semantic_scorer = semantic_scorer

    def score(self, video_path, campaign_text):

        vqa_score = self.vqa_service.score(video_path)

        semantic_score = self.semantic_scorer.score(
            video_path,
            campaign_text
        )

        normalized_vqa = vqa_score / 5.0

        final_score = (
            0.6 * normalized_vqa +
            0.4 * semantic_score
        )

        return {
            "vqa_score": vqa_score,
            "semantic_score": semantic_score,
            "final_score": final_score * 5
        }
    