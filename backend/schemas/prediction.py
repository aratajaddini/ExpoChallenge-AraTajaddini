from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    filename: str
    top_class: str
    confidence: float
    scores: dict[str, float]


class FrameResult(BaseModel):
    timestamp: float
    top_class: str
    confidence: float


class VideoPredictionResponse(PredictionResponse):
    frames_analyzed: int
    class_counts: dict[str, int]
    timeline: list[FrameResult]


class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    source: str
    created_at: str
