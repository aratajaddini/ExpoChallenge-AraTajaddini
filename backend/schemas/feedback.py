"""Request and response schemas for the feedback endpoint."""

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """User correction for a stored prediction."""

    prediction_id: int = Field(gt=0)
    correct_class: str = Field(min_length=1, max_length=64)


class FeedbackResponse(BaseModel):
    """Stored feedback record."""

    id: int
    prediction_id: int
    correct_class: str
