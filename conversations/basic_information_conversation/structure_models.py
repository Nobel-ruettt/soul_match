from pydantic import BaseModel, Field


class SuccessMetOutput(BaseModel):
    feedback: str = Field(description="Feedback on whether the success criteria have been met")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")