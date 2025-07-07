from pydantic import BaseModel, Field


class MessageEvaluationOutput(BaseModel):
    feedback: str = Field(description="Feedback on the previous generated message")
    is_approved: bool = Field(description="Whether the message is approved or not")

class ConversationCompletionOutput(BaseModel):
    is_completed: bool = Field(description="Whether the conversation is enough for the personality trait or not")