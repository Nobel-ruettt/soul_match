from typing import Annotated, List, Any, TypedDict
from langgraph.graph.message import add_messages

class BasicInformationState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    feedback: str
    is_success_met: bool

