from typing import Annotated, List, Any, TypedDict
from langgraph.graph.message import add_messages

class PersonalityTraitsState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    personality_traits_covered: List[str]
    current_personality_trait: str
    is_message_approved: bool
    message_feedback: str
    current_user_message: str
    is_facetwise_conversation_finished: bool
    facetwise_conversations: dict[str, List[Any]]
    is_all_objective_completed: bool
