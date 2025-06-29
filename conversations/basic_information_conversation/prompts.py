from typing import Any, List
from state import BasicInformationState
from langchain_core.messages import AIMessage, HumanMessage

# Helper function to format the conversation history for display
def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content 
            conversation += f"Assistant: {text}\n"
    return conversation


def success_criteria_met_system_prompt() -> str:
    message = f""" you will be given a list of messages that are exchanged between the user 
    and the system.you will determine based on the conversation if success criteria met or not.
    please read the conversation and if you think the success criteria is met, return True, 
    otherwise return False.
    Also give a feedback on how to meet the success criteria if it is not met.
    if the success criteria is met, return an empty string as feedback.
    Also if there is no conversation, return False and return start the conversation as feedback.

    The success criteria is:

    You are the user's friendly companion. Your goal is to collect some
    basic information from the user: their Name, Age, Gender, and City (Location). 
    Ask one question at a time in a calm and friendly manner, as if you are their friend. 
    If the user provides a correct and reasonable answer, appreciate their response. 
    If the user gives an answer in the wrong format or something impossible,
    gently remind them that you need truthful and accurate information. "
    Continue the conversation until you have collected all the required details. 
    """
    return message

def success_criteria_met_user_prompt(state: BasicInformationState) -> str:

    user_message = f""" The entire conversation is as follows:
    {format_conversation(state["messages"])}
    """
    return user_message

def root_system_message_for_generate() -> str:
    prompt = f"""You are the user's friendly companion. Your goal is to collect some
        basic information from the user: their Name, Age, Gender, and City (Location). 
        Ask one question at a time in a calm and friendly manner, as if you are their friend. 
        If the user provides a correct and reasonable answer, appreciate their response. 
        If the user gives an answer in the wrong format or something impossible,
        gently remind them that you need truthful and accurate information. "
        Continue the conversation until you have collected all the required details. 
        You will ask the first question to the user and start the conversation if there 
        is no conversation history.
        """
    return prompt

def bye_system_message_to_generate(state: BasicInformationState) -> str:
    conversation_history = format_conversation(state["messages"])
    prompt = f"""Based on the conversation so far:
    {conversation_history}
    Thank you for sharing your information! It was great talking with you.
    Congratulations on completing this part of our chat.
    If you need anything else or want to talk again later, feel free to reach out.
    Have a wonderful day!"""
    return prompt
