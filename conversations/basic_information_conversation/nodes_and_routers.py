from typing import Any, Dict
from langchain_openai import ChatOpenAI
from prompts import bye_system_message_to_generate, root_system_message_for_generate, success_criteria_met_system_prompt, success_criteria_met_user_prompt
from state import BasicInformationState
from langchain_core.messages import  HumanMessage, SystemMessage
from structure_models import SuccessMetOutput

class NodeAndRouters:
    def success_met_node(self, state: BasicInformationState) -> BasicInformationState:
        system_message = success_criteria_met_system_prompt()
        user_message = success_criteria_met_user_prompt(state)
        evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        llm = ChatOpenAI(model="gpt-4o-mini")
        evaluator_llm_with_response = llm.with_structured_output(SuccessMetOutput)
        eval_result = evaluator_llm_with_response.invoke(evaluator_messages)
        print(f"success_met_node. Evaluation Result: {eval_result}")
        new_state = {
            "is_success_met": eval_result.success_criteria_met,
            "feedback": eval_result.feedback
        }
        return new_state
    
    def generate_next_question_node(self, state: BasicInformationState) ->  Dict[str, Any]:
        system_message = root_system_message_for_generate()

        messages = state["messages"]
        isInitialMessage = False
        if(len(messages) == 0):
            isInitialMessage = True
            messages = [SystemMessage(content=system_message)] + messages

        print(f"generate_next_question_node. Messages: {messages}")
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(messages)

        if isInitialMessage:
            return {
                "messages": messages+[response],
            }

        return {
            "messages": [response],
        }
    
    def generate_bye_message_node(self, state: BasicInformationState) -> Dict[str, Any]:
        system_message = bye_system_message_to_generate(state)
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(system_message)
        return {
            "messages": [response],
        }
    
    def success_met_router(self, state: BasicInformationState) -> str:
        if state["is_success_met"]:
            return "generate_bye_message"
        else:
            return "generate_next_question"
    
    
    
    
        
        