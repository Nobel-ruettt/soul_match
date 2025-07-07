from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from structure_models import ConversationCompletionOutput, MessageEvaluationOutput
from prompts import generate_facetwise_next_message, generate_intial_message_system_prompt, generate_prompt_for_facetwise_conversation_finished, generate_prompt_for_facetwise_message_feedback, generate_prompt_for_facetwise_summary
from state import PersonalityTraitsState

class NodeAndRouters:
    #personality_traits = ["Imagination","Artistic Interests","Emotionality","Adventurousness","Intellect","Liberalism"]
    personality_traits = ["Imagination","Artistic Interests"]

    ### Nodes
    def generate_initial_message_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:
        system_message = generate_intial_message_system_prompt()

        self.print_debug_info(
            "generate_initial_message_node",
            f"SYSTEM PROMPT {system_message}\n")
        
        messages = [SystemMessage(content=system_message)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(messages)

        self.print_debug_info(
            "generate_initial_message_node",
            f"AI response {response}\n")
        
        return {
            "messages": [response],
        }

    def initialize_state_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:
        facetwise_conversations = {trait: [] for trait in self.personality_traits}
        initial_state = {
            "personality_traits_covered": [],
            "current_user_message": "",
            "current_personality_trait": self.personality_traits[0],
            "is_facetwise_conversation_finished": False,
            "facetwise_conversations": facetwise_conversations,
            "message_feedback": "",
            "is_message_approved": True,
            "is_all_objective_completed": False,
        }
        return initial_state
    
    def generate_facetwise_message_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:

        self.print_state("generate_facetwise_message_node", state)

        current_personality_trait = state.get("current_personality_trait")
        facetwise_conversations = state.get("facetwise_conversations")
        current_facetwise_conversation = facetwise_conversations[current_personality_trait]
        current_user_message = state.get("current_user_message", "")
        
        system_prompt = generate_facetwise_next_message(state)

        self.print_debug_info("generate_facetwise_message_node",
                              f"system prompt {system_prompt}\n")
         
        found_system_message = False
        for message in current_facetwise_conversation:
            if isinstance(message, SystemMessage):
                found_system_message = True
                message.content = system_prompt
                
        if not found_system_message:
            current_facetwise_conversation = [SystemMessage(content=system_prompt)]
        else:
            if len(current_user_message) > 0:
                current_facetwise_conversation = current_facetwise_conversation + [HumanMessage(content=current_user_message)]
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(current_facetwise_conversation)

        self.print_debug_info("generate_facetwise_message_node",
                              f"AI Response {response.content}\n")

        current_facetwise_conversation = current_facetwise_conversation + [response]
        facetwise_conversations[current_personality_trait] = current_facetwise_conversation

        return {
            "messages": [response],
            "facetwise_conversations": facetwise_conversations,
            "is_message_approved": True,
            "current_user_message": "",
            "message_feedback": "",
        }
    
    
    def facetwise_message_feedback_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:

        self.print_state("facetwise_message_feedback_node", state)

        system_prompt = generate_prompt_for_facetwise_message_feedback(state)

        self.print_debug_info("facetwise_message_feedback_node",
                              f"SYSTEM PROMPT {system_prompt}\n")
        
        evaluator_messages = [SystemMessage(content=system_prompt)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        evaluator_llm_with_output = llm.with_structured_output(MessageEvaluationOutput) 
        eval_result = evaluator_llm_with_output.invoke(evaluator_messages)

        self.print_debug_info("facetwise_message_feedback_node",
                              f"EVAL RESULT {eval_result}\n")
        
        return{
            "is_message_approved": eval_result.is_approved,
            "message_feedback": eval_result.feedback,
        }
    
    def is_facetwise_conversation_finished_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:

        self.print_state("is_facetwise_conversation_finished_node", state)

        system_prompt = generate_prompt_for_facetwise_conversation_finished(state)
        evaluator_messages = [SystemMessage(content=system_prompt)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        evaluator_llm_with_output = llm.with_structured_output(ConversationCompletionOutput) 
        eval_result = evaluator_llm_with_output.invoke(evaluator_messages)
        
        return {
            "is_facetwise_conversation_finished": eval_result.is_completed,
        }
    
    def generate_facetwise_marking_and_summary_node(self, state: PersonalityTraitsState) -> PersonalityTraitsState:
        # self.print_state("generate_facetwise_marking_and_summary_node", state)
        system_prompt = generate_prompt_for_facetwise_summary(state)
        messages = [SystemMessage(content=system_prompt)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_with_tools = llm.bind_tools(self.tools)
        llm_with_tools.invoke(messages)

        current_personality_trait = state.get("current_personality_trait")
        personality_traits_covered = state.get("personality_traits_covered")
        personality_traits_covered.append(current_personality_trait)

        next_personality_trait_index = self.personality_traits.index(current_personality_trait) + 1
        next_personality_trait = None
        if next_personality_trait_index < len(self.personality_traits):
            next_personality_trait = self.personality_traits[next_personality_trait_index]
        else:
            next_personality_trait = self.personality_traits[0]
        return {
            "current_personality_trait" : next_personality_trait,
            "personality_traits_covered": personality_traits_covered,
            "is_facetwise_conversation_finished": False,
            "message_feedback": "",
            "is_message_approved": True,
            "is_all_objective_completed": False,
        }
    
    ### Routers
    def initial_conversation_router(self, state: PersonalityTraitsState) -> str:
        messages = state["messages"]
        if(messages is None or len(messages) == 0):
            return "YES"
        else:
            return "NO"
    def facetwise_evaluation_router(self, state: PersonalityTraitsState) -> str:
        # print(f"---------------facetwise_evaluation_router. state: {state}------------\n")
        is_message_approved = state.get("is_message_approved")
        if is_message_approved:
            return "APPROVED"
        else:
            return "NOT_APPROVED"
        
    def facetwise_conversation_finished_router(self, state: PersonalityTraitsState) -> str:
        is_facetwise_conversation_finished = state.get("is_facetwise_conversation_finished")
        if is_facetwise_conversation_finished:
            return "YES"
        else:
            return "NO"
        
    
    def print_state(self, nodeName:str, state: PersonalityTraitsState):
        self.print_to_file(f"---------------STATE FOR NODE {nodeName}------------\n")
        messages = state.get("messages",[])
        current_personality_trait = state.get('current_personality_trait',"")
        facetwise_conversations = state.get('facetwise_conversations')
        current_facetwise_conversation = facetwise_conversations[current_personality_trait]
        self.print_to_file(f"last message {messages[-1].content}\n")
        self.print_to_file(f"personality_traits_covered {state.get('personality_traits_covered')}\n")
        self.print_to_file(f"current_personality_trait {current_personality_trait}\n")
        self.print_to_file(f"is_message_approved {state.get('is_message_approved')}\n")
        self.print_to_file(f"message_feedback {state.get('message_feedback')}\n")
        self.print_to_file(f"is_facetwise_conversation_finished {state.get('is_facetwise_conversation_finished')}\n")
        if len(current_facetwise_conversation) > 0:
            self.print_to_file(f"current facetwise conversation last message {current_facetwise_conversation[-1].content}\n")
        self.print_to_file(f"-----------------------------------------------------\n")
    
    def print_debug_info(self,nodeName, info: str):
        self.print_to_file(f"---------------DEBUG FOR NODE {nodeName}------------\n")
        self.print_to_file(f"{info}\n")
        self.print_to_file(f"----------------------------------------------------\n")
        

    
    def print_to_file(self, output: str, filename: str = "output.txt"):
        """Write the given output string to a text file."""
        with open(filename, "a", encoding="utf-8") as f:
            f.write(output)



    
    
    
    
        
        