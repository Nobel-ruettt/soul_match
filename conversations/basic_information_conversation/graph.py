import uuid
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from nodes_and_routers import NodeAndRouters
from state import BasicInformationState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from IPython.display import Image, display

load_dotenv(override=True)  

class InitialConversationGraph:
    def __init__(self):
        self.graph = None
        self.memory = None
        self.conversation_id = str(uuid.uuid4())
        self.nodes_and_routers = NodeAndRouters()
    
    def setup(self):
        self.set_memory()
        self.build_graph()
        
    def set_memory(self):
        db_path = "memory.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.memory = SqliteSaver(conn)

    def build_graph(self):
        graph_builder = StateGraph(BasicInformationState)

        # Add Nodes
        graph_builder.add_node("success_met_node", self.nodes_and_routers.success_met_node)
        graph_builder.add_node("generate_next_question_node", self.nodes_and_routers.generate_next_question_node)
        graph_builder.add_node("generate_bye_message_node", self.nodes_and_routers.generate_bye_message_node)

        # Add Edges
        graph_builder.add_edge(START, "success_met_node")
        graph_builder.add_conditional_edges("success_met_node",self.nodes_and_routers.success_met_router, {
            "generate_next_question": "generate_next_question_node",
            "generate_bye_message": "generate_bye_message_node"
        })
        graph_builder.add_edge("generate_next_question_node", END)
        graph_builder.add_edge("generate_bye_message_node", END)

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def show_graph(self):
        if self.graph is None:
            print("Graph is not built yet. Please call setup() first.")
            return
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass
        
    def run_graph_initially(self):
        config = {"configurable": {"thread_id": self.conversation_id}}
        state = {
            "is_success_met": False,
        }
        result = self.graph.invoke(state, config=config)
        print(f"graph initial invocation result: {result}")
        return result["messages"][-1].content
    
    def run_graph(self, message):
        config = {"configurable": {"thread_id": self.conversation_id}}
        state = {
            "messages": {"role": "user", "content": message},
            "is_success_met": False,
        }
        result = self.graph.invoke(state, config=config)
        print(f"graph invocation result: {result}")
        return result["messages"][-1].content