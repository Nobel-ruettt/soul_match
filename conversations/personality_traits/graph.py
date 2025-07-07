import uuid
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from state import PersonalityTraitsState
from nodes_and_routers import NodeAndRouters
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from IPython.display import Image, display

load_dotenv(override=True)  

class PersonalityTraitsGraph:
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
        graph_builder = StateGraph(PersonalityTraitsState)

        # Add Nodes
        graph_builder.add_node("generate_initial_message_node",
                                self.nodes_and_routers.generate_initial_message_node)
        graph_builder.add_node("initialize_state_node",
                                self.nodes_and_routers.initialize_state_node)
        graph_builder.add_node("generate_facetwise_message_node",
                                self.nodes_and_routers.generate_facetwise_message_node)
        graph_builder.add_node("facetwise_message_feedback_node",
                                self.nodes_and_routers.facetwise_message_feedback_node)
        graph_builder.add_node("is_facetwise_conversation_finished_node",
                               self.nodes_and_routers.is_facetwise_conversation_finished_node)
        graph_builder.add_node("generate_facetwise_marking_and_summary_node",
                                self.nodes_and_routers.generate_facetwise_marking_and_summary_node)
        
        # Add Edges
        graph_builder.add_conditional_edges(START, 
                                            self.nodes_and_routers.initial_conversation_router,
                                            {
                                                "YES": "generate_initial_message_node",
                                                "NO": "generate_facetwise_message_node"
                                            })
        graph_builder.add_edge("generate_initial_message_node",
                                "initialize_state_node")
        graph_builder.add_edge("initialize_state_node", END)


        graph_builder.add_edge("generate_facetwise_message_node",
                                "facetwise_message_feedback_node")
        graph_builder.add_conditional_edges("facetwise_message_feedback_node",
                                            self.nodes_and_routers.facetwise_evaluation_router,
                                            {
                                                "APPROVED": "is_facetwise_conversation_finished_node",
                                                "NOT_APPROVED": "generate_facetwise_message_node"
                                            })
        
        graph_builder.add_conditional_edges("is_facetwise_conversation_finished_node",
                                            self.nodes_and_routers.facetwise_conversation_finished_router,
                                            {
                                                "YES": "generate_facetwise_marking_and_summary_node",
                                                "NO": END
                                            })
        graph_builder.add_edge("generate_facetwise_marking_and_summary_node", END)

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
        }
        result = self.graph.invoke(state, config=config)
        print(f"graph initial invocation result: {result}")
        return result["messages"][-1].content
    
    def run_graph(self, message):
        config = {"configurable": {"thread_id": self.conversation_id}}
        state = {
            "messages": {"role": "user", "content": message},
            "current_user_message": message,
            "is_success_met": False,
        }
        result = self.graph.invoke(state, config=config)
        print(f"graph invocation result: {result}")
        return result["messages"][-1].content