import gradio as gr
from graph import InitialConversationGraph

graph = InitialConversationGraph()

graph.setup()

initial_message = graph.run_graph_initially()

def chat(message, history):
    if not history:  # If this is the first user message, add the initial assistant message
        history = [("assistant", initial_message)]
    history.append(("user", message))
    response = graph.run_graph(message)
    history.append(("assistant", response))
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[("assistant", initial_message)], label="Soul Match")
    msg = gr.Textbox()

    def user_chat(user_message, chat_history):
        updated_history = chat(user_message, chat_history)
        return "", updated_history  # Clear textbox, update chat

    msg.submit(user_chat, [msg, chatbot], [msg, chatbot])

demo.launch(inbrowser=True)