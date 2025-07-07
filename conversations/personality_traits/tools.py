from langchain_community.agent_toolkits import FileManagementToolkit

def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="output",)
    return toolkit.get_tools()

