import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import os.path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for metadata
if "current_metadata" not in st.session_state:
    st.session_state.current_metadata = None

# Constants
PERSIST_DIR = "./storage"

@st.cache_resource
def initialize_index():
    """Initialize or load the LlamaIndex index"""
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("/doc").load_data()
        node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        

        
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    return index

# Initialize the index and query engine
index = initialize_index()
query_engine = index.as_query_engine(similarity_top_k=5, include_metadata=True)

# Custom CSS for the chat interface
st.markdown("""
<style>
.chat-row {
    display: flex;
    margin-bottom: 1rem;
}
.metadata-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin-top: 10px;
}
.source-files {
    margin-top: 5px;
    padding: 5px 10px;
    background-color: #e6e9ef;
    border-radius: 5px;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("💬 Document Chat Assistant")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source_files"):
            with st.expander("Source Files"):
                for file in message["source_files"]:
                    st.markdown(f"📄 {file}")

# Chat input
if prompt := st.chat_input("Ask me about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Get response from query engine
            response = query_engine.query(prompt)
            
            # Extract unique filenames from source nodes
            unique_files = set()
            for node in response.source_nodes:
                filename = node.node.metadata.get('file_name')
                if filename:
                    unique_files.add(filename)
            
            # Display the response
            st.markdown(response.response)
            
            # Display unique source files in an expander
            with st.expander("Source Files"):
                for file in unique_files:
                    st.markdown(f"📄 {file}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.response,
                "source_files": list(unique_files)
            })
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Sidebar with additional options
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This chat interface allows you to interact with your documents using natural language queries.
    The assistant will provide answers based on the content of your documents and show you the
    source files used for each response.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()