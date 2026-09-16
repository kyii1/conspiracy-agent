import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

st.title("👽 Conspiracy RAG Agent")
st.write("The truth is inside the local folder.")

# @st.cache_resource prevents the AI from having to re-read the 
# text file from scratch every time you type a single letter.
@st.cache_resource
def load_agent():
    # 1. Connect to your local Ollama models
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # 2. Read all text files in the 'data' folder
    documents = SimpleDirectoryReader("data").load_data()
    
    # 3. Chop the text into vectors and create the search engine
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

query_engine = load_agent()

# 4. The Web UI Chat Box
user_question = st.text_input("Ask a question about the Roswell document:")

if user_question:
    with st.spinner("Agent is searching the classified files..."):
        response = query_engine.query(user_question)
        st.write("🕵️ **Agent:**", response.response)

#Dummy text       