import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

st.title("👽 Conspiracy RAG Agent")
st.write("The truth is inside the local folder.")

@st.cache_resource
def load_agent():
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # 2. Read the raw PDFs
    raw_documents = SimpleDirectoryReader("data").load_data()
    
    # 2.5 DATA SCRUBBING: Extract, clean, and create fresh Documents
    documents = []
    for doc in raw_documents:
        clean_text = doc.text.replace("\x00", "")
        
        # Create a new Document object with the clean text and old metadata
        clean_doc = Document(
            text=clean_text,
            metadata=doc.metadata
        )
        documents.append(clean_doc)
    
    # 3. Setup PostgreSQL Vector Database Connection
    vector_store = PGVectorStore.from_params(
        database="postgres",
        host="localhost",
        password="password",
        port=5432,
        user="postgres",
        table_name="fbi_vault",
        embed_dim=768 
    )
    
    # 4. Tell LlamaIndex to use Postgres
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5. Create the index using our NEW clean documents
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    
    return index.as_query_engine()

# Load the agent (Streamlit caches this so it doesn't reload constantly)
query_engine = load_agent()

# --- THE CHAT INTERFACE ---

# 1. Initialize the chat memory if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Draw all previous messages to the screen every time the app updates
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. The new Chat Input box at the bottom of the screen
if prompt := st.chat_input("Ask about the classified files..."):
    
    # Immediately draw the user's question to the screen and save it to memory
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Trigger the AI Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is searching the Vault..."):
            # Query the database
            response = query_engine.query(prompt)
            # Print the response to the screen
            st.markdown(response.response)
            
    # Save the AI's response to memory
    st.session_state.messages.append({"role": "assistant", "content": response.response})