import os
import streamlit as st
from groq import Groq
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()
# Load environment variables from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Page Title
st.set_page_config(page_title="Ask your DigiDoctor👨‍⚕️", layout="wide")
st.title("Ask your DigiDoctor👨‍⚕️")

with st.sidebar:
    st.selectbox("Select a model", options=["Llama 3.3 70B Versatile","gemma2-9b-it"], key="model_select")
    st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature_slider")
    st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1, key="top_p_slider")
    st.slider("Top K", min_value=0, max_value=100, value=50, step=1, key="top_k_slider")
    st.slider("Max Tokens", min_value=100, max_value=2000, value=1000, step=100, key="max_tokens_slider")


# Cache and load documents, chunks, vectorstore
@st.cache_resource
def load_vector_store():
    loader = DirectoryLoader( path="data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings for the text chunks using HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}, )

    # create a vector store
    vector_store = FAISS.from_documents(text_chunks,embeddings)
    return vector_store

vector_store = load_vector_store()

# Initialize Grog LLM(Langchain) 
llama_model = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile"
)

gemma_model = ChatGroq(
    api_key=groq_api_key,
    model="gemma2-9b-it"
)

# Customize the name of the model
prompt_template = ChatPromptTemplate.from_template(

    """
    You are DigiDoctor, a friendly medical AI assistant. 
    Always introduce yourself as DigiDoctor. 
    Answer the user's question using the following context.
    {context}
    Question: {question}

    """
)

# Select model based on user input
if st.session_state.get("model_select") == "Llama 3.3 70B Versatile":
    llm = llama_model
    st.session_state.temperature = st.session_state.get("temperature_slider")
    st.session_state.top_p = st.session_state.get("top_p_slider")
    st.session_state.top_k = st.session_state.get("top_k_slider")
    st.session_state.max_tokens = st.session_state.get("max_tokens_slider")
else:
    llm = gemma_model
    st.session_state.temperature = st.session_state.get("temperature_slider")
    st.session_state.top_p = st.session_state.get("top_p_slider")
    st.session_state.top_k = st.session_state.get("top_k_slider")
    st.session_state.max_tokens = st.session_state.get("max_tokens_slider")

# Set up memory and retriever chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm = llm, 
                                              chain_type = "stuff",
                                              retriever = vector_store.as_retriever(search_kwargs = {"k":2}),
                                              memory = memory,
                                              combine_docs_chain_kwargs = {"prompt": prompt_template}
                                              )



# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Take Input Query from User
query_input = st.chat_input("Ask your questions about stroke: ", key="input")

# Handle User Query
if query_input:
    with st.spinner("Consulting your DigiDoctor..."):
        response = chain.run(query_input)
        st.session_state.messages.append({"role":"user", "content":query_input})
        st.session_state.messages.append({"role":"assistant", "content":response})

# Display chat messages
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    message(msg["content"], is_user = is_user, key=str(i))

