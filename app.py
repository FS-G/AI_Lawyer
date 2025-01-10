# Import dependencies
import streamlit as st
import os
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  
import pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone


# for the  code  file 
openai_api_key = os.getenv("OPENAI_API_KEY")
pine_api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = pine_api_key

# Initialize Pinecone 
pc = pinecone.Pinecone(api_key=pine_api_key, environment="us-east1-gcp")

# Defining Constants
namespace = "wondervector5000"
index_name = "ai-rag"

# Loading embedding model storing cache
@st.cache_resource
def load_embeddings():
    embedding_model = "BAAI/bge-base-en-v1.5"
    return HuggingFaceEmbeddings(model_name=embedding_model)

# Instantiate language model 
llm_gpt = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# Initialize Pinecone Vector Store 
@st.cache_resource
def initialize_docsearch(_embeddings):
    return Pinecone.from_documents(
        documents=[],  # Empty initially, 
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

# Initialize QA 
@st.cache_resource
def initialize_qa_system(_docsearch):
    return RetrievalQA.from_chain_type(
        llm=llm_gpt,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 10})
    )

# Defining function to add background color
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #a0bfb9;
            color: #895051;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# Streamlit app settings
st.set_page_config(page_title="AI Lawyer by Shreyas", page_icon=":bar_chart:")
add_bg_from_url()

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login logic
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "shreyas" and password == "admin":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()  # Refresh after login
        else:
            st.error("Invalid username or password")

# Main app logic 
else:
    st.title("AI Lawyer by Shreyas")
    
    # Question input and response
    question = st.text_input("Ask legal queries:")
    if st.button("Submit query"):
        if question:
            with st.spinner("Getting your answer from the AI Lawyer..."):
                embeddings = load_embeddings()  # Load embeddings 
                docsearch = initialize_docsearch(embeddings)  # Initialize the vector store
                qa = initialize_qa_system(docsearch)  # Initialize the QA system
                
                # Retrieving relevant documents from Pinecone
                retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            
                # Defining Customize the prompt with retrieved context and user question
                custom_prompt = f"""
                You are an AI lawyer specialized in legal documents. Your task is to provide a concise, well-reasoned answer to the legal question based on the provided context. Please do not rely on any information outside the context.

                Context:
                {context}

                Question:
                {question}

                Answer:
                Your answer should be based solely on the context provided above. If the answer is not explicitly covered in the context, clearly state that the information is insufficient. Avoid including any personal opinions or knowledge outside of the retrieved content.
                """

                
                # Getting the answer from the QA system
                answer = qa.run(custom_prompt)
                
                st.subheader("Answer")
                st.write(answer)
                # st.subheader("Context")
                # st.write(context)
        else:
            st.error("Please enter a question.")



    
