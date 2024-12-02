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


# for the  code  file 
gemni_api_key = os.getenv("GEMNI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = api_key

# Define constants
namespace = "wondervector5000"
index_name = "thesis"
chunk_size = 1000

USERNAME = "shreyas"
PASSWORD = "admin"

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = PineconeVectorStore.from_documents(
    documents="", 
    index_name=index_name, 
    embedding=embeddings, 
    namespace=namespace
)
time.sleep(1)

# Set up LLM and QA chain
llm =  ChatGoogleGenerativeAI(
    google_api_key = gemni_api_key,
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={"k": 10})
)

# Function to add background color
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
st.set_page_config(page_title="AI Lawyer by shreyas", page_icon=":bar_chart:")
add_bg_from_url()

# Session state for login, feedback, question, and visibility
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login logic
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app logic (after login)
else:
    st.title("AI Lawyer by shreyas")

    # Check if Pinecone database is empty

    # Show file upload section only if the database is empty
    st.write("Upload documents")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file:
        if st.button("Submit the file"):
            with st.spinner("Uploading and processing document..."):
                with open("uploaded_pdf.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("uploaded_pdf.pdf")
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
                documents = text_splitter.split_documents(pages)
                docsearch = PineconeVectorStore.from_documents(
                    documents=documents,
                    index_name=index_name,
                    embedding=embeddings,
                    namespace=namespace,
                )
            st.success("Document uploaded and processed. You can now ask questions about its content.")


    # Question input and response
    question = st.text_input("Ask legal queries:")
    if st.button("Submit query"):
        with st.spinner("Getting your answer from the AI Lawyer..."):
            retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            answer = qa.invoke(question)
            st.subheader("Answer")
            st.write(answer["result"])
            st.subheader("Context")
            st.write(context)


        # Clear database button
    if st.button("Clear the database"):
        with st.spinner("Clearing the database..."):
            try:
                pc = Pinecone(api_key=api_key)
                index = pc.Index(index_name)
                index.delete(delete_all=True, namespace=namespace)
                st.success("Database cleared!")
            except:
                st.error("The database is already empty.")



    