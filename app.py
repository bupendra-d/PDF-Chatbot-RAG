
import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setup
load_dotenv()
st.set_page_config(page_title="PDF Chatbot")
st.title("PDF Chatbot")

# API Key
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None

GROQ_API_KEY = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please provide your GROQ API key.")
    st.stop()

# Models
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the following context.
If the answer is unavailable, say:
"I couldn't find this information in the uploaded PDF."

Context:
{context}

Question: {input}
""")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_hash = None

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = os.path.join(temp_dir, "document.pdf")

                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)

                    docs = PyPDFLoader(pdf_path).load()

                docs = [doc for doc in docs if doc.page_content.strip()]

                if not docs:
                    st.error("No readable text found in PDF.")
                    st.stop()

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # Create vector database
                st.session_state.vectorstore = FAISS.from_documents(
                    chunks, embeddings
                )
                st.session_state.file_hash = file_hash

                st.success(f"PDF processed! {len(chunks)} chunks created.")

            except Exception as e:
                st.error(f"Processing error: {e}")

    # Question answering
    if (
        st.session_state.vectorstore is not None
        and st.session_state.file_hash == file_hash
    ):
        question = st.text_input("Ask a question about your PDF")

        if st.button("Get Answer") and question.strip():
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                document_chain = create_stuff_documents_chain(llm, prompt)

                rag_chain = create_retrieval_chain(
                    retriever, document_chain
                )

                with st.spinner("Generating answer..."):
                    response = rag_chain.invoke({"input": question})

                st.subheader("Answer")
                st.write(response["answer"])

                # Display sources
                with st.expander("Source Passages"):
                    for i, doc in enumerate(response["context"], 1):
                        page = doc.metadata.get("page")
                        st.markdown(f"**Source {i} | Page {page + 1 if isinstance(page, int) else 'N/A'}**")
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error generating answer: {e}")
