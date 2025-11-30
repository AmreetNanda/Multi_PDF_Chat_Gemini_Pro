# app.py
import streamlit as st
from dotenv import load_dotenv
import google.genai as genai
import os

from pdf_utils import extract_text_from_pdfs, split_text_into_chunks
from vectorstores_utils import save_vector_store, load_vector_store
from rag_chain import get_rag_chain, answer_question

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
st.set_page_config("Chat with PDFs")
st.header("📚 Chat with Multiple PDFs using Gemini")

# Question area
user_question = st.text_input("Ask a question from the uploaded PDF files:")

# Answer question
if user_question:
    vector_store = load_vector_store()
    chain = get_rag_chain()
    response = answer_question(chain, vector_store, user_question)
    
    st.subheader("📌 Answer:")
    st.write(response)

# Sidebar PDF upload
with st.sidebar:
    st.title("📁 Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.button("Submit & Process"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                text = extract_text_from_pdfs(pdf_docs)
                chunks = split_text_into_chunks(text)
                save_vector_store(chunks)
                st.success("PDFs processed and embeddings stored!")
