import streamlit as st
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.genai as genai

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
# from langchain.chains import create_stuff_documents_chain
from langchain_community.chains import create_stuff_documents_chain

from dotenv import load_dotenv
import os

# -----------------------------------------
# Load API Keys
# -----------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------------------
# Read PDF File
# -----------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------------------
# Split into Text Chunks
# -----------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    return splitter.split_text(text)

# -----------------------------------------
# Create FAISS Vector Store
# -----------------------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -----------------------------------------
# Build Conversational Retrieval Chain (Updated)
# -----------------------------------------
def get_conversational_chain():
    prompt_template = """
Answer the question ONLY using the provided context.
Be very detailed and accurate.
If the answer is not in the context, respond with:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    # NEW Replacement for load_qa_chain
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    return chain

# -----------------------------------------
# Answer User Query
# -----------------------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # FIX: embeddings argument spelled correctly
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(user_question, k=4)

    chain = get_conversational_chain()

    # NEW LCEL style
    response = chain.invoke({
        "context": docs,
        "question": user_question
    })

    st.write("### Reply:")
    st.write(response)

# -----------------------------------------
# Streamlit App
# -----------------------------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("📚 Chat with Multiple PDFs using Google Gemini")

    user_question = st.text_input("Ask a question from your uploaded PDF files:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDFs processed and indexed successfully!")

if __name__ == "__main__":
    main()
