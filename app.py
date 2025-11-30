import streamlit as st
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.question_answering import load_qa_chain

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import os 
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Read the PDF file
def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:                # Read all the pdfs from the pdf_docs
        pdf_reader = PdfReader(pdf)     # Read the pdf with pdf reader
        for page in pdf_reader.pages:   # multiple pages read
            text+=page.extract_text()   # append it in the text variable 
    return text

# Divide the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap= 1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Store these chunks into the vector stores for embedding 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create the conversational chain
def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided content, make sure to provide all the details, if the answer is not in the provided context just say "answer is not available in the context", don't provide wrong answer \n\n
Context:\n{context}?\n
Question:\n{question}\n
Answer:
"""

    model = ChatGoogleGenerativeAI(model = "Gemini 2.5 Pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents":docs, "question":user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with multiple pdfs using Google Gemini")

    user_question = st.text_input("Ask a question from the pdf files")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your pdf files and click on the submit button", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing ... "):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__=="__main__":
    main()

# ---------------------------------------------------------------------------------------------------------
# import streamlit as st
# from PyPDF2 import PdfReader

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.genai as genai

# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_core.prompts import PromptTemplate
# # from langchain.chains import create_stuff_documents_chain
# from langchain_community.chains import create_stuff_documents_chain

# from dotenv import load_dotenv
# import os

# # -----------------------------------------
# # Load API Keys
# # -----------------------------------------
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # -----------------------------------------
# # Read PDF File
# # -----------------------------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # -----------------------------------------
# # Split into Text Chunks
# # -----------------------------------------
# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=5000,
#         chunk_overlap=500
#     )
#     return splitter.split_text(text)

# # -----------------------------------------
# # Create FAISS Vector Store
# # -----------------------------------------
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # -----------------------------------------
# # Build Conversational Retrieval Chain (Updated)
# # -----------------------------------------
# def get_conversational_chain():
#     prompt_template = """
# Answer the question ONLY using the provided context.
# Be very detailed and accurate.
# If the answer is not in the context, respond with:
# "Answer is not available in the context."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=prompt_template
#     )

#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

#     # NEW Replacement for load_qa_chain
#     chain = create_stuff_documents_chain(
#         llm=model,
#         prompt=prompt
#     )

#     return chain

# # -----------------------------------------
# # Answer User Query
# # -----------------------------------------
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # FIX: embeddings argument spelled correctly
#     db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#     docs = db.similarity_search(user_question, k=4)

#     chain = get_conversational_chain()

#     # NEW LCEL style
#     response = chain.invoke({
#         "context": docs,
#         "question": user_question
#     })

#     st.write("### Reply:")
#     st.write(response)

# # -----------------------------------------
# # Streamlit App
# # -----------------------------------------
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("📚 Chat with Multiple PDFs using Google Gemini")

#     user_question = st.text_input("Ask a question from your uploaded PDF files:")
#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("📁 Upload PDFs")
#         pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

#         if st.button("Submit & Process"):
#             with st.spinner("Processing PDFs..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 chunks = get_text_chunks(raw_text)
#                 get_vector_store(chunks)
#                 st.success("PDFs processed and indexed successfully!")

# if __name__ == "__main__":
#     main()
