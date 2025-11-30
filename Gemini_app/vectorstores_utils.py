from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def save_vector_store(chunks, save_path="faiss_index"):
    embeddings = create_embeddings_model()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(save_path)

def load_vector_store(save_path="faiss_index"):
    embeddings = create_embeddings_model()
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
