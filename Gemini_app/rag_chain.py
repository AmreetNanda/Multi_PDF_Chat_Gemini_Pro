from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# from langchain.chains import create_stuff_documents_chain
# from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def get_rag_chain():
    template = """
Answer the question ONLY using the provided context.
Be detailed and accurate.
If the answer is not in the context, say:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3
    )

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    
    return chain

def answer_question(chain, vector_store, question):
    docs = vector_store.similarity_search(question, k=4)
    response = chain.invoke({"context": docs, "question": question})
    return response
