from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY

def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
