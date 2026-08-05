import os
import faiss

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# ----------------------------------------------------
# Gemini API Key
# ----------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    google_api_key=GOOGLE_API_KEY,
)

# ----------------------------------------------------
# Knowledge Base
# ----------------------------------------------------
big_paragraph = """
The Internet is a global system of interconnected computer networks that uses
the Internet protocol suite (TCP/IP).

The origins of the Internet date back to the development of packet switching
during the 1960s. The primary precursor network was the ARPANET.

The commercialization of the Internet in the mid-1990s led to worldwide adoption.
"""

documents = [Document(page_content=big_paragraph)]

# ----------------------------------------------------
# Split Documents
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

# ----------------------------------------------------
# Embeddings
# ----------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

dimension = len(embeddings.embed_query("hello"))

index = faiss.IndexFlatL2(dimension)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ----------------------------------------------------
# Prompt
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer ONLY from the retrieved context.

Context:
{context}

Question:
{question}

Answer:
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------
app = FastAPI(
    title="Gemini RAG API",
    version="1.0",
    description="LangServe + Gemini + FAISS",
)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
