import os

import faiss
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# ============================================================
# Set your Gemini API Key
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Enter your GOOGLE_API_KEY: ")

# ============================================================
# Initialize LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    google_api_key=GOOGLE_API_KEY,
)

# ============================================================
# Sample Knowledge Base
# ============================================================

big_paragraph = """
The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies.

The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s to enable time-sharing of computers. The primary precursor network, the ARPANET, initially served as a backbone for interconnection of academic and research networks. The funding of the National Science Foundation Network (NSFNET) in the 1980s, as well as private commercial Internet service providers, led to the worldwide participation in the development of new networking technologies.

Today, the Internet supports cloud computing, online gaming, social media, education, healthcare, and communication. While it provides unprecedented access to information, it also presents challenges related to privacy, security, and misinformation.
"""

documents = [Document(page_content=big_paragraph)]

# ============================================================
# Split Documents
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

# ============================================================
# Embeddings
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

embedding_dim = len(embeddings.embed_query("hello"))

index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY from the retrieved context.

If the answer is not present, say:

"I don't know based on the provided context."

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

# ============================================================
# Chat Loop
# ============================================================

print("=" * 60)
print("Simple Gemini RAG Chatbot")
print("Type 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\nBot:", answer)
