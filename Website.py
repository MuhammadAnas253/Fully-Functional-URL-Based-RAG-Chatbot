from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load documents from the web
# Replace the URL with the web page you want to load
url = "https://en.wikipedia.org/wiki/Lahore"  # Change this to your target URL
loader = WebBaseLoader(url)
documents = loader.load()

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
# Using HuggingFace embeddings to avoid OpenAI; requires sentence-transformers library
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Set up the retriever
retriever = vectorstore.as_retriever()

# Step 5: Define the prompt template
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Question: {question}

Context: {context}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Step 6: Set up the LLM using Groq API (requires Groq API key)
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0.2,
    api_key="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" 
)
# Step 7: Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
query = "when does the lahore served as a capital of an Empire and which empire it was?"
response = rag_chain.invoke(query)
print(response)