import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Configuration
DATA_FOLDER = "data"
MODEL_NAME = "llama3.1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Load documents
documents = []
for filename in os.listdir(DATA_FOLDER):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

if not documents:
    raise ValueError("Nessun PDF trovato nella cartella 'data/'")

print(f"Loaded {len(documents)} pages")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test retrieval
test_questions = [
    "Che cos'è bitcoin?",
    "Cos'è il mining di Bitcoin?",
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} chunks\n")
    
    for i, doc in enumerate(retrieved_docs):
        print(f"[Chunk {i+1}]")
        print(doc.page_content)
        print(f"(Page {doc.metadata.get('page', 'unknown')})\n")
