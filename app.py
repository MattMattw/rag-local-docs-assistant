import os
import fitz  # PyMuPDF

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document


# =========================
# 1. CONFIGURAZIONE
# =========================

DATA_FOLDER = "data"          # cartella dove metti i PDF
MODEL_NAME = "llama3.1"       # modello Ollama
CHUNK_SIZE = 1200             # Chunk più grandi per tenere Q&A insieme
CHUNK_OVERLAP = 300
TOP_K = 5                     # quanti chunk recuperare


# =========================
# 2. LLM LOCALE (GENERAZIONE)
# =========================

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0,
    num_predict=256,  # limite risposte
)


# =========================
# 3. EMBEDDINGS (INDICIZZAZIONE)
# =========================

embeddings = OllamaEmbeddings(
    model=MODEL_NAME
)


# =========================
# 4. CARICAMENTO PDF CON PYMUPDF
# =========================

documents = []

for filename in os.listdir(DATA_FOLDER):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(DATA_FOLDER, filename)
        print(f"Loading PDF: {file_path}")
        
        # Use PyMuPDF instead of PyPDFLoader for better text extraction
        pdf_document = fitz.open(file_path)
        
        for page_num, page in enumerate(pdf_document, start=1):
            text = page.get_text()
            
            # Clean up the text - remove common TOC patterns and junk
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                stripped = line.strip()
                # Skip empty lines, page numbers, dots (TOC), and lines that are too short
                if not stripped or stripped.isdigit() or all(c == '.' or c.isspace() for c in line):
                    continue
                # Skip lines that look like TOC entries (mostly dots)
                if len(stripped) < 2 or stripped.count('.') > len(stripped) // 2:
                    continue
                cleaned_lines.append(stripped)
            
            text = '\n'.join(cleaned_lines).strip()
            
            # Only add pages with substantial content
            if len(text) > 100:  # More than 100 chars
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": filename}
                ))
        
        pdf_document.close()

if not documents:
    raise ValueError("Nessun PDF trovato nella cartella 'data/'")

print(f"Loaded {len(documents)} pages")


# =========================
# 5. SPLIT DEL TESTO
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", " ", ""]  # Better splitting strategy
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")


# =========================
# 6. VECTOR STORE (FAISS)
# =========================

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K},
    search_type="similarity"  # Use similarity search
)


# =========================
# 7. HELPER FUNCTION
# =========================

def answer_question(question: str) -> str:
    """Answer a question using RAG"""
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"[DEBUG] Retrieved {len(retrieved_docs)} chunks")
    
    # Format context
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        page = doc.metadata.get('page', '?')
        print(f"[DEBUG] Chunk {i} (Page {page}): {doc.page_content[:100]}...")
        context_parts.append(f"[Pagina {page}]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create messages
    system_prompt = """Sei un assistente che risponde domande basandoti ESCLUSIVAMENTE sul contesto fornito.

REGOLE IMPORTANTI:
1. Rispondi SOLO se la risposta è chiaramente nel contesto
2. Se la risposta non è nel contesto, rispondi ESATTAMENTE: "Informazione non trovata nei documenti"
3. Non aggiungere informazioni da altre fonti
4. Sii breve e diretto
5. Se il contesto è confuso, rispondi comunque che l'informazione non è trovata"""

    user_message = f"""CONTESTO:
{context}

DOMANDA: {question}

Ricorda: rispondi SOLO con quello che è nel contesto."""

    # Call LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    return response


# =========================
# 8. LOOP INTERATTIVO
# =========================

print("\nRAG pronto. Scrivi una domanda (digita 'exit' per uscire).\n")

while True:
    question = input("> ").strip()

    if question.lower() == "exit":
        break
    
    if not question:
        continue

    try:
        response = answer_question(question)
        print(f"\nRisposta:\n{response}\n")
    except Exception as e:
        print(f"Errore: {e}\n")
