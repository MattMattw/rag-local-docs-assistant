from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# 1. Carica LLM locale
llm = OllamaLLM(
    model="llama3.1",
    temperature=0
)

# 2. Carica documento
loader = PyPDFLoader("data/example.pdf")
documents = loader.load()

# 3. Split del testo
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# 4. Vector store
vectorstore = FAISS.from_documents(
    chunks,
    embedding=llm.embed
)

retriever = vectorstore.as_retriever()

# 5. Prompt
prompt = ChatPromptTemplate.from_template("""
Rispondi alla domanda usando SOLO il contesto seguente.

Contesto:
{context}

Domanda:
{question}
""")

# 6. Loop interattivo
while True:
    question = input("\nDomanda (exit per uscire): ")
    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)

    response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )

    print("\nRisposta:\n", response)
