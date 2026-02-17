from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "knowledge_base" / "ndma guidelines.pdf"
INDEX_PATH = BASE_DIR / "ndma_index"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"NDMA PDF not found at: {PDF_PATH}")

print("Loading NDMA guidelines...")
loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()

print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector index...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local(str(INDEX_PATH))

print("NDMA RAG index created successfully")
