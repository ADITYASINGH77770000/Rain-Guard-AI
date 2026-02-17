from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "ndma_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

query = "What actions should authorities take during high flood risk?"

results = db.similarity_search(query, k=3)

print("\nNDMA-based recommendations:\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content[:400])
    print()
