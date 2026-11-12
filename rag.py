from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_DIR = Path("knowledge_base")
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "assurantia"
CHUNK_SIZE = 500
OVERLAP_PARAGRAPHS = 1

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap_paragraphs: int = OVERLAP_PARAGRAPHS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) for p in current)

        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    files = list(KNOWLEDGE_DIR.glob("*.md"))
    ids, documents, metadatas = [], [], []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{file_path.stem}_{i}")
            documents.append(chunk)
            metadatas.append({"source": file_path.name})

    if not documents:
        print("Aucun document trouvé dans knowledge_base/")
        return

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"{len(documents)} chunks indexés depuis {len(files)} fichiers.")


def search(question: str, n_results: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    results = collection.query(query_texts=[question], n_results=n_results)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits


if __name__ == "__main__":
    build_index()
    print("\n--- Test de recherche ---")
    resultats = search("comment résilier mon assurance auto ?", n_results=5)
    for r in resultats:
        print(f"\n[{r['source']}] (distance: {r['distance']:.3f})\n{r['text'][:200]}...")
