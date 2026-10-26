from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# --- Constantes de config ---
KNOWLEDGE_DIR = Path("knowledge_base")
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "assurantia"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Modèle multilingue, bien meilleur que le défaut pour le français
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)


OVERLAP_PARAGRAPHS = 1   # nombre de paragraphes complets à répéter au chunk suivant


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap_paragraphs: int = OVERLAP_PARAGRAPHS) -> list[str]:
    """Découpe un texte en chunks, avec chevauchement par paragraphes entiers (jamais de coupure en plein mot)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []       # liste de paragraphes accumulés dans le chunk en cours
    current_len = 0

    for para in paragraphs:
        # Si ajouter ce paragraphe dépasse la taille cible, on clôture le chunk actuel
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Chevauchement : on repart avec les derniers paragraphes du chunk précédent,
            # entiers, jamais tronqués au caractère
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) for p in current)

        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # On repart de zéro : les anciens vecteurs (mauvais modèle) doivent disparaître
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # la collection n'existait pas encore, pas grave

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,   # <-- le changement clé
    )

    ids, documents, metadatas = [], [], []
    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{file_path.stem}_{i}")
            documents.append(chunk)
            metadatas.append({"source": file_path.name})

    if not documents:
        print("Aucun document trouvé dans knowledge_base/")
        return

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"{len(documents)} chunks indexés depuis {len(list(KNOWLEDGE_DIR.glob('*.md')))} fichiers.")


def search(question: str, n_results: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,   # <-- même fonction qu'à l'indexation, obligatoire
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