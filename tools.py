from livekit.agents import function_tool, RunContext
from rag import search as rag_search


@function_tool
async def search_knowledge_base(context: RunContext, question: str) -> str:
    """Cherche des informations dans la base de connaissances d'Assurantia
    (contrats, sinistres, résiliation) pour répondre précisément à une question client."""
    results = rag_search(question, n_results=4)
    if not results:
        return "Aucune information trouvée dans la base de connaissances."
    return "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)