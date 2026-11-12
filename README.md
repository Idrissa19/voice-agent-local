# Assurantia — Agent vocal de service client, version locale

Variante 100 % locale de [voice-agent](https://github.com/Idrissa19/voice-agent) : la transcription comme le raisonnement tournent sur la machine, sans API de LLM externe. L'agent répond à l'oral, en français, aux questions sur les contrats, les sinistres et les résiliations d'une compagnie d'assurance fictive (Assurantia). Il s'appuie sur une base de connaissances interrogée par recherche vectorielle (RAG).

## Fonctionnement

```
voix du client ─▶ STT (Whisper, en local) ─▶ LLM (Ollama, en local) ─▶ TTS (ElevenLabs) ─▶ voix de l'agent
                                               │
                                               └─▶ outil search_knowledge_base ─▶ ChromaDB (RAG)
```

- **Temps réel** : [LiveKit Agents](https://docs.livekit.io/agents/) gère la session audio (WebRTC) et la détection de prise de parole (VAD Silero).
- **Transcription** : Whisper large-v3-turbo via `mlx-whisper`, exécuté dans un thread séparé pour ne pas bloquer la boucle audio.
- **Raisonnement** : un modèle servi par Ollama (`llama3.1` par défaut) via son API compatible OpenAI. Aucune requête ne part vers un fournisseur de LLM.
- **Synthèse vocale** : ElevenLabs, en français. C'est le seul service externe.
- **RAG** : les documents de `knowledge_base/` sont découpés par paragraphes entiers (avec chevauchement), vectorisés avec un modèle d'embedding multilingue (`paraphrase-multilingual-MiniLM-L12-v2`) et stockés dans ChromaDB.
- **Suivi des ressources** : la RAM utilisée par l'agent est affichée toutes les 30 secondes. C'est utile pour dimensionner la machine quand le LLM et Whisper tournent en local.
- **Garde-fous** : le prompt système interdit d'inventer un délai, un montant ou une condition. Il demande aussi de proposer un conseiller humain quand la base ne contient pas la réponse.

## Structure

| Fichier | Rôle |
|---|---|
| `agent.py` | Point d'entrée : session vocale, prompt système, LLM Ollama, suivi de la RAM |
| `stt_local.py` | Adaptateur LiveKit pour Whisper en local |
| `rag.py` | Découpage, indexation et recherche dans ChromaDB |
| `tools.py` | Outil `search_knowledge_base` exposé au LLM |
| `ingest.py` | Construit l'index vectoriel à partir de `knowledge_base/` |
| `download_model.py` | Télécharge le modèle Whisper à l'avance |
| `knowledge_base/` | Documents de référence (contrats, sinistres, résiliation) |

## Installation

Prérequis : Python 3.10+, un Mac Apple Silicon (pour `mlx-whisper`), [Ollama](https://ollama.com) et un compte [LiveKit Cloud](https://cloud.livekit.io).

```bash
ollama pull llama3.1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Remplis ensuite `.env` avec tes clés LiveKit et ElevenLabs.

## Lancement

```bash
python download_model.py
python ingest.py
python agent.py dev
```

Ouvre ensuite le [playground LiveKit](https://agents-playground.livekit.io) pour parler à l'agent.
