import asyncio
import logging
import os

import psutil
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import elevenlabs, openai, silero

from stt_local import LocalWhisperSTT
from tools import search_knowledge_base

load_dotenv()
logger = logging.getLogger("assurantia-agent")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def get_llm():
    logger.info(f"LLM : Ollama local ({OLLAMA_MODEL} @ {OLLAMA_BASE_URL})")
    return openai.LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, api_key="ollama")


async def log_ram_periodically(interval: int = 30) -> None:
    process = psutil.Process(os.getpid())
    while True:
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"RAM du process agent : {mem_mb:.1f} Mo")
        await asyncio.sleep(interval)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Tu es l'assistant vocal du service client d'Assurantia, une compagnie "
                "d'assurance auto et habitation. Tu réponds en français, de façon claire "
                "et concise, adaptée à une conversation orale (phrases courtes, pas de listes "
                "à puces à l'oral).\n\n"
                "Pour toute question sur les contrats, sinistres ou résiliations, utilise "
                "l'outil search_knowledge_base pour vérifier l'information avant de répondre. "
                "Ne réponds JAMAIS sur ces sujets sans avoir consulté la base de connaissances : "
                "n'invente jamais un délai, un montant ou une condition.\n\n"
                "Si la base de connaissances ne contient pas la réponse, ou si le client est "
                "insatisfait après deux tentatives, ou s'il s'agit d'une demande de "
                "remboursement contestée, propose de transférer la conversation à un conseiller humain."
            ),
            tools=[search_knowledge_base],
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=LocalWhisperSTT(language="fr"),
        llm=get_llm(),
        tts=elevenlabs.TTS(language="fr"),
        vad=silero.VAD.load(),
        turn_detection="vad",
    )

    asyncio.create_task(log_ram_periodically())

    await session.start(agent=SupportAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
