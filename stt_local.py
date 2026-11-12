import asyncio
import tempfile
from livekit.agents import stt, APIConnectOptions
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.audio import combine_frames, AudioBuffer
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"


class LocalWhisperSTT(stt.STT):
    def __init__(self, language: str = "fr"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._language = language

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = combine_frames(buffer)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(frame.to_wav_bytes())
            tmp.flush()

            result = await asyncio.to_thread(
                mlx_whisper.transcribe,
                tmp.name,
                path_or_hf_repo=MODEL,
                language=self._language,
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=self._language, text=result["text"].strip())],
        )
