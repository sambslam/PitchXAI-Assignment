import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams


# System prompt: defines how the calling agent behaves.
SYSTEM_PROMPT = (
    "You are a friendly phone agent for PitchX, a company that builds AI calling "
    "agents and automations for sales teams. You are speaking to someone who called "
    "in to learn about the product. Keep replies short and conversational, one or two "
    "sentences, since this is a voice call. Be warm, clear, and helpful. If you do not "
    "know a specific detail, say so simply rather than making it up."
)


async def run_bot(transport):
    """Build and run the voice pipeline on a given transport."""

    # Speech to text: faster-whisper on the GPU.
    stt = WhisperSTTService(
        model="base",
        device="cuda",
        compute_type="float16",
    )

    # The brain: local Ollama model. base_url defaults to the local Ollama server.
    llm = OLLamaLLMService(
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
    )

    # Text to speech: Kokoro, using the same voice tested earlier.
    tts = KokoroTTSService(
        voice_id="af_heart",
    )

    # Conversation context, seeded with the system prompt.
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # The pipeline: audio in -> transcribe -> model -> speak -> audio out.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, greeting them.")
        # Kick off the conversation with a greeting.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected.")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point discovered by the Pipecat development runner."""
    transport = None

    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
            webrtc_connection=runner_args.webrtc_connection,
        )
    
    else:
        logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
        return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
