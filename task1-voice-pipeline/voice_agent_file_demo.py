"""
File-based voice agent demo.
 
Runs the full voice loop on a recorded audio file, on the pod, with no
network/WebRTC involved:
 
    audio file -> faster-whisper (STT) -> Ollama 8B (LLM) -> Kokoro (TTS) -> audio file
 
This proves the agent takes speech in and gives a spoken response back. It also
times each stage so you can see where the latency goes.
"""
 
import time
import requests
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro import KPipeline
 
INPUT_AUDIO = "/workspace/input.wav"
OUTPUT_AUDIO = "/workspace/agent_reply.wav"
 
SYSTEM_PROMPT = (
    "You are a friendly phone agent for PitchX, a company that builds AI calling "
    "agents and automations for sales teams. Keep replies short and conversational, "
    "one or two sentences, since this is a voice call. Be warm and clear."
)
 
 
def transcribe(path):
    """Speech to text with faster-whisper."""
    model = WhisperModel("base", device="cuda", compute_type="float16")
    segments, info = model.transcribe(path)
    text = " ".join(seg.text for seg in segments).strip()
    return text
 
 
def get_reply(user_text):
    """Send the transcript to the local Ollama model and return its reply."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
 
 
def speak(text, path):
    """Text to speech with Kokoro, written to a wav file."""
    pipeline = KPipeline(lang_code="a")
    generator = pipeline(text, voice="af_heart")
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(path, audio, 24000)
    return path
 
 
def main():
    print("=" * 60)
    print("PitchX file-based voice agent demo")
    print("=" * 60)
 
    total_start = time.time()
 
    # 1. Speech to text
    t0 = time.time()
    user_text = transcribe(INPUT_AUDIO)
    t_stt = time.time() - t0
    print(f"\n[1] Heard (STT, {t_stt:.2f}s):\n    {user_text}")
 
    # 2. Model reply
    t0 = time.time()
    reply = get_reply(user_text)
    t_llm = time.time() - t0
    print(f"\n[2] Agent reply (LLM, {t_llm:.2f}s):\n    {reply}")
 
    # 3. Text to speech
    t0 = time.time()
    speak(reply, OUTPUT_AUDIO)
    t_tts = time.time() - t0
    print(f"\n[3] Spoke reply (TTS, {t_tts:.2f}s) -> {OUTPUT_AUDIO}")
 
    total = time.time() - total_start
    print("\n" + "=" * 60)
    print("Stage timings")
    print(f"  STT (whisper):  {t_stt:.2f}s")
    print(f"  LLM (ollama):   {t_llm:.2f}s")
    print(f"  TTS (kokoro):   {t_tts:.2f}s")
    print(f"  TOTAL:          {total:.2f}s")
    print("=" * 60)
    print(f"\nDone. Play {OUTPUT_AUDIO} to hear the agent's spoken reply.")
 
 
if __name__ == "__main__":
    main()
