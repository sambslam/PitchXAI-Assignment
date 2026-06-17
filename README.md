# PitchX Assignment

Three tasks built on a RunPod GPU pod (RTX 6000 Ada), using a local Llama 3.1 8B model served through Ollama. Each task has its own folder with a Colab notebook documenting what was done, the setup, and the results.

## Task 1 - Real-time voice agent
A phone-style voice agent: speech in, transcribe, model reply, speech out. Built with Pipecat using faster-whisper, Ollama, and Kokoro. Includes a working file-based demo and the full live pipeline. See `task1-voice-agent/`.

## Task 2 - Research assistant model
A sales-research assistant built on Ollama using a Modelfile and system prompt. See `task2-research-finetune/`.

## Task 3 - Browser agent
An autonomous browser agent using browser-use driven by the local model, completing web tasks on its own. See `task3-browser-agent/`.

## Setup
All three run on a RunPod pod with Ollama and the llama3.1:8b model. Each notebook lists the specific install steps for that task. The work runs on the pod; the notebooks are documentation and can be re-executed on RunPod.
