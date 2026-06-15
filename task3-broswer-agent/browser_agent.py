import asyncio
from browser_use import Agent
from browser_use.llm import ChatOllama


async def main():
    # Connect browser-use to the local Ollama model running on the pod
    llm = ChatOllama(
        model="llama3.1:8b",
        host="http://localhost:11434",
    )

    agent = Agent(
        task=(
            "Navigate to https://en.wikipedia.org/wiki/OpenAI. Read the page. "
            "In one sentence, describe what OpenAI does. Once you have this, "
            "immediately mark the task done and stop."
        ),
        llm=llm,
        use_vision=False,   # text-only model: do not send screenshots
    )

    result = await agent.run(max_steps=10)   # hard cap to prevent runaway loops
    print("\n\n=== AGENT RESULT ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
