import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os

load_dotenv()


openrouter_gemini = OpenAIChatCompletionClient(
    model="google/gemini-2.0-flash-001",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_info=ModelInfo(
        vision=False, function_calling=True, json_output=False, family="unknown"
    ),
)


agent = AssistantAgent("assistant", model_client=openrouter_gemini)


async def main() -> None:
    await Console(
        agent.run_stream(
            task="I want you to introduce yourself to me in a verbose manner"
        )
    )
    await openrouter_gemini.close()


asyncio.run(main())
