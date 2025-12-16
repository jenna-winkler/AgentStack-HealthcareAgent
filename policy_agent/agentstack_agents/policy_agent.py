import asyncio
import os
from pathlib import Path
from typing import Annotated, Optional

import google.generativeai as genai
from a2a.types import Message
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer, LLMServiceExtensionSpec
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from dotenv import load_dotenv


class PolicyAgent:
    """
    A policy agent that reads a benefits PDF and answers coverage questions.
    Original logic preserved in spirit: load PDF, add it to the prompt, return concise answers.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        system_prompt: str = (
            "You are an expert insurance agent designed to assist with coverage queries. "
            "Use the provided documents to answer questions about insurance policies. "
            "If the information is not available in the documents, respond with \"I don't know\"."
        ),
        max_output_tokens: int = 1024,
    ) -> None:
        # Ensure .env values are picked up even when the working directory differs.
        self._load_env()

        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt

        # Default to the same PDF used by the original policy agent
        self.pdf_path = Path(pdf_path) if pdf_path else Path(__file__).resolve().parent / "2026AnthemgHIPSBC.pdf"
        self.pdf_bytes = self._load_pdf(self.pdf_path)

    @staticmethod
    def _load_env() -> None:
        """
        Load environment variables from the repo .env file plus the current working directory.

        This handles cases where the process is started outside the project root.
        """
        project_env = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(project_env)
        load_dotenv()

    @staticmethod
    def _load_pdf(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {path.resolve()}")
        return path.read_bytes()

    def answer_query(self, prompt: str, api_key_override: Optional[str] = None) -> str:
        """
        Send the user prompt and the embedded PDF to Gemini and return the text response.
        """
        if not api_key_override:
            return "LLM service not available. Please enable the LLM extension for this agent."

        # Configure Gemini using the platform-provided API key.
        genai.configure(api_key=api_key_override)
        model = genai.GenerativeModel(self.model_name)

        parts = [
            {"text": self.system_prompt},
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": self.pdf_bytes,
                }
            },
            {"text": prompt},
        ]

        response = model.generate_content(
            parts,
            generation_config={"max_output_tokens": self.max_output_tokens},
        )

        if not response or not response.text:
            return "I don't know"

        # Maintain escaping behavior similar to the original agent
        return response.text.replace("$", r"\\$")


load_dotenv()

server = Server()
policy_agent = PolicyAgent()


@server.agent(
    name="Policy Agent",
)
async def policy_agent_wraper(
    input: Message,
    context: RunContext,
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash",)),
    ] = None,
):
    """Wrapper around the policy agent."""
    prompt = get_message_text(input)
    llm_config = None

    if llm and llm.data and llm.data.llm_fulfillments:
        llm_config = llm.data.llm_fulfillments.get("default")

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        policy_agent.answer_query,
        prompt,
        llm_config.api_key if llm_config else None,
    )
    yield AgentMessage(text=response)


def run() -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("POLICY_AGENT_PORT", 9999))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
