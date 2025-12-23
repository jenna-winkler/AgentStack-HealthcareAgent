import json
import os
from typing import Annotated

from a2a.types import Message, Role
from a2a.utils.message import get_message_text
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.a2a.extensions import (
    PlatformApiExtensionServer,
    PlatformApiExtensionSpec,
    AgentDetail,
    AgentDetailContributor,
    AgentDetailTool,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from agentstack_sdk.a2a import A2AClient
import httpx


server = Server()


def create_handoff_tools(specialist_agents: dict) -> list:
    """Create OpenAI function definitions for specialist agent handoffs."""
    tools = []
    
    handoff_descriptions = {
        "PolicyAgent": "Hand off to PolicyAgent for specific questions about the user's insurance policy details, coverage, benefits, deductibles, copays, and plan documents.",
        "ResearchAgent": "Hand off to ResearchAgent for medical information about symptoms, health conditions, treatments, procedures, or medications using up-to-date web resources.",
        "ProviderAgent": "Hand off to ProviderAgent for information about in-network healthcare providers, doctor search, and provider directory lookups."
    }
    
    for agent_name in specialist_agents.keys():
        tools.append({
            "type": "function",
            "function": {
                "name": f"handoff_to_{agent_name.lower()}",
                "description": handoff_descriptions.get(agent_name, f"Hand off to {agent_name}"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific query or question to send to the specialist agent"
                        },
                        "context": {
                            "type": "string",
                            "description": "Relevant context from the current conversation (optional)"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
    
    return tools


async def call_specialist_agent(agent_info: dict, query: str, context: str = "") -> str:
    """Call a specialist agent via A2A protocol."""
    full_query = f"Context: {context}\n\nQuery: {query}" if context else query
    
    # Use A2A client to call the agent
    client = A2AClient(base_url=agent_info["url"])
    response = await client.send_message(
        agent_id=agent_info["id"],
        message=full_query
    )
    
    return response.text if hasattr(response, 'text') else str(response)


@server.agent(
    name="Healthcare Concierge [Jenna]",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Hi there! I can help navigate benefits, providers, and coverage details.",
        input_placeholder="Ask a healthcare question...",
        programming_language="Python",
        framework="AgentStack SDK",
        contributors=[
            AgentDetailContributor(
                name="Sandi Besen and Ken Ocheltree",
                email="name@example.com",
            )
        ],
        tools=[
            AgentDetailTool(
                name="PolicyAgent",
                description="Consults specialist for policy details",
            ),
            AgentDetailTool(
                name="ResearchAgent",
                description="Consults specialist for medical information",
            ),
            AgentDetailTool(
                name="ProviderAgent",
                description="Consults specialist for provider information",
            ),
        ],
    ),
)
async def healthcare_concierge(
    message: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-2.5-flash-lite",)),
    ],
    platform: Annotated[PlatformApiExtensionServer, PlatformApiExtensionSpec()],
):
    """Healthcare concierge agent that orchestrates specialist agents."""
    
    yield trajectory.trajectory_metadata(
        title="Initializing Agent...",
        content="Setting up your Healthcare Concierge.",
    )
