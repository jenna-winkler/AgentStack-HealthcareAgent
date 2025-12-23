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
    
    # Validate LLM extension
    if not llm or not llm.data:
        yield trajectory.trajectory_metadata(title="LLM Error", content="LLM extension missing.")
        yield "LLM selection is required."
        return
    
    llm_config = llm.data.llm_fulfillments.get("default")
    if not llm_config:
        yield trajectory.trajectory_metadata(title="LLM Error", content="No LLM fulfillment available.")
        yield "No LLM configuration available."
        return
    
    # Load conversation history
    history = [msg async for msg in context.load_history() if isinstance(msg, Message) and msg.parts]
    
    # Discover specialist agents deployed on platform
    available_agents = await platform.list_agents() if platform else []
    specialist_agents = {
        agent["name"]: agent 
        for agent in available_agents 
        if agent["name"] in {"PolicyAgent", "ResearchAgent", "ProviderAgent"}
    }
    
    if not specialist_agents:
        yield trajectory.trajectory_metadata(
            title="Warning",
            content=f"No specialist agents found. Available: {[a['name'] for a in available_agents]}"
        )
    
    # Create function definitions for handoffs
    tools = create_handoff_tools(specialist_agents)
    
    # System instructions
    system_prompt = """You are a friendly healthcare concierge assistant.

Your role is to help users navigate their healthcare benefits, find providers, and understand their coverage.

When to use specialist agents:
- Use PolicyAgent for questions about specific policy details, coverage, benefits, deductibles, copays
- Use ResearchAgent for medical information about symptoms, conditions, treatments, procedures
- Use ProviderAgent for finding in-network doctors, specialists, or healthcare facilities

Guidelines:
- Ask clarifying questions if the user's need is unclear
- Use specialist agents when their expertise is needed
- Synthesize information from specialists into friendly, helpful responses
- Be warm and empathetic - healthcare can be confusing and stressful"""
    
    # Build message history for LLM
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        role = "assistant" if msg.role == Role.agent else "user"
        messages.append({"role": role, "content": get_message_text(msg)})
    
    messages.append({"role": "user", "content": get_message_text(message)})
    
    # Orchestration loop
    max_iterations = 10
    final_response = ""
    
    async with httpx.AsyncClient() as http_client:
        for iteration in range(max_iterations):
            # Call LLM with streaming
            response = await http_client.post(
                f"{llm_config.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_config.api_model,
                    "messages": messages,
                    "tools": tools if specialist_agents else None,
                    "stream": True,
                    "temperature": 0.7,
                },
                timeout=60.0,
            )
            
            # Process streaming response
            assistant_message = {"role": "assistant", "content": "", "tool_calls": []}
            current_tool_call = None
            
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                
                try:
                    chunk = json.loads(line[6:])  # Remove "data: " prefix
                    delta = chunk["choices"][0]["delta"]
                    
                    # Handle content streaming
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        assistant_message["content"] += content
                        final_response += content
                        yield content
                    
                    # Handle tool calls
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            
                            # Initialize tool call if needed
                            while len(assistant_message["tool_calls"]) <= idx:
                                assistant_message["tool_calls"].append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            current_tool_call = assistant_message["tool_calls"][idx]
                            
                            if "id" in tc_delta:
                                current_tool_call["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                if "name" in tc_delta["function"]:
                                    current_tool_call["function"]["name"] += tc_delta["function"]["name"]
                                if "arguments" in tc_delta["function"]:
                                    current_tool_call["function"]["arguments"] += tc_delta["function"]["arguments"]
                
                except json.JSONDecodeError:
                    continue
            
            # If no tool calls, we're done
            if not assistant_message["tool_calls"] or not assistant_message["tool_calls"][0]["id"]:
                break
            
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message["content"] or None,
                "tool_calls": assistant_message["tool_calls"]
            })
            
            # Execute tool calls
            for tool_call in assistant_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Extract agent name from function name
                agent_name = function_name.replace("handoff_to_", "").title()
                if agent_name.lower() == "policyagent":
                    agent_name = "PolicyAgent"
                elif agent_name.lower() == "researchagent":
                    agent_name = "ResearchAgent"
                elif agent_name.lower() == "provideragent":
                    agent_name = "ProviderAgent"
                
                yield trajectory.trajectory_metadata(
                    title=f"Consulting {agent_name}",
                    content=f"Query: {arguments.get('query', '')}"
                )
                
                try:
                    # Call specialist agent
                    specialist_response = await call_specialist_agent(
                        specialist_agents[agent_name],
                        query=arguments["query"],
                        context=arguments.get("context", "")
                    )
                    
                    yield trajectory.trajectory_metadata(
                        title=f"{agent_name} Response",
                        content=specialist_response[:500] + ("..." if len(specialist_response) > 500 else "")
                    )
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": specialist_response
                    })
                    
                except Exception as e:
                    error_msg = f"Error calling {agent_name}: {str(e)}"
                    yield trajectory.trajectory_metadata(
                        title=f"{agent_name} Error",
                        content=error_msg
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": error_msg
                    })
        
        # Check if we hit max iterations
        if iteration == max_iterations - 1:
            yield trajectory.trajectory_metadata(
                title="Max Iterations Reached",
                content="Orchestration completed at maximum iteration limit."
            )
    
    # Store final response in context
    if final_response:
        await context.store(AgentMessage(text=final_response))


def run() -> None:
    """Start the AgentStack server."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
    
