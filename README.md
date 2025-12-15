# AgentStack-HealthcareAgent
An healthcare based example of how to build and deploy an A2A Agent that calls other A2A Agents on the open source platform Agent Stack by IBM Research.

## Gemini policy agent server
- Agent uses the platform-provided LLM extension (no local `GEMINI_API_KEY` required). Set `AGENT_HOST` / `POLICY_AGENT_PORT` in `.env` if you need non-default values. Dependency: `agentstack-sdk`.
- Start the server with `uv run python -m agents.policy_agent` from the repo root; it will expose the AgentStack serve endpoint on `http://0.0.0.0:9999` by default so AgentStack can discover it.
