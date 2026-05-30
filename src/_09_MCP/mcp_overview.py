"""MCP learning map for LLM application integrations.

MCP, the Model Context Protocol, gives LLM clients a standard way to connect
to tools, resources, prompts, and external systems. In this roadmap it belongs
after LLMOps because it builds on the full application lifecycle: tracing,
evaluation, deployment, permissions, and operational safety.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPComponent:
    name: str
    purpose: str
    examples: tuple[str, ...]


def mcp_learning_map() -> list[MCPComponent]:
    """Return the core MCP concepts this project should teach."""

    return [
        MCPComponent(
            name="Servers",
            purpose="Expose tools, resources, and prompts to an LLM client.",
            examples=("filesystem server", "database server", "internal API server"),
        ),
        MCPComponent(
            name="Tools",
            purpose="Let the model request actions with typed inputs and outputs.",
            examples=("search_docs", "create_ticket", "query_vector_index"),
        ),
        MCPComponent(
            name="Resources",
            purpose="Expose readable context such as files, schemas, reports, or traces.",
            examples=("README.md", "evaluation report", "database schema"),
        ),
        MCPComponent(
            name="Prompts",
            purpose="Package reusable prompt workflows for common tasks.",
            examples=("judge_answer", "summarize_trace", "generate_rag_query"),
        ),
        MCPComponent(
            name="Client Integration",
            purpose="Connect an LLM application to MCP servers with permission boundaries.",
            examples=("local development", "CI assistant", "production debugging assistant"),
        ),
    ]

