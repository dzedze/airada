from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from src.prompts.prompts import return_instructions
from src.tools.tools_github import search_github_repos
from src.tools.tools_papers import search_ai_papers
from src.tools.tools_news import get_ai_news
from src.providers.llm_factory import create_chat_agent

# Initialize the LLM
chat_agent = create_chat_agent(model="gpt-4o")

# Our three tools
tools = [search_github_repos, search_ai_papers, get_ai_news]

# System prompt
instructions = return_instructions()

# Pre-bind tools
agent_with_tools = chat_agent.bind_tools(tools)

# InMemorySaver keeps state in RAM for the lifetime of the process (sufficient
# for short-term / in-session memory).
checkpointer = InMemorySaver()

# ── Context window settings ───────────────────────────────────────────────────
MAX_HISTORY_TOKENS = (
    4_000  # ~last 8-12 conversational turns depending on length
)


def _trim(messages: list) -> list:
    """
    Trim conversation history to MAX_HISTORY_TOKENS using LangChain's
    official trim_messages utility.
    """
    trimmed = trim_messages(
        messages,
        strategy="last",  # keep the tail (most recent)
        token_counter=count_tokens_approximately,
        max_tokens=MAX_HISTORY_TOKENS,
        start_on="human",  # window must start with a user msg
        end_on=(
            "human",
            "tool",
        ),  # window must end on user or tool msg
        include_system=False,  # we inject system prompt separately
    )

    if len(trimmed) < len(messages):
        dropped = len(messages) - len(trimmed)
        print(
            f"[Memory] Context window limit reached — dropped {dropped} oldest "
            f"message(s), keeping ~{MAX_HISTORY_TOKENS:,} tokens of recent history."
        )

    return trimmed


# ── Graph nodes ───────────────────────────────────────────────────────────────


async def call_model(state: MessagesState):
    """
    Main reasoning node — the only place the LLM is called.

    The checkpointer (InMemorySaver) has already restored the full message
    history from the checkpoint into state["messages"] before this node runs.
    This node's job:
      1. Trim the restored history to fit the context window.
      2. Prepend the system prompt (injected fresh every call, never stored
         in the checkpoint so it doesn't accumulate duplicates).
      3. Call the LLM with tools bound and return its response.

    The returned response is appended to state["messages"] by LangGraph and
    saved to the checkpoint automatically — no manual history management needed.
    """
    trimmed = _trim(state["messages"])

    response = await agent_with_tools.ainvoke(
        [SystemMessage(content=instructions)] + trimmed
    )
    return {"messages": [response]}


# ── Graph definition ──────────────────────────────────────────────────────────


def get_graph():
    """
    Build and compile the LangGraph agent with InMemorySaver checkpointing.
    """
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    # Attach checkpointer for shor-term memory
    return builder.compile(checkpointer=checkpointer)
