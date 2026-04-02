import uuid
import gradio as gr
from langchain_core.messages import HumanMessage

from src.main import get_graph

import os

print("AIRADA - Your AI Radar. Zero Hype. Maximum Signal....")
print("Please wait while services initialise...")

# stop langsmith tracing, not required for this assignment
# prevent HTTPError messages at runtime
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

print("Initialising agent...")
llm_graph = get_graph()
print("Agent ready!")
print("Starting Gradio interface...\n")


# ── Session state ─────────────────────────────────────────────────────────────
# Each browser session gets a unique thread_id on first load.
# The LangGraph checkpointer uses this ID to store and retrieve the full
# conversation state — Gradio's `history` list is only used for rendering
# the chat UI, not for feeding context into the model.


def _make_thread_id() -> str:
    return f"aria-{uuid.uuid4().hex[:8]}"


async def chat(
    message: str, history: list, thread_id: str
) -> tuple[str, str]:
    """
    Gradio calls this on every user message.

    Args:
        message:   The new user input.
        history:   Gradio's display history (used only for rendering).
        thread_id: The session's unique LangGraph thread ID (gr.State).

    Memory model:
        - The checkpointer in main.py holds the real conversation state,
          keyed by thread_id.
        - We send ONLY the new user message to graph.invoke(); LangGraph
          automatically loads the prior checkpoint and appends to it.
        - When the conversation grows beyond MAX_HISTORY_TOKENS,
          _trim() in main.py silently drops the oldest messages before the
          LLM call — the checkpoint still holds the full history, but only
          the recent window is sent to the model.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Send only the new message — the checkpointer provides the rest
    response_state = await llm_graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    reply = response_state["messages"][-1].content
    return reply, thread_id


# ── Gradio UI ──────────────────────────────────────────────────────────────────

_DESCRIPTION = """
Your AI Radar. Zero Hype. Maximum Signal.

| Service | What I cover |
|---------|-------------|
| 🐙 **GitHub Trending** | Top repos on AI agents, LLMs, RAG, agentic AI & more |
| 📄 **AI Paper Knowledge** | Semantic search over ~8,500 arXiv papers |
| 📰 **AI/LLM News** | Live headlines from VentureBeat, MIT Tech Review, The Decoder |

> ⚠️ I only answer from those three services. If it's outside my scope, I'll say so.
"""

# examples list for gradio chat
_EXAMPLES = [
    ["Show me the top 5 AI/LLM GitHub projects"],
    ["Give me repos with the topic agentic-ai"],
    ["Compare GPT-3 and PaLM papers"],
    ["Summarize top LLM research trends"],
    ["What's new in AI this week?"],
    ["What's happening in agentic AI right now?"],
]

# gr.State holds the thread_id for each browser session independently
thread_id_state = gr.State(_make_thread_id)

chat_ui = gr.ChatInterface(
    fn=chat,
    additional_inputs=[thread_id_state],
    additional_outputs=[thread_id_state],
    title="👾 AIRADA",
    description=_DESCRIPTION,
    examples=_EXAMPLES,
    # type="messages",
    # theme=gr.themes.Soft(),
    chatbot=gr.Chatbot(
        placeholder="<strong>AIRADA is online.</strong> What's on your radar? 🔍",
        # show_copy_button=True,
        render_markdown=True,
        height=480,
        # type="messages",   # must match ChatInterface type="messages"
    ),
    textbox=gr.Textbox(
        placeholder="Ask about AI repos, research papers, or the latest AI news...",
        container=False,
    ),
)

if __name__ == "__main__":
    print("=" * 55)
    print("  👾 AIRADA is online. What's on your radar? 🔍")
    print("=" * 55)
    chat_ui.launch()
