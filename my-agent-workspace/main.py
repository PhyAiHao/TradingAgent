"""
main.py
=======
Upsonic agent — Telegram interface only.
The autonomous trading heartbeat runs separately in heartbeat.py.

Start both processes:
    Terminal 1:  python3 main.py
    Terminal 2:  python3 heartbeat.py

System prompt is loaded from Agent.md in the workspace root.
Edit Agent.md to change agent behaviour without touching this file.

KEY DESIGN DECISION:
    The agent uses TRADING_AGENT_TOOLS, NOT ALL_TOOLS.
    ALL_TOOLS includes shell_tool, terminal_tool, screen_tool — when
    the LLM sees those, it writes Python files and runs terminals instead
    of calling the direct Alpaca functions. TRADING_AGENT_TOOLS excludes
    them, forcing the agent to use active_inference_trade directly.
"""

import os
from dotenv import load_dotenv
from upsonic import AutonomousAgent
from upsonic.interfaces import InterfaceManager, TelegramInterface, InterfaceMode

from tools import TRADING_AGENT_TOOLS
from config import validate_config, AGENT_MODEL, WORKSPACE

load_dotenv()
validate_config()

# ── Load system prompt from Agent.md ─────────────────────────────
_AGENT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Agent.md")
if os.path.exists(_AGENT_MD):
    with open(_AGENT_MD, encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    print(f"   System prompt loaded from Agent.md ({len(SYSTEM_PROMPT)} chars)")
else:
    SYSTEM_PROMPT = (
        "You are a quantitative trading AI agent. "
        "When asked to trade, call active_inference_trade directly. "
        "Never write code or use shell tools."
    )
    print("   WARNING: Agent.md not found — using fallback system prompt")

# ── Agent ─────────────────────────────────────────────────────────
agent = AutonomousAgent(
    model=AGENT_MODEL,
    workspace=WORKSPACE,
    tools=TRADING_AGENT_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# ── Telegram interface ────────────────────────────────────────────
telegram = TelegramInterface(
    agent=agent,
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    webhook_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
    mode=InterfaceMode.CHAT,
    reset_command="/reset",
)

manager = InterfaceManager(interfaces=[telegram])

if __name__ == "__main__":
    print("🤖 Trading Agent starting...")
    print(f"   Model : {AGENT_MODEL}")
    print(f"   Tools : {[t.__name__ for t in TRADING_AGENT_TOOLS]}")
    print(f"   Shell/Terminal tools : EXCLUDED")
    manager.serve(host="0.0.0.0", port=8000)
