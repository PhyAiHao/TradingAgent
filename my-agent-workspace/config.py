"""
config.py  (workspace root)
============================
Single source of truth for ALL adjustable parameters.

Two sections:
  1. Agent / infrastructure  — Telegram, Alpaca API, shell safety, etc.
  2. Active inference trading — inference engine, MDP structure, execution

Edit values here. No other file should hardcode these values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===========================================================================
# Agent / infrastructure
# ===========================================================================

# ── Upsonic agent ─────────────────────────────────────────────────────────
AGENT_MODEL = "anthropic/claude-sonnet-4-6"

WORKSPACE = os.path.expanduser(
    "/Users/phyaihao/Desktop/new ideas related paper/AI model and Intellegence/Agent with Active Inference/my-agent-workspace"
)
MEMORY_DIR = os.path.join(WORKSPACE, "memory")

# ── Telegram ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")

# ── Anthropic / Vision ────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VISION_MODEL      = os.getenv("VISION_MODEL", "claude-opus-4-6")

# ── Alpaca ────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")

# ── Alpaca risk limits ────────────────────────────────────────────────────
MAX_ORDER_VALUE_USD  = float(os.getenv("MAX_ORDER_VALUE_USD",  "1000.0"))
MAX_OPEN_POSITIONS   = int(os.getenv("MAX_OPEN_POSITIONS",     "10"))
DEFAULT_ORDER_TYPE   = os.getenv("DEFAULT_ORDER_TYPE",         "market")
DEFAULT_TIME_IN_FORCE = os.getenv("DEFAULT_TIME_IN_FORCE",     "day")
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT",        "0.02"))
TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT",      "0.05"))

# ── Shell safety ──────────────────────────────────────────────────────────
SHELL_BLOCKED = [
    "rm -rf", "rmdir", "mkfs", "dd if=", "shutdown", "reboot",
    "kill -9", "killall", ":(){:|:&};:", "chmod 777", "sudo",
    "curl | bash", "wget | sh", "> /dev/", "format",
]
SHELL_SYSTEM_COMMANDS = [
    "brew", "pip", "pip3", "python", "python3", "node", "npm",
    "git clone", "ssh", "scp",
]
BLOCKED_COMMAND_PATTERNS = [
    r"rm\s+-[rRfF]+",
    r">\s*/dev/",
    r"mkfs",
    r"dd\s+if=",
    r"shutdown",
    r"reboot",
    r":()\s*\{",
]

# ===========================================================================
# Active inference — VB engine
# (previously hardcoded in trading_model.py)
# ===========================================================================

# Number of VB gradient-descent iterations per timestep.
# Lower than auditory model (32) because market inference is noisier.
NI = int(os.getenv("NI", "16"))

# Gradient-descent time constant.
TAU = float(os.getenv("TAU", "4.0"))

# Action precision — higher = more deterministic policy selection.
ALPHA = float(os.getenv("ALPHA", "512.0"))

# Prior precision over policies.
BETA = float(os.getenv("BETA", "1.0"))

# Belief-reset parameter (1.0 = no reset between bars).
ERP = float(os.getenv("ERP", "1.0"))

# Learning rate for Dirichlet updates.
# Slower than auditory (1.0) — market regimes are more stable.
ETA = float(os.getenv("ETA", "0.5"))

# Forgetting rate (1.0 = no forgetting; 0.98 = slight forgetting per cycle).
OMEGA = float(os.getenv("OMEGA", "0.98"))

# Concentration scale for learnable a matrices.
CONCENTRATION = float(os.getenv("CONCENTRATION", "64.0"))

# ===========================================================================
# Active inference — MDP structure
# ===========================================================================

# Number of price bars per decision window.
T = int(os.getenv("T", "8"))

# Number of trading policies: hold / buy / sell.
N_POLICIES = int(os.getenv("N_POLICIES", "3"))

# Number of market regime hidden states in L2.
N_REGIME = int(os.getenv("N_REGIME", "4"))

# Number of position states: flat / long / short.
N_POSITION = int(os.getenv("N_POSITION", "3"))

# Freeze scale for time-in-window and position D priors
# (prevents learning in deterministic clock factors).
D_FREEZE_SCALE = float(os.getenv("D_FREEZE_SCALE", "100.0"))

# ===========================================================================
# Active inference — preferences (C matrix)
# ===========================================================================

# How strongly to prefer profit at the last bar.
# Increased by LLM reflection after loss streaks.
PREF_PROFIT = float(os.getenv("PREF_PROFIT", "2.0"))

# How strongly to avoid loss. Make more negative to increase risk aversion.
PREF_LOSS   = float(os.getenv("PREF_LOSS",  "-3.0"))

# Preference for flat/hold outcome (neutral reference).
PREF_FLAT   = float(os.getenv("PREF_FLAT",   "0.0"))

# ===========================================================================
# Active inference — execution & market data
# ===========================================================================

# Default symbol traded by the heartbeat and Telegram tools.
SYMBOL = os.getenv("SYMBOL", "SPY")

# Bar timeframe passed to Alpaca.
TIMEFRAME = os.getenv("TIMEFRAME", "15Min")

# Number of historical bars to fetch per cycle.
BAR_LIMIT = int(os.getenv("BAR_LIMIT", "50"))

# Backward-compatible aliases — alpaca_tools.py uses these names
DEFAULT_TIMEFRAME = TIMEFRAME
DEFAULT_BAR_LIMIT = BAR_LIMIT

# Shares per order.
QTY = float(os.getenv("QTY", "1.0"))

# Minimum policy posterior required before executing a trade.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.65"))

# ===========================================================================
# Active inference — LLM / memory
# ===========================================================================

# Anthropic model used for L1 estimation and reflection.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Number of past trades included in the LLM reflection prompt.
REFLECTION_WINDOW = int(os.getenv("REFLECTION_WINDOW", "10"))

# Hard limits on number of hidden states the LLM can propose.
MIN_STATES = int(os.getenv("MIN_STATES", "2"))
MAX_STATES = int(os.getenv("MAX_STATES", "8"))

# Heartbeat interval in seconds (900 = 15 min).
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "900"))

# Set to "false" to run 24/7 (e.g. crypto).
MARKET_HOURS_ONLY = os.getenv("MARKET_HOURS_ONLY", "true").lower() == "true"

# ===========================================================================
# Validation
# ===========================================================================

def validate_config():
    """Raise EnvironmentError if any required .env variable is missing."""
    required = [
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_URL",
        "ANTHROPIC_API_KEY",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please add them to your .env file."
        )

if __name__ == "__main__":
    validate_config()
    print("✅ Config loaded successfully.")
    print(f"   Symbol      : {SYMBOL}  ({TIMEFRAME}  {BAR_LIMIT} bars)")
    print(f"   LLM model   : {LLM_MODEL}")
    print(f"   NI={NI}  T={T}  N_regime={N_REGIME}  N_policies={N_POLICIES}")
    print(f"   Confidence  : {MIN_CONFIDENCE}  QTY={QTY}")
    print(f"   Alpaca URL  : {ALPACA_BASE_URL}")
