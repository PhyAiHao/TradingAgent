from .web_search_tool import web_search_tool
from .shell_tool      import shell_tool
from .terminal_tool   import terminal_tool
from .docx_tool       import docx_tool
from .screen_tool     import screen_tool

from .alpaca_tools import (
    get_account_info,
    get_latest_quote,
    get_historical_bars,
    place_market_order,
    place_limit_order,
    cancel_order,
    cancel_all_orders,
    get_orders,
    get_all_positions,
    get_position,
    close_position,
    close_all_positions,
    get_portfolio_summary,
)

try:
    from active_inference_engine import TRADING_TOOLS
except Exception as e:
    print(f"[tools] Warning: active inference tools not loaded ({e})")
    TRADING_TOOLS = []

# ALL_TOOLS — full set for developer/admin use
ALL_TOOLS = [
    web_search_tool,
    shell_tool,
    terminal_tool,
    docx_tool,
    screen_tool,
    get_account_info,
    get_latest_quote,
    get_historical_bars,
    place_market_order,
    place_limit_order,
    cancel_order,
    cancel_all_orders,
    get_orders,
    get_all_positions,
    get_position,
    close_position,
    close_all_positions,
    get_portfolio_summary,
    *TRADING_TOOLS,
]

# TRADING_AGENT_TOOLS — only what the Telegram trading agent should see.
#
# Excludes shell_tool, terminal_tool, screen_tool, docx_tool deliberately:
# their presence causes the LLM to write Python scripts and run terminals
# instead of calling Alpaca functions directly.
#
# Includes direct Alpaca functions so the agent can:
#   - Check account balance before trading
#   - Query positions and open orders
#   - Place and cancel orders directly
#   - Run active_inference_signal / active_inference_trade (from TRADING_TOOLS)
TRADING_AGENT_TOOLS = [
    web_search_tool,        # market news research
    get_account_info,       # check balance / buying power
    get_latest_quote,       # get current price
    get_all_positions,      # see open positions
    get_position,           # check a specific position
    get_orders,             # see open orders
    get_portfolio_summary,  # full snapshot
    place_market_order,     # direct order placement (fallback)
    cancel_order,           # cancel a specific order
    close_position,         # close a position
    close_all_positions,    # emergency liquidation
    *TRADING_TOOLS,         # active_inference_signal + active_inference_trade
]
