"""
alpaca_tools.py
───────────────
Alpaca Paper-Trading tool functions for the Upsonic AutonomousAgent.
Each public function is a self-contained "skill" the agent can call.

Dependencies:
    pip install alpaca-py python-dotenv
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
from typing import Optional

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ALPACA_DATA_FEED,
    MAX_ORDER_VALUE_USD,
    DEFAULT_TIME_IN_FORCE,
    DEFAULT_BAR_LIMIT,
    DEFAULT_TIMEFRAME,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

# ──────────────────────────────────────────
# Client Initialisation (shared singletons)
# ──────────────────────────────────────────

_trading_client: Optional[TradingClient] = None
_data_client: Optional[StockHistoricalDataClient] = None


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        paper = "paper" in ALPACA_BASE_URL
        _trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=paper,
        )
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
    return _data_client


# ──────────────────────────────────────────
# Account Tools
# ──────────────────────────────────────────

def get_account_info() -> dict:
    """
    Return key account metrics: cash, portfolio value, buying power,
    equity, and day-trade count.

    Returns:
        dict with keys: cash, portfolio_value, buying_power, equity,
                        daytrade_count, pattern_day_trader
    """
    client = _get_trading_client()
    acct = client.get_account()
    return {
        "cash": float(acct.cash),
        "portfolio_value": float(acct.portfolio_value),
        "buying_power": float(acct.buying_power),
        "equity": float(acct.equity),
        "daytrade_count": acct.daytrade_count,
        "pattern_day_trader": acct.pattern_day_trader,
        "status": acct.status,
    }


# ──────────────────────────────────────────
# Market Data Tools
# ──────────────────────────────────────────

def get_latest_quote(symbol: str) -> dict:
    """
    Fetch the latest bid/ask quote for a stock symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys: symbol, ask_price, bid_price, ask_size, bid_size
    """
    client = _get_data_client()
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=ALPACA_DATA_FEED)
    quotes = client.get_stock_latest_quote(req)
    q = quotes[symbol]
    return {
        "symbol": symbol,
        "ask_price": float(q.ask_price),
        "bid_price": float(q.bid_price),
        "ask_size": q.ask_size,
        "bid_size": q.bid_size,
    }


def get_historical_bars(
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_BAR_LIMIT,
) -> list[dict]:
    """
    Fetch OHLCV bars for a symbol.

    Args:
        symbol:    Ticker, e.g. "TSLA"
        timeframe: "1Min" | "5Min" | "15Min" | "1Hour" | "1Day"
        limit:     Number of bars to return (default from config)

    Returns:
        List of dicts with keys: timestamp, open, high, low, close, volume
    """
    tf_map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))
    end   = datetime.utcnow()
    start = end - timedelta(days=limit * 2)  # generous window

    client = _get_data_client()
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        limit=limit,
        feed=ALPACA_DATA_FEED,
    )
    bars = client.get_stock_bars(req)[symbol]
    return [
        {
            "timestamp": str(b.timestamp),
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": int(b.volume),
        }
        for b in bars
    ]


# ──────────────────────────────────────────
# Order Tools
# ──────────────────────────────────────────

def place_market_order(
    symbol: str,
    qty: float,
    side: str = "buy",
    time_in_force: str = DEFAULT_TIME_IN_FORCE,
) -> dict:
    """
    Place a market order with a safety value cap.

    Args:
        symbol:        Ticker, e.g. "AAPL"
        qty:           Number of shares (fractional allowed)
        side:          "buy" or "sell"
        time_in_force: "day" | "gtc" | "ioc" | "fok"

    Returns:
        dict with order id, status, symbol, qty, side
    """
    # Safety check — estimate value using latest quote
    quote = get_latest_quote(symbol)
    estimated_value = quote["ask_price"] * qty
    if estimated_value > MAX_ORDER_VALUE_USD:
        raise ValueError(
            f"Order value ${estimated_value:.2f} exceeds MAX_ORDER_VALUE_USD "
            f"(${MAX_ORDER_VALUE_USD}). Reduce qty or update config."
        )

    client = _get_trading_client()
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce(time_in_force),
    )
    order = client.submit_order(req)
    return {
        "id":             str(order.id),
        "client_order_id": str(order.client_order_id),
        "symbol":         order.symbol,
        "qty":            float(order.qty),
        "side":           str(order.side),
        "type":           str(order.order_type),
        "status":         str(order.status),
        "submitted_at":   str(order.submitted_at),
    }


def place_limit_order(
    symbol: str,
    qty: float,
    limit_price: float,
    side: str = "buy",
    time_in_force: str = "gtc",
) -> dict:
    """
    Place a limit order.

    Args:
        symbol:      Ticker
        qty:         Shares
        limit_price: Target price
        side:        "buy" or "sell"
        time_in_force: "day" | "gtc" | "ioc" | "fok"

    Returns:
        dict with order details
    """
    client = _get_trading_client()
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        limit_price=limit_price,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce(time_in_force),
    )
    order = client.submit_order(req)
    return {
        "id":          str(order.id),
        "symbol":      order.symbol,
        "qty":         float(order.qty),
        "limit_price": float(order.limit_price),
        "side":        str(order.side),
        "status":      str(order.status),
    }


def cancel_order(order_id: str) -> dict:
    """
    Cancel an open order by its ID.

    Args:
        order_id: UUID string of the order

    Returns:
        dict confirming cancellation
    """
    client = _get_trading_client()
    client.cancel_order_by_id(order_id)
    return {"cancelled_order_id": order_id, "status": "cancel_requested"}


def cancel_all_orders() -> dict:
    """
    Cancel ALL open orders.

    Returns:
        dict with count of cancelled orders
    """
    client = _get_trading_client()
    cancelled = client.cancel_orders()
    return {"cancelled_count": len(cancelled)}


def get_orders(status: str = "open") -> list[dict]:
    """
    List orders filtered by status.

    Args:
        status: "open" | "closed" | "all"

    Returns:
        List of order dicts
    """
    status_map = {
        "open":   QueryOrderStatus.OPEN,
        "closed": QueryOrderStatus.CLOSED,
        "all":    QueryOrderStatus.ALL,
    }
    client = _get_trading_client()
    req = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
    orders = client.get_orders(req)
    return [
        {
            "id":     str(o.id),
            "symbol": o.symbol,
            "qty":    float(o.qty),
            "side":   str(o.side),
            "type":   str(o.order_type),
            "status": str(o.status),
        }
        for o in orders
    ]


# ──────────────────────────────────────────
# Position Tools
# ──────────────────────────────────────────

def get_all_positions() -> list[dict]:
    """
    Return all current open positions.

    Returns:
        List of dicts: symbol, qty, avg_entry_price, current_price,
                       market_value, unrealized_pl, unrealized_plpc
    """
    client = _get_trading_client()
    positions = client.get_all_positions()
    return [
        {
            "symbol":          p.symbol,
            "qty":             float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price":   float(p.current_price),
            "market_value":    float(p.market_value),
            "unrealized_pl":   float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "side":            str(p.side),
        }
        for p in positions
    ]


def get_position(symbol: str) -> dict:
    """
    Return position details for a single symbol.

    Args:
        symbol: Ticker, e.g. "NVDA"

    Returns:
        Position dict or {"error": "no position"} if not held
    """
    client = _get_trading_client()
    try:
        p = client.get_open_position(symbol)
        return {
            "symbol":          p.symbol,
            "qty":             float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price":   float(p.current_price),
            "market_value":    float(p.market_value),
            "unrealized_pl":   float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        }
    except Exception:
        return {"error": f"No open position for {symbol}"}


def close_position(symbol: str, qty: Optional[float] = None) -> dict:
    """
    Close a position fully or partially.

    Args:
        symbol: Ticker
        qty:    Shares to sell; omit to close entire position

    Returns:
        dict with order details of the closing order
    """
    client = _get_trading_client()
    req = ClosePositionRequest(qty=str(qty)) if qty else None
    order = client.close_position(symbol, close_options=req)
    return {
        "id":     str(order.id),
        "symbol": order.symbol,
        "qty":    float(order.qty),
        "side":   str(order.side),
        "status": str(order.status),
    }


def close_all_positions() -> dict:
    """
    Liquidate every open position immediately.

    Returns:
        dict with count of closed positions
    """
    client = _get_trading_client()
    closed = client.close_all_positions(cancel_orders=True)
    return {"closed_count": len(closed)}


# ──────────────────────────────────────────
# Portfolio Summary Tool
# ──────────────────────────────────────────

def get_portfolio_summary() -> dict:
    """
    Combined snapshot: account info + all positions + open orders.
    Useful as a single "status check" skill for the agent.

    Returns:
        dict with keys: account, positions, open_orders
    """
    return {
        "account":     get_account_info(),
        "positions":   get_all_positions(),
        "open_orders": get_orders(status="open"),
    }
