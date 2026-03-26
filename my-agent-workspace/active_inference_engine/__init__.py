try:
    from .trading_model import TRADING_TOOLS
except Exception as e:
    print(f"[active_inference_engine] Warning: trading tools not loaded ({e})")
    TRADING_TOOLS = []
