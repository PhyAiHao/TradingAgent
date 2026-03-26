"""
tools/web_search_tool.py
────────────────────────
Skill 1 — DuckDuckGo web search.

Install: pip install ddgs
"""

from datetime import datetime

try:
    from upsonic.tools import tool
    _HAS_UPSONIC = True
except ImportError:
    _HAS_UPSONIC = False
    def tool(fn):           # no-op decorator when upsonic is absent
        return fn


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return results.

    Args:
        query:       The search query.
        max_results: Number of results to return (default 5, max 10).
    Returns:
        Formatted results with title, snippet, and URL per result.
    """
    return run_web_search(query, max_results)


def run_web_search(query: str, max_results: int = 5) -> str:
    """Plain Python DuckDuckGo search — can also be called by the heartbeat loop."""

    # ── DEBUG: proves the real function was called, not LLM memory ──
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[web_search] 🔍 LIVE CALL at {timestamp} | query='{query}' | max={max_results}",
          flush=True)
    # ────────────────────────────────────────────────────────────────

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        print(f"[web_search] ✅ Got {len(results)} results", flush=True)

        if not results:
            return f"No results for: {query}"
        lines = [f"Web search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   {r.get('body', '')}")
            lines.append(f"   {r.get('href', '')}")
        return "\n".join(lines)
    except ImportError:
        return "Web search unavailable — run: pip install ddgs"
    except Exception as e:
        print(f"[web_search] ❌ Error: {e}", flush=True)
        return f"Web search error: {e}"
