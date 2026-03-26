"""
tools/terminal_tool.py
──────────────────────
Skill 3 — Open a visible Terminal.app window and run a command in it.
macOS only (uses AppleScript via osascript). No extra install needed.
"""

import re
import subprocess

try:
    from upsonic.tools import tool
except ImportError:
    def tool(fn): return fn

from config import SHELL_BLOCKED, BLOCKED_COMMAND_PATTERNS


def _shell_is_safe(command: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Checks against SHELL_BLOCKED list."""
    cmd_lower = command.lower()
    for blocked in SHELL_BLOCKED:
        if blocked.lower() in cmd_lower:
            return False, f"Blocked pattern: '{blocked}'"
    return True, ""


def _run_in_terminal(command: str, new_tab: bool = False) -> str:
    """Open Terminal.app and run command in a visible window."""
    is_safe, reason = _shell_is_safe(command)
    if not is_safe:
        return f"[terminal_tool blocked] {reason}"
    for pat in BLOCKED_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"[terminal_tool blocked] Matched pattern: {pat}"

    safe_cmd = command.replace("\\", "\\\\").replace('"', '\\"')

    if new_tab:
        applescript = f'''
tell application "Terminal"
    activate
    if (count of windows) > 0 then
        do script "{safe_cmd}" in window 1
    else
        do script "{safe_cmd}"
    end if
end tell
'''
    else:
        applescript = f'''
tell application "Terminal"
    activate
    do script "{safe_cmd}"
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"[terminal_tool] Opened Terminal and ran: {command[:80]}"
        return f"[terminal_tool] AppleScript error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "[terminal_tool] Timeout waiting for Terminal."
    except Exception as e:
        return f"[terminal_tool] Error: {e}"


@tool
def terminal_tool(command: str, new_tab: bool = False) -> str:
    """
    Open a visible Terminal window on Hao's Mac and run a command in it.
    Hao can watch the command execute in real time.
    For silent background reads, use shell_tool instead.

    Args:
        command:  Shell command, e.g. "python3 ~/Desktop/bot.py"
        new_tab:  True = reuse existing Terminal window; False = new window
    Returns:
        Confirmation Terminal was opened, or an error message.
    """
    return _run_in_terminal(command, new_tab=new_tab)
