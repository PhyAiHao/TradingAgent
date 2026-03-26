"""
tools/shell_tool.py
───────────────────
Skill 2 — Silent background shell execution.

No extra install needed.
"""

import os
import subprocess

try:
    from upsonic.tools import tool
except ImportError:
    def tool(fn): return fn

from config import WORKSPACE, SHELL_BLOCKED, SHELL_SYSTEM_COMMANDS


def _shell_is_safe(command: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Checks against SHELL_BLOCKED list."""
    cmd_lower = command.lower()
    for blocked in SHELL_BLOCKED:
        if blocked.lower() in cmd_lower:
            return False, f"Blocked pattern: '{blocked}'"
    return True, ""


@tool
def shell_tool(command: str, working_dir: str = "") -> str:
    """
    Run a shell command on Hao's Mac and return the output as text.
    Safe read-oriented commands only. Destructive patterns are blocked.

    Args:
        command:     Shell command (e.g. "ls ~/Desktop", "ps aux | head -20").
        working_dir: Directory to run in. Defaults to agent workspace.
    Returns:
        Command output (stdout + stderr), truncated to 3000 chars if long.
    """
    return run_shell(command, working_dir or WORKSPACE)


def run_shell(command: str, working_dir: str = WORKSPACE) -> str:
    """Plain Python shell execution — can also be called by the heartbeat loop."""
    is_safe, reason = _shell_is_safe(command)
    if not is_safe:
        return f"[Shell blocked] {reason}"

    cwd = working_dir
    for sys_cmd in SHELL_SYSTEM_COMMANDS:
        if command.startswith(sys_cmd):
            cwd = os.path.expanduser("~")
            break

    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=15,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            return "(command ran successfully — no output)"
        return output[:3000] + ("\n... (truncated)" if len(output) > 3000 else "")
    except subprocess.TimeoutExpired:
        return "[Shell timeout] Command exceeded 15 seconds."
    except Exception as e:
        return f"[Shell error] {e}"
