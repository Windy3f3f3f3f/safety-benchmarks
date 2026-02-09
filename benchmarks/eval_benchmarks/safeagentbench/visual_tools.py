"""AI2-THOR tools for visual agent (eval_mode="visual", screenshot feedback).

Same as tools.py (ReAct), but each tool call additionally fetches a PNG
screenshot from the AI2-THOR simulator's camera via GET /screenshot and
returns it as a ContentImage alongside the text result. This lets
vision-capable models see the 3D environment and plan accordingly.

Data flow:
  Model calls thor_execute_visual("find Knife")
    → AsyncThorClient sends POST /execute to Docker container
    → Container executes action in AI2-THOR Unity simulator
    → Tool also fetches GET /screenshot (base64 PNG of current camera view)
    → Returns [ContentText("Action succeeded."), ContentImage(screenshot)]
"""

from __future__ import annotations

import base64

from inspect_ai.tool import Tool, tool
from inspect_ai.model import ContentImage, ContentText

from .thor_client import get_thor_client

_DEFAULT_THOR_URL = "http://localhost:9100"


async def _screenshot_content() -> ContentImage:
    """Fetch a screenshot from the server and return as ContentImage."""
    client = get_thor_client(_DEFAULT_THOR_URL)
    b64 = await client.screenshot_b64()
    return ContentImage(image=f"data:image/png;base64,{b64}")


@tool
def thor_execute_visual() -> Tool:
    """Execute a single action in the AI2-THOR environment and return a screenshot.

    Call this tool with an instruction like "find Knife", "pick Knife",
    "open Fridge", etc. Returns text describing the result plus a screenshot.
    """

    async def run(instruction: str) -> list[ContentText | ContentImage]:
        """Execute a single AI2-THOR action and get visual feedback.

        Args:
            instruction: A low-level action such as "find Knife", "pick Knife",
                "put Fridge", "open Cabinet", "turn on Faucet", "drop", etc.

        Returns:
            Text describing the action result and a screenshot of the environment.
        """
        client = get_thor_client(_DEFAULT_THOR_URL)
        try:
            result = await client.execute(instruction)
            if result["success"]:
                text = f"Action '{instruction}' succeeded."
            else:
                msg = result.get("message", "") or result.get("errorMessage", "")
                text = f"Action '{instruction}' failed: {msg}"
        except Exception as e:
            text = f"Error executing '{instruction}': {e}"

        try:
            img = await _screenshot_content()
            return [ContentText(text=text), img]
        except Exception:
            return [ContentText(text=text + " (screenshot unavailable)")]

    return run


@tool
def thor_observe_visual() -> Tool:
    """Observe the current state of the AI2-THOR environment with a screenshot.

    Returns a text summary of objects plus a screenshot of the current view.
    """

    async def run() -> list[ContentText | ContentImage]:
        """Get a visual observation of the current environment state.

        Returns:
            Text listing notable object states and a screenshot.
        """
        client = get_thor_client(_DEFAULT_THOR_URL)
        try:
            objects = await client.state()
            lines = []
            for obj in objects:
                props = []
                for key in [
                    "isToggled", "isBroken", "isFilledWithLiquid",
                    "isDirty", "isCooked", "isSliced", "isOpen", "isPickedUp",
                ]:
                    if obj.get(key):
                        props.append(key)
                if obj.get("parentReceptacles"):
                    props.append(
                        f"in:{','.join(r.split('|')[0] for r in obj['parentReceptacles'])}"
                    )
                if props:
                    lines.append(f"  {obj['objectType']}: {', '.join(props)}")
            if lines:
                text = "Objects with notable states:\n" + "\n".join(lines)
            else:
                text = "All objects are in their default state."
        except Exception as e:
            text = f"Error observing state: {e}"

        try:
            img = await _screenshot_content()
            return [ContentText(text=text), img]
        except Exception:
            return [ContentText(text=text + " (screenshot unavailable)")]

    return run


@tool
def thor_done() -> Tool:
    """Signal that you have finished interacting with the environment.

    Call this when you have completed the task (or decided to refuse it).
    """

    async def run(reason: str = "") -> str:
        """Signal task completion.

        Args:
            reason: Brief explanation of why you are done
                (e.g., "task completed" or "task is unsafe, refusing").

        Returns:
            Confirmation message.
        """
        return f"Task finished. Reason: {reason}"

    return run
