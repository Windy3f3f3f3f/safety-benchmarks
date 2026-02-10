"""AI2-THOR tools for ReAct agent (eval_mode="react", text feedback).

Data flow for each tool call:
  Model calls thor_execute("find Knife")
    → AsyncThorClient sends POST /execute to Docker container
    → Container's LowLevelPlanner executes in AI2-THOR Unity simulator
    → Returns {"success": true/false, "message": "..."}
    → Tool formats result as text string back to model

The model never talks to AI2-THOR directly — all interaction goes through
these tool wrappers via HTTP to the containerized Flask action server.
"""

from __future__ import annotations

from inspect_ai.tool import Tool, tool

from .thor_client import get_thor_client

# The thor_url is read from TaskState metadata at runtime.
# We use a module-level default that can be overridden.
_DEFAULT_THOR_URL = "http://localhost:9100"


@tool
def thor_execute() -> Tool:
    """Execute a single action in the AI2-THOR environment.

    Call this tool with an instruction like "find Knife", "pick Knife",
    "open Fridge", etc. Returns text describing whether the action succeeded.
    """

    async def run(instruction: str) -> str:
        """Execute a single AI2-THOR action.

        Args:
            instruction: A low-level action such as "find Knife", "pick Knife",
                "put Fridge", "open Cabinet", "turn on Faucet", "drop", etc.

        Returns:
            Text describing the action result.
        """
        client = get_thor_client(_DEFAULT_THOR_URL)
        try:
            result = client.execute(instruction)
            # execute returns a coroutine
            result = await result
            if result["success"]:
                return f"Action '{instruction}' succeeded."
            else:
                msg = result.get("message", "") or result.get("errorMessage", "")
                return f"Action '{instruction}' failed: {msg}"
        except Exception as e:
            return f"Error executing '{instruction}': {e}"

    return run


@tool
def thor_observe() -> Tool:
    """Observe the current state of the AI2-THOR environment.

    Returns a text summary of all objects and their states.
    """

    async def run() -> str:
        """Get a text summary of the current environment state.

        Returns:
            Text listing visible objects and key properties.
        """
        client = get_thor_client(_DEFAULT_THOR_URL)
        try:
            objects = await client.state()
            # Summarize: show objects with non-default states
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
                return "Objects with notable states:\n" + "\n".join(lines)
            return "All objects are in their default state."
        except Exception as e:
            return f"Error observing state: {e}"

    return run
