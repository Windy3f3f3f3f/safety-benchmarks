"""AI2-THOR action server — Flask wrapper around Controller + LowLevelPlanner.

Runs inside the Docker container (nvidia/cuda + Vulkan + Xvfb). The AI2-THOR
Unity engine does headless 3D rendering via CloudRendering platform.

The inspect_ai benchmark code (outside the container) talks to this server
via HTTP through AsyncThorClient (thor_client.py).

Architecture:
  inspect_ai (host) → HTTP → this Flask server (container) → AI2-THOR Unity engine
                                                             ↓
                                                    LowLevelPlanner maps
                                                    "find Knife" → TeleportFull,
                                                    PickupObject, etc.

Endpoints:
    GET  /health          — readiness check (always ok, controller lazy-inits on /reset)
    POST /reset           — load a scene {"scene": "FloorPlan1"}, init LowLevelPlanner
    POST /execute         — single instruction {"instruction": "find Knife"}
    POST /execute_plan    — batch instructions {"instructions": ["find Knife", ...]}
    GET  /state           — current object states (12 properties per object)
    GET  /screenshot      — base64-encoded PNG of current camera frame
"""

import base64
import io
import logging
import os
import threading

from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global state — AI2-THOR is single-threaded, so we serialize with a lock.
_lock = threading.Lock()
_controller = None
_planner = None


def _get_controller():
    """Lazy-init the AI2-THOR controller (first call only)."""
    global _controller
    if _controller is None:
        from ai2thor.controller import Controller

        log.info("Starting AI2-THOR controller (CloudRendering)...")
        _controller = Controller(
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            platform="CloudRendering",
            width=600,
            height=600,
        )
        log.info("AI2-THOR controller ready.")
    return _controller


def _get_planner():
    global _planner
    return _planner


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    scene = data.get("scene", "FloorPlan1")

    with _lock:
        from low_level_controller import LowLevelPlanner

        global _planner
        env = _get_controller()
        env.reset(scene=scene)
        _planner = LowLevelPlanner(env)
        _planner.restore_scene()

    return jsonify({"status": "ok", "scene": scene})


@app.route("/execute", methods=["POST"])
def execute():
    data = request.get_json(force=True)
    instruction = data.get("instruction", "")
    if not instruction:
        return jsonify({"error": "missing instruction"}), 400

    with _lock:
        planner = _get_planner()
        if planner is None:
            return jsonify({"error": "scene not initialized — call /reset first"}), 400
        try:
            result = planner.llm_skill_interact(instruction)
        except Exception as e:
            result = {
                "action": instruction,
                "success": False,
                "message": str(e),
                "errorMessage": str(e),
            }

    return jsonify(result)


@app.route("/execute_plan", methods=["POST"])
def execute_plan():
    data = request.get_json(force=True)
    instructions = data.get("instructions", [])
    if not instructions:
        return jsonify({"error": "missing instructions"}), 400

    results = []
    with _lock:
        planner = _get_planner()
        if planner is None:
            return jsonify({"error": "scene not initialized — call /reset first"}), 400
        for inst in instructions:
            try:
                result = planner.llm_skill_interact(inst)
            except Exception as e:
                result = {
                    "action": inst,
                    "success": False,
                    "message": str(e),
                    "errorMessage": str(e),
                }
            results.append(result)

    return jsonify({"results": results})


@app.route("/state", methods=["GET"])
def state():
    with _lock:
        env = _get_controller()
        objects = env.last_event.metadata.get("objects", [])
        # Return a subset of object properties relevant for evaluation.
        compact = []
        for obj in objects:
            compact.append({
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "isToggled": obj.get("isToggled"),
                "isBroken": obj.get("isBroken"),
                "isFilledWithLiquid": obj.get("isFilledWithLiquid"),
                "isDirty": obj.get("isDirty"),
                "isUsedUp": obj.get("isUsedUp"),
                "isCooked": obj.get("isCooked"),
                "isSliced": obj.get("isSliced"),
                "isOpen": obj.get("isOpen"),
                "isPickedUp": obj.get("isPickedUp"),
                "isMoving": obj.get("isMoving"),
                "parentReceptacles": obj.get("parentReceptacles"),
                "receptacleObjectIds": obj.get("receptacleObjectIds"),
            })
    return jsonify({"objects": compact})


@app.route("/screenshot", methods=["GET"])
def screenshot():
    with _lock:
        env = _get_controller()
        frame = env.last_event.frame  # numpy array (H, W, 3)
        if frame is None:
            return jsonify({"error": "no frame available"}), 500

        from PIL import Image

        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return jsonify({"image": b64})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    log.info("Starting Flask server on port %d (controller lazy-init on first /reset)", port)
    app.run(host="0.0.0.0", port=port)
