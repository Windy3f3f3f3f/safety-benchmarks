"""Low-level planner for AI2-THOR — maps high-level actions to simulator commands.

Copied from SafeAgentBench/low_level_controller/low_level_controller.py with
minimal adaptation: replaced scipy.spatial.KDTree with a pure-numpy fallback
to avoid the scipy dependency inside the Docker container.
"""

import logging
import math
import re
import string

import numpy as np

log = logging.getLogger(__name__)


class _SimpleKDTree:
    """Minimal nearest-neighbour lookup using brute-force (no scipy)."""

    def __init__(self, points: np.ndarray):
        self.points = points

    def query(self, point, k=1):
        dists = np.linalg.norm(self.points - np.array(point), axis=1)
        idx = np.argsort(dists)[:k]
        return dists[idx], idx


class LowLevelPlanner:
    def __init__(self, env):
        self.actions = [
            "find", "pick", "put", "open", "close", "slice",
            "turn on", "turn off", "drop", "throw", "break",
            "cook", "dirty", "clean", "fillLiquid", "emptyLiquid", "pour",
        ]
        self.env = env
        self.multi_objs_dict = {}
        self.agent_height = self.env.last_event.metadata["agent"]["position"]["y"]
        self.CAMERA_HEIGHT_OFFSET = 0.675
        self.reachable_positions = None
        self.reachable_position_kdtree = None

    def restore_scene(self):
        self.reachable_positions, self.reachable_position_kdtree = (
            self.get_reachable_positions()
        )
        self.cur_receptacle = None

    def get_obj_idx(self, obj_id):
        for idx, obj in enumerate(self.env.last_event.metadata["objects"]):
            if obj["objectId"] == obj_id:
                return idx

    def get_obj_information(self, obj_id):
        for obj in self.env.last_event.metadata["objects"]:
            if obj["objectId"] == obj_id:
                return obj

    def get_reachable_positions(self):
        free_positions = self.env.step(
            dict(action="GetReachablePositions")
        ).metadata["actionReturn"]
        free_positions = np.array([[p["x"], p["y"], p["z"]] for p in free_positions])
        kd_tree = _SimpleKDTree(free_positions)
        return free_positions, kd_tree

    def natural_word_to_ithor_name(self, w):
        if w == "CD":
            return w
        return "".join([string.capwords(x) for x in w.split()])

    def extract_number_from_string(self, s):
        match = re.match(r"^(.*\D)\s*(\d+)?$", s)
        if match:
            text_part = match.group(1).strip()
            number_part = int(match.group(2)) if match.group(2) else None
            return text_part, number_part
        return s, None

    def split_string_for_fill(self, s):
        parts = s.split()
        part1 = " ".join(parts[:-1])
        part2 = parts[-1]
        return part1, part2

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def llm_skill_interact(self, instruction: str):
        if instruction.startswith("find "):
            obj_name = (
                instruction.replace("find a ", "")
                .replace("find an ", "")
                .replace("find the ", "")
                .replace("find ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.find(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("pick "):
            obj_name = (
                instruction.replace("pick up ", "")
                .replace("pick ", "")
                .replace("a ", "")
                .replace("an ", "")
                .replace("the ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.pick(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("put "):
            obj_name = (
                instruction.replace("put on ", "")
                .replace("put down ", "")
                .replace("put ", "")
                .replace("the ", "")
                .replace("a ", "")
                .replace("an ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.put(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("open "):
            obj_name = (
                instruction.replace("open the ", "")
                .replace("open a ", "")
                .replace("open an ", "")
                .replace("open ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.open(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("close "):
            obj_name = (
                instruction.replace("close the ", "")
                .replace("close a ", "")
                .replace("close an ", "")
                .replace("close ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.close(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("slice "):
            obj_name = (
                instruction.replace("slice the ", "")
                .replace("slice a ", "")
                .replace("slice an ", "")
                .replace("slice ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.slice(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("turn on ") or instruction.startswith("toggle on "):
            obj_name = (
                instruction.replace("turn on the ", "")
                .replace("turn on a ", "")
                .replace("turn on an ", "")
                .replace("turn on ", "")
                .replace("toggle on the ", "")
                .replace("toggle on a ", "")
                .replace("toggle on an ", "")
                .replace("toggle on ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.turn_on(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("turn off ") or instruction.startswith("toggle off "):
            obj_name = (
                instruction.replace("turn off the ", "")
                .replace("turn off a ", "")
                .replace("turn off an ", "")
                .replace("turn off ", "")
                .replace("toggle off the ", "")
                .replace("toggle off a ", "")
                .replace("toggle off an ", "")
                .replace("toggle off ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.turn_off(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("drop"):
            ret = self.drop()
        elif instruction.startswith("throw"):
            ret = self.throw()
        elif instruction.startswith("break "):
            obj_name = (
                instruction.replace("break the ", "")
                .replace("break a ", "")
                .replace("break an ", "")
                .replace("break ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.break_(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("cook "):
            obj_name = (
                instruction.replace("cook the ", "")
                .replace("cook a ", "")
                .replace("cook an ", "")
                .replace("cook ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.cook(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("dirty "):
            obj_name = (
                instruction.replace("dirty the ", "")
                .replace("dirty a ", "")
                .replace("dirty an ", "")
                .replace("dirty ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.dirty(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("clean "):
            obj_name = (
                instruction.replace("clean the ", "")
                .replace("clean a ", "")
                .replace("clean an ", "")
                .replace("clean ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.clean(self.natural_word_to_ithor_name(obj_name), obj_num)
        elif instruction.startswith("fillLiquid ") or instruction.startswith("fill "):
            obj_name = (
                instruction.replace("fillLiquid", "fill")
                .replace("fill the ", "")
                .replace("fill a ", "")
                .replace("fill an ", "")
                .replace("fill ", "")
            )
            obj_name, liquid_name = self.split_string_for_fill(obj_name)
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.fillLiquid(
                self.natural_word_to_ithor_name(obj_name), obj_num, liquid_name
            )
        elif instruction.startswith("emptyLiquid ") or instruction.startswith("empty "):
            obj_name = (
                instruction.replace("emptyLiquid", "empty")
                .replace("empty the ", "")
                .replace("empty a ", "")
                .replace("empty an ", "")
                .replace("empty ", "")
            )
            obj_name, obj_num = self.extract_number_from_string(obj_name)
            ret = self.emptyLiquid(
                self.natural_word_to_ithor_name(obj_name), obj_num
            )
        elif instruction.startswith("pour"):
            ret = self.pour()
        else:
            assert False, f"instruction not supported: {instruction}"

        if (
            self.env.last_event.metadata["lastActionSuccess"]
            and "Nothing Done. " not in ret
        ):
            log.info("Last action succeeded")

        error_message = (
            ret
            if "Nothing Done. " in ret
            else self.env.last_event.metadata["errorMessage"]
        )

        return {
            "action": instruction,
            "success": len(ret) <= 0,
            "message": ret,
            "errorMessage": error_message,
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def get_object_prop(self, obj_id, prop_name, metadata):
        for obj in metadata["objects"]:
            if obj["objectId"] == obj_id:
                return obj[prop_name]
        return None

    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def get_obj_id_from_name(
        self,
        obj_name,
        obj_num=None,
        parent_receptacle_penalty=True,
        priority_in_visibility=False,
        exclude_obj_id=None,
        get_inherited=False,
    ):
        obj_id = None
        obj_data = None
        min_distance = 1e8

        if obj_num is not None:
            if obj_num < 1:
                log.warning("obj_num should be greater than 0")
                return None, None
            if obj_name in self.multi_objs_dict:
                for tmp_id in self.multi_objs_dict[obj_name]:
                    tmp_num = self.multi_objs_dict[obj_name][tmp_id]
                    if tmp_num == obj_num:
                        obj_id = tmp_id
                        break
                if obj_id is not None:
                    for obj in self.env.last_event.metadata["objects"]:
                        if obj["objectId"] == obj_id:
                            obj_data = obj
                            break
                    return obj_id, obj_data

        for obj in self.env.last_event.metadata["objects"]:
            if obj["objectId"] == exclude_obj_id:
                continue
            if (
                obj_name in self.multi_objs_dict
                and obj["objectId"] in self.multi_objs_dict[obj_name]
            ):
                continue
            if obj["objectId"].split("|")[0].casefold() == obj_name.casefold() and (
                get_inherited is False or len(obj["objectId"].split("|")) == 5
            ):
                flag = False
                if obj["distance"] < min_distance:
                    penalty_advantage = 0
                    if parent_receptacle_penalty and obj["parentReceptacles"]:
                        for p in obj["parentReceptacles"]:
                            is_open = self.get_object_prop(
                                p, "isOpen", self.env.last_event.metadata
                            )
                            openable = self.get_object_prop(
                                p, "openable", self.env.last_event.metadata
                            )
                            if openable is True and is_open is False:
                                flag = True
                                break
                    if flag:
                        continue
                    if obj_name.casefold() in ("stoveburner", "toaster"):
                        if len(obj["receptacleObjectIds"]) > 0:
                            penalty_advantage += 10000
                    if priority_in_visibility and obj["visible"] is False:
                        penalty_advantage += 1000
                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        if obj_id is not None and obj_num is not None:
            if obj_name not in self.multi_objs_dict:
                self.multi_objs_dict[obj_name] = {}
            self.multi_objs_dict[obj_name][obj_id] = obj_num

        return obj_id, obj_data

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def find(self, target_obj, obj_num):
        objects = self.env.last_event.metadata["objects"]
        ret_msg = ""
        log.info(f"object navigation ({target_obj})")

        obj_id, obj_data = self.get_obj_id_from_name(target_obj, obj_num=obj_num)

        obj_idx = -1
        for i, o in enumerate(objects):
            if o["objectId"] == obj_id:
                obj_idx = i
                break

        if obj_idx == -1:
            ret_msg = f"Cannot find {target_obj}"
        else:
            max_attempts = 20
            teleport_success = False
            loc = objects[obj_idx]["position"]
            obj_rot = objects[obj_idx]["rotation"]["y"]

            if objects[obj_idx]["visible"] and objects[obj_idx]["distance"] < 1.0:
                max_attempts = 0
                teleport_success = True

            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and target_obj in ("Fridge", "Microwave"):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position(
                    [loc["x"], loc["y"], loc["z"]], reachable_pos_idx
                )

                rot_angle = math.atan2(
                    -(loc["x"] - closest_loc[0]), loc["z"] - closest_loc[2]
                )
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle

                if i < 10 and target_obj in ("Fridge", "Microwave"):
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj == "Fridge" and not (
                        (70 < angle_diff < 110) or (250 < angle_diff < 290)
                    ):
                        continue
                    if target_obj == "Microwave" and not (
                        (160 < angle_diff < 200) or (-20 < angle_diff < 20)
                    ):
                        continue

                camera_height = self.agent_height + self.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(
                    loc["x"] - closest_loc[0], loc["z"] - closest_loc[2]
                )
                hor_angle = math.atan2((loc["y"] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle
                hor_angle *= 0.9

                self.env.step(
                    dict(
                        action="TeleportFull",
                        x=closest_loc[0],
                        y=self.agent_height,
                        z=closest_loc[2],
                        rotation=rot_angle,
                        horizon=-hor_angle,
                        standing=True,
                    )
                )

                if not self.env.last_event.metadata["lastActionSuccess"]:
                    if i == max_attempts - 1:
                        log.warning(
                            f"TeleportFull failed: {self.env.last_event.metadata['errorMessage']}"
                        )
                        break
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f"Cannot move to {target_obj}"

        return ret_msg

    def fillLiquid(self, obj_name, obj_num, liquid_name):
        log.info(f"fillLiquid {obj_name} with {liquid_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f"Cannot find {obj_name} to fill"
        else:
            self.env.step(
                action="FillObjectWithLiquid",
                objectId=obj_id,
                fillLiquid=liquid_name,
            )
            if not self.env.last_event.metadata["lastActionSuccess"]:
                ret_msg = "FillLiquid action failed"
        return ret_msg

    def emptyLiquid(self, obj_name, obj_num):
        log.info(f"emptyLiquid {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f"Cannot find {obj_name} to empty"
        else:
            self.env.step(action="EmptyLiquidFromObject", objectId=obj_id)
            if not self.env.last_event.metadata["lastActionSuccess"]:
                ret_msg = "EmptyLiquid action failed"
        return ret_msg

    def break_(self, obj_name, obj_num):
        log.info(f"break {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            ret_msg = f"Cannot find {obj_name} to break"
        else:
            self.env.step(action="BreakObject", objectId=obj_id)
            if not self.env.last_event.metadata["lastActionSuccess"]:
                ret_msg = "Break action failed"
        return ret_msg

    def _camera_adjust_loop(self, action_fn):
        """Run action_fn in a loop with camera adjustments (16 attempts)."""
        ret_msg = ""
        for j in range(16):
            if j == 1:
                self.env.step(dict(action="LookUp", degrees=15))
            elif j == 2:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 3:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 4:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 5:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 6:
                self.env.step(dict(action="LookUp"), degrees=5)
            elif j == 7:
                self.env.step(dict(action="LookDown"), degrees=55)
            elif j == 8:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 9:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 10:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 11:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 12:
                self.env.step(dict(action="LookDown"), degrees=5)
            elif j == 13:
                self.env.step(dict(action="LookUp"), degrees=40)
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="LookUp"))
                self.env.step(dict(action="MoveBack"))
            elif j == 14:
                self.env.step(dict(action="MoveAhead"))
                for _ in range(4):
                    self.env.step(dict(action="MoveRight"))
            elif j == 15:
                for _ in range(8):
                    self.env.step(dict(action="MoveLeft"))

            action_fn()

            if self.env.last_event.metadata["lastActionSuccess"]:
                return ""
            elif j == 15:
                ret_msg = "action failed"
        return ret_msg

    def drop(self):
        if len(self.env.last_event.metadata["inventoryObjects"]) == 0:
            return "Nothing Done. Robot is not holding any object"

        def action_fn():
            self.env.step(action="DropHandObject", forceAction=True)

        ret = self._camera_adjust_loop(action_fn)
        return f"Drop {ret}" if ret else ""

    def throw(self):
        if len(self.env.last_event.metadata["inventoryObjects"]) == 0:
            return "Nothing Done. Robot is not holding any object"

        def action_fn():
            self.env.step(
                action="ThrowObject", moveMagnitude=1500.0, forceAction=True
            )

        ret = self._camera_adjust_loop(action_fn)
        return f"Throw {ret}" if ret else ""

    def pour(self):
        if len(self.env.last_event.metadata["inventoryObjects"]) == 0:
            return "Nothing Done. Robot is not holding any object"

        obj_id = self.env.last_event.metadata["inventoryObjects"][0]["objectId"]
        obj_inf = self.get_obj_information(obj_id)
        if obj_inf is None:
            return "Nothing Done. Cannot find the object"
        is_filled = obj_inf["isFilledWithLiquid"]

        if not is_filled:
            self.env.step(action="RotateHeldObject", pitch=90.0)
            if not self.env.last_event.metadata["lastActionSuccess"]:
                return "Pour action failed"
            return ""

        for angle in [60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            self.env.step(action="RotateHeldObject", pitch=float(angle))
            if not self.env.last_event.metadata["lastActionSuccess"]:
                continue
            if not self.get_obj_information(obj_id)["isFilledWithLiquid"]:
                return ""

        log.warning("Can not pour the liquid")
        return "Pour action failed"

    def pick(self, obj_name, obj_num, manualInteract=False):
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        log.info(f"pick {obj_id}")

        if obj_id is None:
            return f"Nothing Done. Cannot find {obj_name} to pick up"

        if (
            obj_data["visible"] is False
            and obj_data["parentReceptacles"] is not None
            and len(obj_data["parentReceptacles"]) > 0
        ):
            recep_name = obj_data["parentReceptacles"][0].split("|")[0]
            return f"{obj_name} is not visible because it is in {recep_name}"

        def action_fn():
            self.env.step(
                action="PickupObject",
                objectId=obj_id,
                forceAction=False,
                manualInteract=manualInteract,
            )

        ret = self._camera_adjust_loop(action_fn)
        return f"Picking up {obj_name} failed" if ret else ""

    def put(self, receptacle_name, obj_num):
        ret_msg = ""
        if len(self.env.last_event.metadata["inventoryObjects"]) == 0:
            return "Nothing Done. Robot is not holding any object"

        holding_obj_id = self.env.last_event.metadata["inventoryObjects"][0][
            "objectId"
        ]

        halt = False
        last_recep_id = None
        exclude_obj_id = None

        for k in range(2):
            for j in range(17):
                for i in range(2):
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id

                    if i == 0:
                        recep_id, _ = self.get_obj_id_from_name(
                            receptacle_name,
                            exclude_obj_id=exclude_obj_id,
                            obj_num=obj_num,
                        )
                    else:
                        recep_id, _ = self.get_obj_id_from_name(
                            receptacle_name,
                            get_inherited=True,
                            exclude_obj_id=exclude_obj_id,
                            obj_num=obj_num,
                        )

                    if not recep_id:
                        ret_msg = f"Cannot find {receptacle_name} {obj_num}"
                        continue

                    flag = False
                    if j == 1:
                        self.env.step(dict(action="LookUp", degrees=15))
                    elif j == 2:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 3:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 4:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 5:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 6:
                        self.env.step(dict(action="LookUp"), degrees=5)
                    elif j == 7:
                        self.env.step(dict(action="LookDown"), degrees=55)
                    elif j == 8:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 9:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 10:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 11:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 12:
                        self.env.step(dict(action="LookDown"), degrees=5)
                    elif j == 13:
                        self.env.step(dict(action="LookUp"), degrees=40)
                        self.env.step(dict(action="LookUp"))
                        self.env.step(dict(action="LookUp"))
                        self.env.step(dict(action="MoveBack"))
                    elif j == 14:
                        self.env.step(dict(action="MoveAhead"))
                        for _ in range(4):
                            self.env.step(dict(action="MoveRight"))
                    elif j == 15:
                        for _ in range(8):
                            self.env.step(dict(action="MoveLeft"))
                    elif j == 16:
                        for _ in range(4):
                            self.env.step(dict(action="MoveRight"))
                        self.env.step(dict(action="RotateRight", degrees=15))
                    elif j == 17:
                        event = self.env.step(
                            action="GetSpawnCoordinatesAboveReceptacle",
                            objectId=recep_id,
                            anywhere=False,
                        )
                        position_above = event.metadata["actionReturn"]
                        self.env.step(
                            action="PlaceObjectAtPoint",
                            objectId=holding_obj_id,
                            position={
                                "x": sum(t["x"] for t in position_above)
                                / len(position_above),
                                "y": sum(t["y"] for t in position_above)
                                / len(position_above),
                                "z": sum(t["z"] for t in position_above)
                                / len(position_above),
                            },
                        )
                        obj_info = self.get_obj_information(holding_obj_id)
                        flag = True

                    last_recep_id = recep_id
                    if not flag:
                        self.env.step(
                            dict(
                                action="PutObject",
                                objectId=recep_id,
                                forceAction=True,
                            )
                        )
                        if self.env.last_event.metadata["lastActionSuccess"]:
                            halt = True
                            ret_msg = ""
                            break
                        elif j == 16:
                            ret_msg = (
                                f"Putting the object on {receptacle_name} failed"
                            )
                    else:
                        if recep_id in obj_info["parentReceptacles"]:
                            halt = True
                            ret_msg = ""
                            break
                        else:
                            ret_msg = (
                                f"Putting the object on {receptacle_name} failed"
                            )

                if halt:
                    break
            if halt:
                break

        return ret_msg

    def slice(self, obj_name, obj_num):
        log.info(f"slice {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to slice"
        self.env.step(action="SliceObject", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Slice action failed"
        return ret_msg

    def cook(self, obj_name, obj_num):
        log.info(f"cook {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to cook"
        self.env.step(action="CookObject", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Cook action failed"
        return ret_msg

    def dirty(self, obj_name, obj_num):
        log.info(f"dirty {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to dirty"
        self.env.step(action="DirtyObject", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Dirty action failed"
        return ret_msg

    def clean(self, obj_name, obj_num):
        log.info(f"clean {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to clean"
        self.env.step(action="CleanObject", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Clean action failed"
        return ret_msg

    def turn_on(self, obj_name, obj_num):
        log.info(f"turn on {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to turn on"
        self.env.step(action="ToggleObjectOn", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Turn on action failed"
        return ret_msg

    def turn_off(self, obj_name, obj_num):
        log.info(f"turn off {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to turn off"
        self.env.step(action="ToggleObjectOff", objectId=obj_id)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Turn off action failed"
        return ret_msg

    def close(self, obj_name, obj_num):
        log.info(f"close {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to close"
        self.env.step(action="CloseObject", objectId=obj_id, forceAction=True)
        if not self.env.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Close action failed"
        return ret_msg

    def open(self, obj_name, obj_num):
        log.info(f"open {obj_name}")
        ret_msg = ""
        obj_id, _ = self.get_obj_id_from_name(obj_name, obj_num=obj_num)
        if obj_id is None:
            return f"Cannot find {obj_name} to open"
        for i in range(4):
            self.env.step(action="OpenObject", objectId=obj_id, openness=1.0)
            if self.env.last_event.metadata["lastActionSuccess"]:
                return ""
            ret_msg = "Open action failed"
            if i == 0:
                self.env.step(action="MoveBack")
            elif i == 1:
                self.env.step(action="MoveBack")
                self.env.step(action="MoveRight")
            elif i == 2:
                self.env.step(action="MoveLeft")
                self.env.step(action="MoveLeft")
        return ret_msg
