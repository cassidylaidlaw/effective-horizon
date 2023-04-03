import pickle
from typing import Any, Iterable, List, Optional, Set, TypedDict, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Box, Door, Goal, Lava, WorldObj
from minigrid.envs.blockedunlockpickup import BlockedUnlockPickupEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.obstructedmaze import ObstructedMazeEnv
from minigrid.envs.unlockpickup import UnlockPickupEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper


class DeterministicWrapper(gym.core.Wrapper):
    """
    Set the environment seed consistently each episode.
    """

    def __init__(self, env, seed=0):
        super().__init__(env)
        self._constant_seed = seed

    def reset(self, *args, **kwargs):
        return self.env.reset(seed=self._constant_seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            reward = 1

        return obs, reward, terminated, truncated, info


class NavigationOnlyWrapper(gym.core.Wrapper):
    """
    Wrapper which removes all non-navigation actions.
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Discrete(3)


class NoDoneActionWrapper(gym.core.Wrapper):
    """
    Wrapper which removes the "done" action.
    """

    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Discrete(6)


env_types_with_obj = (
    KeyCorridorEnv,
    UnlockPickupEnv,
    BlockedUnlockPickupEnv,
    ObstructedMazeEnv,
)


def get_distance_to_nearest_goal(env: MiniGridEnv) -> Optional[float]:
    agent_x, agent_y = env.agent_pos
    goal_distances: List[float] = []
    for grid_x in range(env.grid.width):
        for grid_y in range(env.grid.height):
            tile = env.grid.get(grid_x, grid_y)
            if (tile is not None and tile.type == "goal") or (
                isinstance(env, env_types_with_obj) and tile == env.obj
            ):
                # Calculate the number of steps it would take to get to the goal,
                # assuming no barriers in between.
                dir_x, dir_y = env.dir_vec
                delta_x, delta_y = grid_x - agent_x, grid_y - agent_y
                dot_product = np.sign(delta_x) * dir_x + np.sign(delta_y) * dir_y

                # Case 1: at the goal.
                if delta_x == 0 and delta_y == 0:
                    turns_necessary = 0
                # Case 2: in line with the goal.
                elif delta_x == 0 or delta_y == 0:
                    turns_necessary = 1 - dot_product
                # Case 3: not in line with the goal.
                else:
                    turns_necessary = 1
                    if dot_product < 0:
                        turns_necessary += 1

                goal_distances.append(turns_necessary + abs(delta_x) + abs(delta_y))
    if isinstance(env, env_types_with_obj) and env.carrying == env.obj:
        goal_distances.append(0)
    if goal_distances:
        return min(goal_distances)
    else:
        return None


class GoalDistanceShapedRewardWrapper(gym.core.Wrapper):
    """
    Gives a shaped reward based on the distance towards the goal that was reduced.
    """

    def step(self, action):
        before_distance = get_distance_to_nearest_goal(self.unwrapped)
        obs, reward, terminated, truncated, info = self.env.step(action)
        after_distance = get_distance_to_nearest_goal(self.unwrapped)

        assert before_distance is not None and after_distance is not None
        reward += 0.1 * (before_distance - after_distance)

        return obs, reward, terminated, truncated, info


class PickupShapedRewardWrapper(gym.core.Wrapper):
    """
    Gives a shaped reward of +0.1 for picking up an object and -0.1 for dropping it.
    """

    def step(self, action):
        unwrapped_env: MiniGridEnv = self.unwrapped
        prev_hash = unwrapped_env.hash()
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_hash = unwrapped_env.hash()

        if new_hash != prev_hash:
            if action == MiniGridEnv.Actions.pickup:
                reward += 0.1
            elif action == MiniGridEnv.Actions.drop:
                reward -= 0.1

        return obs, reward, terminated, truncated, info


class MinigridShapedRewardConfig(TypedDict):
    distance: bool
    open_doors: bool
    picked_up_objects: bool
    lava: bool


DEFAULT_SHAPED_REWARD_CONFIG: MinigridShapedRewardConfig = {
    "distance": False,
    "open_doors": False,
    "picked_up_objects": False,
    "lava": False,
}


class WorldObjWithHasBeenPickedUp(WorldObj):
    has_been_picked_up: bool


class MinigridShapedRewardWrapper(gym.core.Wrapper):
    def __init__(self, env, config: MinigridShapedRewardConfig):
        super().__init__(env)
        self.config = config

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        for obj in self._iterate_objects():
            if obj.can_pickup():
                obj.has_been_picked_up = False

        return obs, info

    def _iterate_objects(self) -> Iterable[WorldObj]:
        env: MiniGridEnv = self.unwrapped
        grid: Grid = env.grid
        if env.carrying is not None:
            yield env.carrying
        for x in range(env.width):
            for y in range(env.height):
                obj = cast(Optional[WorldObj], grid.get(x, y))
                while obj is not None:
                    yield obj
                    if isinstance(obj, Box):
                        obj = obj.contains
                    else:
                        obj = None

    def _get_num_open_doors(self):
        env: MiniGridEnv = self.unwrapped
        grid: Grid = env.grid
        num_open = 0
        for x in range(env.width):
            for y in range(env.height):
                obj = cast(Optional[WorldObj], grid.get(x, y))
                if isinstance(obj, Door):
                    if obj.is_open:
                        num_open += 1
        return num_open

    def _get_num_picked_up_objects(self):
        num_picked_up = 0
        for obj in self._iterate_objects():
            if obj.can_pickup() and obj.has_been_picked_up:
                num_picked_up += 1
        return num_picked_up

    def _is_lava_in_front(self) -> bool:
        env: MiniGridEnv = self.unwrapped
        grid: Grid = env.grid
        fwd_pos = env.front_pos
        fwd_cell = grid.get(*fwd_pos)
        return fwd_cell is not None and fwd_cell.type == "lava"

    def _did_die_from_lava(self, action, terminated) -> bool:
        env: MiniGridEnv = self.unwrapped
        return action == env.actions.forward and terminated and self._is_lava_in_front()

    def step(self, action):
        distance_before = get_distance_to_nearest_goal(self.unwrapped)
        open_doors_before = self._get_num_open_doors()
        picked_up_before = self._get_num_picked_up_objects()

        obs, reward, terminated, truncated, info = self.env.step(action)

        carrying = cast(
            Optional[WorldObjWithHasBeenPickedUp],
            cast(MiniGridEnv, self.unwrapped).carrying,
        )
        if carrying is not None:
            carrying.has_been_picked_up = True

        distance_after = get_distance_to_nearest_goal(self.unwrapped)
        open_doors_after = self._get_num_open_doors()
        picked_up_after = self._get_num_picked_up_objects()

        shaping_reward: float = 0
        if self.config["distance"]:
            if distance_before is not None and distance_after is not None:
                # We round distances to make distance shaping reward come in increments
                # of 0.001. This helps avoid floating-point error issues.
                shaping_reward += 0.1 * (
                    round(distance_before, 2) - round(distance_after, 2)
                )

                if not (-2 <= distance_before - distance_after <= 1):
                    breakpoint()
            else:
                assert self.unwrapped.obj.type == "box"  # type: ignore
        if self.config["open_doors"]:
            shaping_reward += 0.2 * (open_doors_after - open_doors_before)
        if self.config["picked_up_objects"]:
            shaping_reward += 0.2 * (picked_up_after - picked_up_before)
        if self.config["lava"] and self._did_die_from_lava(action, terminated):
            shaping_reward -= 1
        reward += shaping_reward

        return obs, reward, terminated, truncated, info

    @staticmethod
    def get_meaningful_shaping_functions(env: MiniGridEnv) -> List[str]:
        """
        Return a list of keys of MinigridShapedRewardConfig indicating which
        shaping functions would actually affect this environment. For instance, the lava
        shaping function is only meaningful if there is lava in the environment.
        """

        meaningful_shaping_functions: Set[str] = set()
        grid: Grid = env.grid

        for x in range(env.width):
            for y in range(env.height):
                obj = cast(Optional[WorldObj], grid.get(x, y))
                if isinstance(obj, Goal):
                    meaningful_shaping_functions.add("distance")
                if isinstance(obj, Door):
                    meaningful_shaping_functions.add("open_doors")
                if obj is not None and obj.can_pickup():
                    # Pickup shaped reward is useless for goal objects.
                    if not (isinstance(env, env_types_with_obj) and env.obj == obj):
                        meaningful_shaping_functions.add("picked_up_objects")
                if isinstance(obj, Lava):
                    meaningful_shaping_functions.add("lava")

        if isinstance(env, env_types_with_obj):
            # These envs don't have a goal position but they do have a goal
            # object that the shaping function works with.
            meaningful_shaping_functions.add("distance")

        return list(meaningful_shaping_functions)


class TimeInvariantWrapper(gym.core.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cast(MiniGridEnv, self.unwrapped).step_count = 0
        return obs, reward, terminated, truncated, info


class StateWrapper(gym.core.Wrapper):
    state_attributes = [
        "agent_pos",
        "agent_dir",
        "obj",
        "obj_type",
    ]

    def get_state(self) -> bytes:
        unwrapped_env: MiniGridEnv = self.unwrapped
        grid: Grid = unwrapped_env.grid
        state = {
            "random_state": unwrapped_env._np_random.__getstate__(),
            "grid": grid.encode(),
            "carrying": unwrapped_env.carrying.encode()
            if unwrapped_env.carrying
            else None,
        }
        for attribute in self.state_attributes:
            if hasattr(unwrapped_env, attribute):
                state[attribute] = getattr(unwrapped_env, attribute)

        # Sometimes the environment has references to specific objects in the world,
        # so grab those and put them in world_obj_refs.
        world_obj_refs = {}
        for attribute, value in unwrapped_env.__dict__.items():
            if isinstance(value, WorldObj):
                for i in range(unwrapped_env.width):
                    for j in range(unwrapped_env.height):
                        if grid.get(i, j) == value:
                            world_obj_refs[attribute] = (i, j)
        state["world_obj_refs"] = world_obj_refs

        # Objects in boxes don't get serialized by Grid.encode(). Also, encode
        # has_been_picked_up for MinigridShapedRewardWrapper.
        box_contents = {}
        has_been_picked_up = {}
        for i in range(unwrapped_env.width):
            for j in range(unwrapped_env.height):
                world_obj: Optional[WorldObj] = grid.get(i, j)
                if isinstance(world_obj, Box):
                    box = world_obj
                    if box.contains is not None:
                        contents = cast(WorldObj, box.contains)
                        box_contents[i, j] = contents.encode()
                        if hasattr(contents, "has_been_picked_up"):
                            assert not contents.has_been_picked_up
                if hasattr(world_obj, "has_been_picked_up"):
                    has_been_picked_up[i, j] = cast(
                        WorldObjWithHasBeenPickedUp, world_obj
                    ).has_been_picked_up
        state["box_contents"] = box_contents
        state["has_been_picked_up"] = has_been_picked_up

        return pickle.dumps(state)

    def set_state(self, state_bytes: bytes):
        unwrapped_env: MiniGridEnv = self.unwrapped
        state = pickle.loads(state_bytes)
        unwrapped_env._np_random.__setstate__(state["random_state"])
        unwrapped_env.grid, _ = Grid.decode(state["grid"])
        unwrapped_env.carrying = (
            WorldObj.decode(*state["carrying"]) if state["carrying"] else None
        )
        if unwrapped_env.carrying is not None:
            unwrapped_env.carrying.has_been_picked_up = True
            if isinstance(unwrapped_env.carrying, Box):
                box = unwrapped_env.carrying
                if box.contains is not None:
                    unwrapped_env.carrying.contains.has_been_picked_up = False

        for attribute in self.state_attributes:
            if attribute in state:
                setattr(unwrapped_env, attribute, state[attribute])
        for attribute, (i, j) in state["world_obj_refs"].items():
            setattr(unwrapped_env, attribute, unwrapped_env.grid.get(i, j))
        for (i, j), contains in state["box_contents"].items():
            box = unwrapped_env.grid.get(i, j)
            assert isinstance(box, Box)
            box.contains = WorldObj.decode(*contains)
            if box.contains.can_pickup():
                box.contains.has_been_picked_up = False
        for (i, j), has_been_picked_up in state["has_been_picked_up"].items():
            obj: WorldObjWithHasBeenPickedUp = unwrapped_env.grid.get(i, j)
            obj.has_been_picked_up = has_been_picked_up


def wrap_minigrid_env(
    env,
    navigation_only=False,
    shaping_config: MinigridShapedRewardConfig = DEFAULT_SHAPED_REWARD_CONFIG,
    seed=0,
):
    env = TimeInvariantWrapper(env)
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    env = DeterministicWrapper(env, seed=seed)
    if navigation_only:
        env = NavigationOnlyWrapper(env)
    else:
        env = NoDoneActionWrapper(env)
    env = MinigridShapedRewardWrapper(env, shaping_config)
    env = StateWrapper(env)
    return env


def build_env_maker(
    env_id: str,
    shaping_config: MinigridShapedRewardConfig = DEFAULT_SHAPED_REWARD_CONFIG,
):
    def env_maker(config={}):
        env = gym.make(env_id)
        env = wrap_minigrid_env(
            env,
            navigation_only="Empty" in env_id,
            shaping_config=shaping_config,
        )
        cast(Any, env).horizon = 100
        return env

    return env_maker
