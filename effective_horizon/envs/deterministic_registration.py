"""
This module creates gym environments corresponding to the deterministic MDPs
with all states enumerated that are stored as numpy arrays.
"""

import copy
from typing import Dict, Iterable, List, Literal, TypeVar

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec, register, registry
from procgen.env import EXPLORATION_LEVEL_SEEDS

from .atari import AtariEnv, AtariEnvConfig, all_games
from .minigrid import DEFAULT_SHAPED_REWARD_CONFIG, MinigridShapedRewardWrapper
from .minigrid import build_env_maker as build_minigrid_env_maker
from .procgen import DeterministicProcgenEnv, DeterministicProcgenEnvConfig
from .utils import register_rllib_env_if_installed
from .wrappers import StickyActionsWrapper

GYM_NAMESPACE = "BRIDGE"


def register_atari_envs():
    atari_reward_factors: Dict[str, float] = {
        "alien": 10.0,
        "amidar": 10.0,
        "assault": 21.0,
        "asterix": 50.0,
        "asteroids": 10.0,
        "atlantis": 100.0,
        "bank_heist": 10.0,
        "battle_zone": 1000.0,
        "beam_rider": 44.0,
        "bowling": 1.0,
        "breakout": 1.0,
        "centipede": 100.0,
        "chopper_command": 100.0,
        "crazy_climber": 100.0,
        "demon_attack": 10.0,
        "enduro": 1.0,
        "fishing_derby": 1.0,
        "freeway": 1.0,
        "frostbite": 10.0,
        "gopher": 20.0,
        "hero": 25.0,
        "ice_hockey": 1.0,
        "kangaroo": 100.0,
        "krull": 1.0,
        "kung_fu_master": 1.0,
        "montezuma_revenge": 100.0,
        "ms_pacman": 10.0,
        "name_this_game": 10.0,
        "phoenix": 20.0,
        "pong": 1.0,
        "private_eye": 100.0,
        "qbert": 25.0,
        "road_runner": 100.0,
        "seaquest": 20.0,
        "skiing": 100.0,
        "space_invaders": 5.0,
        "star_gunner": 1.0,
        "tennis": 1.0,
        "time_pilot": 100.0,
        "tutankham": 1.0,
        "venture": 1.0,
        "video_pinball": 100.0,
        "wizard_of_wor": 100.0,
        "zaxxon": 1.0,
    }

    for rom in all_games:
        if rom not in atari_reward_factors:
            continue
        horizons = [10, 20, 30, 40, 50, 70, 100, 200]
        frameskip = 30
        if rom == "montezuma_revenge":
            horizons = [15]
            frameskip = 24
        noops_after_horizon = 0
        if rom == "skiing":
            noops_after_horizon = 200
        for horizon in horizons:
            env_id = f"{rom}_{horizon}_fs{frameskip}"
            config: AtariEnvConfig = {
                "rom_file": rom,
                "horizon": horizon,
                "done_on_reward": False,
                "done_on_life_lost": True,
                "no_done_reward": 0,
                "noops_after_horizon": noops_after_horizon,
                "frameskip": frameskip,
                "repeat_action_probability": 0,
                "deterministic": True,
                "reward_scale": 1 / atari_reward_factors[rom],
                "fire_on_reset": False,
                "clip_rewards": False,
            }
            register(
                id=f"{GYM_NAMESPACE}/{env_id}-v0",
                entry_point=AtariEnv,  # type: ignore
                kwargs={"config": config},
            )
            register_rllib_env_if_installed(
                f"{GYM_NAMESPACE}/{env_id}-v0",
                lambda _: AtariEnv(config),
            )


procgen_frameskips: Dict[str, int] = {
    "bigfish": 8,
    "bossfight": 8,
    "chaser": 2,
    "climber": 6,
    "coinrun": 8,
    "dodgeball": 8,
    "fruitbot": 8,
    "heist": 2,
    "jumper": 8,
    "leaper": 6,
    "maze": 1,
    "miner": 1,
    "ninja": 8,
    "plunder": 8,
    "starpilot": 8,
}
procgen_num_actions: Dict[str, int] = {
    "bigfish": 9,
    "bossfight": 10,
    "caveflyer": 10,
    "chaser": 9,
    "climber": 9,
    "coinrun": 9,
    "dodgeball": 10,
    "fruitbot": 9,
    "heist": 9,
    "jumper": 9,
    "leaper": 9,
    "maze": 9,
    "miner": 9,
    "ninja": 13,
    "plunder": 10,
    "starpilot": 11,
}


def register_procgen_envs():
    for env_name, frameskip in procgen_frameskips.items():
        distribution_modes: List[Literal["easy", "hard", "exploration"]] = [
            "easy",
            "hard",
        ]
        if env_name in EXPLORATION_LEVEL_SEEDS:
            distribution_modes.append("exploration")
        for distribution_mode in distribution_modes:
            levels = [0, 1, 2] if distribution_mode == "easy" else [0]
            for level in levels:
                for horizon in [10, 20, 30, 40, 50, 70, 100, 200]:
                    env_id = f"{env_name}_{distribution_mode}_l{level}_{horizon}_fs{frameskip}"
                    config: DeterministicProcgenEnvConfig = {
                        "env_name": env_name,
                        "distribution_mode": distribution_mode,
                        "level": level,
                        "frameskip": frameskip,
                        "framestack": 1,
                        "num_actions": procgen_num_actions[env_name],
                        "horizon": horizon,
                    }
                    register(
                        id=f"{GYM_NAMESPACE}/{env_id}-v0",
                        entry_point=DeterministicProcgenEnv,  # type: ignore
                        kwargs={"config": config},
                    )
                    register_rllib_env_if_installed(
                        f"{GYM_NAMESPACE}/{env_id}-v0",
                        lambda _: DeterministicProcgenEnv(config),
                    )


T = TypeVar("T")


def subsets(l: List[T]) -> Iterable[List[T]]:  # noqa: E741
    if len(l) == 0:
        yield []
    else:
        for rest_subset in subsets(l[1:]):
            yield rest_subset
            yield [l[0]] + rest_subset


def register_minigrid_envs():
    env_spec: EnvSpec
    for env_spec in list(registry.values()):
        if env_spec.id.startswith("MiniGrid") and env_spec.id.endswith("-v0"):
            base_env = gym.make(env_spec.id)
            base_env.reset()
            all_shaping_functions = (
                MinigridShapedRewardWrapper.get_meaningful_shaping_functions(
                    base_env.unwrapped
                )
            )

            for shaping_functions in subsets(all_shaping_functions):
                shaping_config = copy.copy(DEFAULT_SHAPED_REWARD_CONFIG)
                for shaping_function in shaping_functions:
                    shaping_config[shaping_function] = True  # type: ignore

                env_id = env_spec.id[:-3]  # Remove the -v0.
                if shaping_functions:
                    env_id += "-"
                if shaping_config["distance"]:
                    env_id += "Distance"
                if shaping_config["open_doors"]:
                    env_id += "OpenDoors"
                if shaping_config["picked_up_objects"]:
                    env_id += "Pickup"
                if shaping_config["lava"]:
                    env_id += "Lava"
                if shaping_functions:
                    env_id += "Shaped"

                env_maker = build_minigrid_env_maker(
                    env_spec.id,
                    shaping_config=shaping_config,
                )
                register(
                    id=f"{GYM_NAMESPACE}/{env_id}-v0",
                    entry_point=env_maker,
                    max_episode_steps=100,
                )
                register_rllib_env_if_installed(
                    f"{GYM_NAMESPACE}/{env_id}-v0",
                    env_maker,
                )

                image_obs_env_maker = build_minigrid_env_maker(
                    env_spec.id,
                    shaping_config=shaping_config,
                    flat_observations=False,
                )
                register(
                    id=f"{GYM_NAMESPACE}/{env_id}-ImgObs-v0",
                    entry_point=image_obs_env_maker,
                    max_episode_steps=100,
                )
                register_rllib_env_if_installed(
                    f"{GYM_NAMESPACE}/{env_id}-ImgObs-v0",
                    image_obs_env_maker,
                )


def register_sticky_envs():
    env_spec: EnvSpec
    for env_spec in list(registry.values()):
        if env_spec.id.startswith(f"{GYM_NAMESPACE}/"):
            env_id_parts = env_spec.id.split("-")
            env_id_parts.insert(-1, "Sticky")
            env_id = "-".join(env_id_parts)
            base_env_id = env_spec.id
            if "MiniGrid" in base_env_id:
                # Always use ImgObs version for sticky environments.
                if "-ImgObs-" in base_env_id:
                    env_id = env_id.replace("-ImgObs", "")
                else:
                    continue
            env_creator = (
                lambda base_env_id=base_env_id, **kwargs: StickyActionsWrapper(
                    gym.make(base_env_id)
                )
            )
            register(
                id=env_id,
                entry_point=env_creator,
            )


_envs_registered = False


def register_all():
    global _envs_registered
    if not _envs_registered:
        register_atari_envs()
        register_procgen_envs()
        register_minigrid_envs()
        register_sticky_envs()
        _envs_registered = True


register_all()
