from typing import Any, Optional, Tuple, Type, TypedDict, cast

import ale_py.roms as roms
import gymnasium as gym
import numpy as np
from ale_py.env.gym import AtariEnv as GymAtariEnv
from ale_py.roms.utils import rom_id_to_name, rom_name_to_id
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import TimeLimit
from ray.rllib.env.wrappers.atari_wrappers import FrameStack, WarpFrame
from ray.tune.registry import register_env

from .utils import convert_gym_space


class DeterministicAtariEnv(GymAtariEnv):
    def seed(self, seed: Optional[int] = None) -> Tuple[int, int]:
        ss = np.random.SeedSequence(seed)
        seed1, seed2 = ss.generate_state(n_words=2)
        self.np_random = np.random.default_rng(seed1)
        if seed is not None:
            seed2 = seed
        self.ale.setInt("random_seed", int(np.array(seed2).astype(np.int32)))
        self.ale.setInt("system_random_seed", 4753849)

        self.ale.loadROM(getattr(roms, self._game))

        if self._game_mode is not None:
            self.ale.setMode(self._game_mode)
        if self._game_difficulty is not None:
            self.ale.setDifficulty(self._game_difficulty)

        return (seed1, seed2)

    def reset(self, *args, **kwargs):
        kwargs["seed"] = 0
        return super().reset(*args, **kwargs)


class OneLifeAtariEnv(GymAtariEnv):
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.lives = self.ale.lives()
        return obs, info

    def step(self, action_ind: int):
        obs, reward, terminated, truncated, info = super().step(action_ind)
        lives = self.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        else:
            self.lives = lives
        return obs, reward, terminated, truncated, info


def build_env_maker(
    env_cls: Type[GymAtariEnv],
    env_kwargs: dict,
    max_episode_steps: int,
    id: Optional[str] = None,
):
    def env_maker(config):
        env = cast(gym.Env, env_cls(**env_kwargs))
        if id is not None:
            cast(Any, env).spec = EnvSpec(id, entry_point=env_cls.__name__)
        env = TimeLimit(env, max_episode_steps)
        return env

    return env_maker


all_games = list(map(rom_name_to_id, roms.__all__))
for rom in all_games:
    name = rom_id_to_name(rom)
    frameskip = 3 if rom == "space_invaders" else 4

    deterministic_kwargs = dict(
        game=rom,
        obs_type="rgb",
        frameskip=5,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    register_env(
        f"{name}DeterministicSeed-v0",
        build_env_maker(
            DeterministicAtariEnv,
            deterministic_kwargs,
            max_episode_steps=100000,
            id="{name}DeterministicSeed-v0",
        ),
    )

    # register(
    #     id=f"{name}NoFrameskipOneLife-v4",
    #     entry_point="effective_horizon.envs.atari:OneLifeAtariEnv",
    #     kwargs={
    #         "game": rom,
    #         "obs_type": "rgb",
    #         "frameskip": 1,
    #     },
    #     max_episode_steps=100000 * frameskip,
    # )
    no_frameskip_one_life_kwargs = dict(
        game=rom,
        obs_type="rgb",
        repeat_action_probability=0,
        full_action_space=False,
        frameskip=1,
    )
    register_env(
        f"{name}NoFrameskipOneLife-v4",
        build_env_maker(
            OneLifeAtariEnv,
            no_frameskip_one_life_kwargs,
            max_episode_steps=100000 * frameskip,
            id=f"{name}NoFrameskipOneLife-v4",
        ),
    )

    # register(
    #     id=f"{name}OneLife-v5",
    #     entry_point="effective_horizon.envs.atari:OneLifeAtariEnv",
    #     kwargs={
    #         "game": rom,
    #         "obs_type": "rgb",
    #         "repeat_action_probability": 0.25,
    #         "full_action_space": False,
    #         "frameskip": 4,
    #     },
    #     max_episode_steps=108000 // 4,
    # )
    one_life_kwargs = dict(
        game=rom,
        obs_type="rgb",
        repeat_action_probability=0.25,
        full_action_space=False,
        frameskip=5,
    )
    register_env(
        f"{name}OneLife-v5",
        build_env_maker(
            OneLifeAtariEnv,
            one_life_kwargs,
            max_episode_steps=108000 // 4,
            id=f"{name}OneLife-v5",
        ),
    )


class AtariEnvConfig(TypedDict):
    rom_file: str
    horizon: int
    done_on_reward: bool
    done_on_life_lost: bool
    no_done_reward: float
    noops_after_horizon: int
    frameskip: int
    repeat_action_probability: float
    deterministic: bool
    reward_scale: float


class AtariEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: AtariEnvConfig):
        super().__init__()

        self.config = config
        env_class: Type[GymAtariEnv] = (
            DeterministicAtariEnv if self.config["deterministic"] else GymAtariEnv
        )
        self.atari_env = env_class(
            game=self.config["rom_file"],
            frameskip=self.config["frameskip"],
            repeat_action_probability=0
            if self.config["deterministic"]
            else self.config["repeat_action_probability"],
            full_action_space=False,
        )
        self.horizon = self.config["horizon"]
        self.reward_scale = self.config["reward_scale"]

        self.env: gym.Env = cast(gym.Env, self.atari_env)
        self.env = WarpFrame(self.env, dim=84)
        self.env = FrameStack(self.env, 4)

        self.observation_space = convert_gym_space(self.env.observation_space)
        self.action_space = convert_gym_space(self.env.action_space)
        self.spec = self.env.spec

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.t = 0
        self.lives = self.atari_env.ale.lives()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        terminated = self.atari_env.ale.game_over()
        obs, reward, _, truncated, info = self.env.step(action)
        self.t += 1

        if reward != 0 and self.config["done_on_reward"]:
            terminated = True
        if self.atari_env.ale.lives() < self.lives and self.config["done_on_life_lost"]:
            terminated = True
        self.lives = self.atari_env.ale.lives()
        if self.t >= self.horizon:
            if not terminated:
                terminated = True
                reward += self.config["no_done_reward"]

                for _ in range(self.config["noops_after_horizon"]):
                    obs, additional_reward, terminated, truncated, info = self.env.step(
                        0
                    )
                    reward += additional_reward
                    if terminated:
                        break

        return obs, reward * self.reward_scale, terminated, truncated, info


register_env("atari", AtariEnv)
