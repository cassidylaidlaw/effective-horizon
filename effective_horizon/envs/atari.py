from collections import deque
from typing import Any, Deque, Optional, Tuple, Type, TypedDict, cast

import ale_py.roms as roms
import gymnasium as gym
import numpy as np
from ale_py.env.gym import AtariEnv as GymAtariEnv
from ale_py.roms.utils import rom_id_to_name, rom_name_to_id
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import TimeLimit

from effective_horizon.image_utils import resize, rgb2gray

from .utils import convert_gym_space, register_rllib_env_if_installed


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset.

        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.gym_env.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.gym_env.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = rgb2gray(frame)
        frame = resize(frame, height=self.height, width=self.width)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        ob, infos = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), infos

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class DeterministicAtariEnv(GymAtariEnv):
    def __init__(self, *args, **kwargs):
        self.current_seed = None
        self.start_state = None

        super().__init__(*args, **kwargs)

    def seed(self, seed: Optional[int] = None) -> Tuple[int, int]:
        ss = np.random.SeedSequence(seed)
        seed1, seed2 = ss.generate_state(n_words=2)
        self.np_random = np.random.default_rng(seed1)
        if seed is not None:
            seed2 = seed
        if seed == self.current_seed and self.start_state is not None:
            self.ale.restoreSystemState(self.start_state)
        else:
            self.ale.setInt("random_seed", int(np.array(seed2).astype(np.int32)))
            self.ale.setInt("system_random_seed", 4753849)

            self.ale.loadROM(getattr(roms, self._game))

            if self._game_mode is not None:
                self.ale.setMode(self._game_mode)
            if self._game_difficulty is not None:
                self.ale.setDifficulty(self._game_difficulty)
            self.start_state = self.ale.cloneSystemState()
            self.current_seed = seed

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

    register_rllib_env_if_installed(
        f"{name}DeterministicSeed-v0",
        build_env_maker(
            DeterministicAtariEnv,
            deterministic_kwargs,
            max_episode_steps=100000,
            id="{name}DeterministicSeed-v0",
        ),
    )

    no_frameskip_one_life_kwargs = dict(
        game=rom,
        obs_type="rgb",
        repeat_action_probability=0,
        full_action_space=False,
        frameskip=1,
    )
    register_rllib_env_if_installed(
        f"{name}NoFrameskipOneLife-v4",
        build_env_maker(
            OneLifeAtariEnv,
            no_frameskip_one_life_kwargs,
            max_episode_steps=100000 * frameskip,
            id=f"{name}NoFrameskipOneLife-v4",
        ),
    )

    one_life_kwargs = dict(
        game=rom,
        obs_type="rgb",
        repeat_action_probability=0.25,
        full_action_space=False,
        frameskip=5,
    )
    register_rllib_env_if_installed(
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
    fire_on_reset: bool
    clip_rewards: bool


class AtariEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: AtariEnvConfig, render_mode="rgb_array"):
        super().__init__()

        self.config = config
        env_class: Type[GymAtariEnv] = (
            DeterministicAtariEnv if self.config["deterministic"] else GymAtariEnv
        )
        self.atari_env = env_class(
            game=self.config["rom_file"],
            frameskip=self.config["frameskip"],
            repeat_action_probability=(
                0
                if self.config["deterministic"]
                else self.config["repeat_action_probability"]
            ),
            full_action_space=False,
            render_mode=render_mode,
        )
        atari_env = cast(Any, self.atari_env)
        atari_env.gym_env = self.atari_env
        self.horizon = self.config["horizon"]
        self.reward_scale = self.config["reward_scale"]

        self.env: gym.Env = cast(gym.Env, self.atari_env)
        self.env = WarpFrame(self.env, dim=84)
        self.env = FrameStack(self.env, 4)
        if (
            self.config.get("fire_on_reset", False)
            and "FIRE" in self.atari_env.get_action_meanings()
        ):
            self.env = FireResetEnv(self.env)
        if self.config["clip_rewards"]:
            self.env = ClipRewardEnv(self.env)

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
        reward = float(reward)
        self.t += 1

        if reward != 0 and self.config["done_on_reward"]:
            terminated = True
        if self.atari_env.ale.lives() < self.lives:
            if self.config["done_on_life_lost"]:
                terminated = True
            else:
                if (
                    self.config.get("fire_on_reset", False)
                    and "FIRE" in self.atari_env.get_action_meanings()
                ):
                    if not (terminated or truncated):
                        (
                            obs,
                            additional_reward,
                            terminated,
                            truncated,
                            info,
                        ) = self.env.step(1)
                        reward += float(additional_reward)
                    if not (terminated or truncated):
                        (
                            obs,
                            additional_reward,
                            terminated,
                            truncated,
                            info,
                        ) = self.env.step(2)
                        reward += float(additional_reward)

        self.lives = self.atari_env.ale.lives()
        if self.t >= self.horizon:
            if not terminated:
                terminated = True
                reward += self.config["no_done_reward"]

                for _ in range(self.config["noops_after_horizon"]):
                    obs, additional_reward, terminated, truncated, info = self.env.step(
                        0
                    )
                    reward += float(additional_reward)
                    if terminated:
                        break

        return obs, reward * self.reward_scale, terminated, truncated, info

    def render(self):
        return self.env.render()


register_rllib_env_if_installed("atari", AtariEnv)
gym.register("BRIDGE/Atari-v0", AtariEnv)  # type: ignore[arg-type]
