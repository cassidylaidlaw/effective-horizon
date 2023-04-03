from typing import Literal, Tuple, TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from procgen import ProcgenGym3Env
from ray.tune.registry import register_env


class BaseProcgenEnvConfig(TypedDict):
    env_name: str
    distribution_mode: Literal["easy", "hard", "exploration"]
    frameskip: int
    num_actions: int


class ProcgenEnvConfig(BaseProcgenEnvConfig):
    num_levels: int
    start_level: int


class ProcgenEnv(gym.Env):
    def __init__(self, config: ProcgenEnvConfig, horizon: float = np.inf, **kwargs):
        kwargs.update(
            {
                "num": 1,
                "env_name": config["env_name"],
                "distribution_mode": config["distribution_mode"],
                "rand_seed": 0,
            }
        )
        if config["distribution_mode"] != "exploration":
            kwargs.update(
                {
                    "num_levels": config["num_levels"],
                    "start_level": config["start_level"],
                }
            )
        self.env = ProcgenGym3Env(**kwargs)
        self.frameskip = config["frameskip"]
        self.horizon = horizon
        self.action_space = spaces.Discrete(config["num_actions"])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=self.env.ob_space["rgb"].shape,
        )

    def reset(self, *args, **kwargs):
        reward, base_obs, first = self.env.observe()
        while not first[0]:
            self.env.act(np.array([0]))
            reward, base_obs, first = self.env.observe()
        self.done = False
        self.t = 0
        return self._get_obs(base_obs), {}

    def _get_obs(self, base_obs):
        return base_obs["rgb"][0] / 255

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            reward, base_obs, first = self.env.observe()
            return self._get_obs(base_obs), 0.0, self.done, False, {}

        total_reward = 0.0
        for _ in range(self.frameskip):
            self.env.act(np.array([action]))
            reward, base_obs, first = self.env.observe()
            total_reward += reward[0]

            if first[0]:
                self.done = True
                break

        self.t += 1
        if self.t >= self.horizon:
            self.done = True

        return self._get_obs(base_obs), total_reward, self.done, False, {}

    def render(self, mode="human"):
        assert mode == "rgb_array"
        reward, base_obs, first = self.env.observe()
        return self._get_obs(base_obs)


register_env("procgen", ProcgenEnv)


class DeterministicProcgenEnvConfig(BaseProcgenEnvConfig):
    level: int
    horizon: int


class DeterministicProcgenEnv(ProcgenEnv):
    def __init__(self, config: DeterministicProcgenEnvConfig, **kwargs):
        nondeterministic_config: ProcgenEnvConfig = {
            "env_name": config["env_name"],
            "distribution_mode": config["distribution_mode"],
            "frameskip": config["frameskip"],
            "num_actions": config["num_actions"],
            "start_level": config["level"],
            "num_levels": 1,
        }
        super().__init__(nondeterministic_config, horizon=config["horizon"], **kwargs)
        self.start_state = self.env.get_state()

    def reset(self, *args, **kwargs):
        self.env.set_state(self.start_state)
        reward, base_obs, first = self.env.observe()
        assert first[0] and reward[0] == 0
        self.done = False
        self.t = 0
        return self._get_obs(base_obs), {}
