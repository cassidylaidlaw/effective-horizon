import itertools
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TensorDict,
)
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv


class DummyPolicy(BasePolicy):
    def _predict(self, observation, deterministic: bool = False):
        return torch.zeros(1, dtype=torch.long)


class EpisodeInfo(NamedTuple):
    actions: List[int]
    reward_to_go: float
    length: int


class GORP(BaseAlgorithm):
    policy_aliases = {"ActionSequencePolicy": DummyPolicy}

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        episodes_per_action_sequence: int = 1,
        planning_depth: int = 1,
        gamma: float = 0.99,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
        support_multi_env: bool = True,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            supported_action_spaces=supported_action_spaces,
        )
        self.gamma = gamma
        self.episodes_per_action_sequence = episodes_per_action_sequence
        self.planning_depth = planning_depth

        self._last_state: Optional[torch.Tensor] = None

        self._iterations_with_no_exploration = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.actions: List[int] = []

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        assert isinstance(self.action_space, spaces.Discrete)
        num_actions = self.action_space.n

        while self.num_timesteps < total_timesteps:
            action_sequences: List[Tuple[int, ...]] = list(
                itertools.product(*[range(num_actions)] * self.planning_depth),
            )

            best_action_sequence: Tuple[int, ...] = ()
            best_mean_reward_to_go = float("-inf")
            continue_training = True
            iteration_timesteps = 0
            exploration_timesteps = 0
            for action_sequence in action_sequences:
                rewards_to_go: List[float] = []
                for _ in range(self.episodes_per_action_sequence):
                    rollout, episode_info = self.collect_rollout(
                        self.env,
                        action_sequence,
                        callback=callback,
                        log_interval=log_interval,
                    )
                    continue_training = continue_training and rollout.continue_training
                    rewards_to_go.append(episode_info.reward_to_go)
                    iteration_timesteps += episode_info.length
                    exploration_timesteps += max(
                        episode_info.length - len(self.actions), 0
                    )
                mean_reward_to_go = float(np.mean(rewards_to_go))
                if mean_reward_to_go > best_mean_reward_to_go:
                    best_action_sequence = action_sequence
                    best_mean_reward_to_go = mean_reward_to_go

            if not continue_training:
                break

            self.actions.append(best_action_sequence[0])

            exploration_rate = exploration_timesteps / iteration_timesteps
            self.logger.record("rollout/exploration_rate", exploration_rate)
            if exploration_rate == 0:
                self._iterations_with_no_exploration += 1
                if self._iterations_with_no_exploration > 10:
                    print("No exploration for 10 iterations, stopping training")
                    break
            else:
                self._iterations_with_no_exploration = 0

        callback.on_training_end()

        return self

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray]]]:
        assert isinstance(observation, np.ndarray)
        batch_size = observation.shape[0]

        if state is None:
            timesteps = np.zeros(batch_size, dtype=int)
        else:
            (timesteps,) = state

        if episode_start is not None:
            timesteps[episode_start] = 0

        actions = np.full(batch_size, -1, dtype=int)
        new_states = (timesteps + 1,)

        for i in range(batch_size):
            if timesteps[i] < len(self.actions):
                actions[i] = self.actions[timesteps[i]]
            elif deterministic:
                actions[i] = 0
            else:
                actions[i] = self.action_space.sample()

        return actions, new_states

    def collect_rollout(
        self,
        env: VecEnv,
        action_sequence: Tuple[int, ...],
        *,
        callback: BaseCallback,
        log_interval: Optional[int] = None,
    ) -> Tuple[RolloutReturn, EpisodeInfo]:
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "GORP only supports one environment"

        callback.on_rollout_start()

        t = 0
        done = False
        reward_to_go = 0
        actions = []

        while not done:
            if t < len(self.actions):
                action = self.actions[t]
            elif t < len(self.actions) + len(action_sequence):
                action = action_sequence[t - len(self.actions)]
            else:
                action = self.action_space.sample()

            new_obs, rewards, dones, infos = env.step(np.array([action]))

            self.num_timesteps += 1
            num_collected_steps += 1

            (reward,) = rewards
            (done,) = dones

            actions.append(action)
            if t >= len(self.actions):
                reward_to_go += self.gamma ** (t - len(self.actions)) * reward
            t += 1
            episode_info = EpisodeInfo(actions, reward_to_go, t)

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return (
                    RolloutReturn(
                        num_collected_steps,
                        num_collected_episodes,
                        continue_training=False,
                    ),
                    episode_info,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

        if done:
            # Update stats
            num_collected_episodes += 1
            self._episode_num += 1

            # Log training infos
            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        callback.on_rollout_end()

        return (
            RolloutReturn(
                num_collected_steps,
                num_collected_episodes,
                continue_training=True,
            ),
            episode_info,
        )

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        assert self.ep_info_buffer is not None
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def get_parameters(self) -> Dict[str, Dict]:
        return {"policy": {"actions": self.actions}}

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device="auto",
    ) -> None:
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        self.actions = cast(Any, params)["policy"]["actions"]
