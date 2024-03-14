import io
import pathlib
import sys
import time
from copy import deepcopy
from math import ceil
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import (
    BaseFeaturesExtractor,
    BasePolicy,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
)
from stable_baselines3.common.utils import safe_mean, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from torch.func import functional_call, stack_module_state


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for SQIRL.
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        planning_depth: int = 1,
        net_arch: Optional[List[int]] = None,
        dueling: bool = False,
        num_atoms: int = 1,
        v_min: float = -10,
        v_max: float = 10,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.dueling = dueling
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.num_actions = int(self.action_space.n)
        self.planning_depth = planning_depth
        num_outputs = self.num_actions * self.planning_depth
        if self.num_atoms > 1:
            num_outputs *= self.num_atoms
        q_net = create_mlp(
            self.features_dim, num_outputs, self.net_arch, self.activation_fn
        )
        self.q_net = nn.Sequential(*q_net)
        if self.dueling:
            if self.num_atoms > 1:
                raise ValueError("Dueling network not supported for distributional Q")
            value_net = create_mlp(
                self.features_dim,
                self.planning_depth,
                self.net_arch,
                self.activation_fn,
            )
            self.value_net = nn.Sequential(*value_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        q_outputs: torch.Tensor = self.q_net(features).reshape(
            -1,
            self.planning_depth,
            self.num_actions,
            self.num_atoms,
        )

        if self.num_atoms > 1:
            return q_outputs
        else:
            q_values = q_outputs.squeeze(dim=3)
            if self.dueling:
                q_values = q_values - q_values.mean(dim=2, keepdim=True)
                values = self.value_net(features).reshape(-1, self.planning_depth, 1)
                q_values = q_values + values

            return q_values

    def get_q_values(
        self, obs: torch.Tensor, *, state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if state_dict is None:
            q_outputs = self.forward(obs)
        else:
            q_outputs = functional_call(self, state_dict, (obs,))
        if self.num_atoms > 1:
            q_probs = F.softmax(q_outputs, dim=-1)
            atom_values = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=q_outputs.device
            )
            q_values = (q_probs * atom_values).sum(dim=-1)
        else:
            q_values = q_outputs
        return q_values

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = True,
        *,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert not isinstance(observation, dict)
        q_values = self.get_q_values(observation, state_dict=state_dict)
        action = q_values[:, -1, :].argmax(dim=1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                planning_depth=self.planning_depth,
                dueling=self.dueling,
                num_atoms=self.num_atoms,
                v_min=self.v_min,
                v_max=self.v_max,
            )
        )
        return data


class SQIRLPolicy(BasePolicy):
    """
    This class combines all the Q-networks used in SQIRL. In the case where
    a single Q-network is used for Q_1, Q_2, ..., Q_k, it wraps exactly one QNetwork.
    In the case where separate Q-networks are used for Q_1, Q_2, ..., Q_k, it wraps
    k QNetworks (k == planning_depth).
    """

    q_net: QNetwork
    _predict_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        planning_depth: int = 1,
        separate_networks: bool = False,
        net_arch: Optional[List[int]] = None,
        dueling: bool = False,
        num_atoms: int = 1,
        v_min: float = -10,
        v_max: float = 10,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        net_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.planning_depth = planning_depth
        self.separate_networks = separate_networks

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "dueling": dueling,
            "num_atoms": num_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "planning_depth": 1 if separate_networks else self.planning_depth,
        }
        if net_args is not None:
            self.net_args.update(net_args)

        self._build(lr_schedule)

    def __setattr__(self, name, value):
        if name == "inference_q_net":
            # Don't attach inference_q_net to the list of sub-modules of this policy,
            # since we don't want to save its parameters or try to move them to
            # another device, etc.
            return object.__setattr__(self, name, value)
        else:
            return super().__setattr__(name, value)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create a Q-network with no built-in parameters to be used for inference
        # and as a dummy for running multiple Q-networks in parallel via vmap.
        self.inference_q_net = self.make_q_net().to("meta")

        if self.separate_networks:
            q_nets = [self.make_q_net() for _ in range(self.planning_depth)]

            def get_q_net_output(params, buffers, obs):
                return functional_call(self.inference_q_net, (params, buffers), (obs,))

            self.q_net_vmap = torch.vmap(get_q_net_output, in_dims=(0, 0, None))
            q_net_params, q_net_buffers = stack_module_state(q_nets)

            self.q_net_params: Dict[str, nn.Parameter] = {}
            self.q_net_buffers: Dict[str, torch.Tensor] = {}
            for param_name, param_value in q_net_params.items():
                param = nn.Parameter(param_value)
                self.q_net_params[param_name] = param
                param_name = param_name.replace(".", "__")
                self.register_parameter(f"q_nets__{param_name}", param)
            for buffer_name, buffer_value in q_net_buffers.items():
                self.q_net_buffers[buffer_name] = buffer_value
                buffer_name = buffer_name.replace(".", "__")
                self.register_buffer(f"q_nets__{buffer_name}", buffer_value)
        else:
            self.q_net = self.make_q_net()

        # Setup optimizer with initial learning rate.
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return QNetwork(**net_args).to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = True,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if state_dict is None:
            if self.separate_networks:
                state_dict = self._get_q_network_state_dict()
                return self.inference_q_net._predict(
                    obs, deterministic=deterministic, state_dict=state_dict
                )
            else:
                return self.q_net._predict(obs, deterministic=deterministic)
        else:
            return self.inference_q_net._predict(
                obs, deterministic=deterministic, state_dict=state_dict
            )

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = True,
        *,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert not isinstance(observation, dict)
        if state_dict is None:
            state_dict = self._predict_state_dict
        return cast(
            torch.Tensor,
            self(observation, deterministic=deterministic, state_dict=state_dict),
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        *,
        state_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if state_dict is not None:
            self._predict_state_dict = {
                param_name: torch.from_numpy(param_value).to(self.device)
                for param_name, param_value in state_dict.items()
            }
        actions, state = super().predict(
            observation, state, episode_start, deterministic
        )
        self._predict_state_dict = None
        return actions, state

    def get_q_network_outputs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.separate_networks:
            q_net_outputs: torch.Tensor = self.q_net_vmap(
                self.q_net_params, self.q_net_buffers, obs
            )
            assert isinstance(self.action_space, spaces.Discrete)
            assert q_net_outputs.size()[:4] == (
                self.planning_depth,
                obs.size(0),
                1,
                self.action_space.n,
            )
            return q_net_outputs.squeeze(2).transpose(0, 1)
        else:
            return cast(torch.Tensor, self.q_net(obs))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                net_args=self.net_args,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        if self.training != mode:
            if not self.separate_networks:
                self.q_net.set_training_mode(mode)
            self.inference_q_net.set_training_mode(mode)
            self.training = mode

    def _get_q_network_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.separate_networks:
            return {
                **{
                    param_name: param_value.data[-1]
                    for param_name, param_value in self.q_net_params.items()
                },
                **{
                    buffer_name: buffer_value[-1]
                    for buffer_name, buffer_value in self.q_net_buffers.items()
                },
            }
        else:
            return self.q_net.state_dict()

    def get_q_network_state_dict_copy(self) -> Dict[str, np.ndarray]:
        state_dict = self._get_q_network_state_dict()
        return {
            param_name: param_value.detach().cpu().numpy().copy()
            for param_name, param_value in state_dict.items()
        }


MlpPolicy = SQIRLPolicy


class CnnPolicy(SQIRLPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            **kwargs,
        )


class ReplayBufferSamplesWithValueTargets(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    value_targets: torch.Tensor


class ReplayBufferWithValueTargets(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        self.value_targets = np.zeros_like(self.rewards)
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
        value_target: Optional[np.ndarray] = None,
        priority: Optional[np.ndarray] = None,
    ) -> None:
        if value_target is not None:
            self.value_targets[self.pos] = np.array(value_target).copy()
        if priority is None:
            priority = np.ones(reward.shape, dtype=np.float32)
        self.priorities[self.pos] = np.array(priority).copy()

        return super().add(obs, next_obs, action, reward, done, infos)

    def multiply_priorities(self, factor: float) -> None:
        self.priorities *= factor

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        weights = self.priorities.copy()
        weights[self.pos] = 0
        batch_inds = np.random.choice(
            self.buffer_size, size=batch_size, p=weights.sum(axis=1) / weights.sum()
        )
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        samples = super()._get_samples(batch_inds, env)
        assert self.n_envs == 1
        return ReplayBufferSamplesWithValueTargets(
            *samples, self.to_torch(self.value_targets[batch_inds, 0])
        )


class Episode(NamedTuple):
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[np.ndarray]
    infos: List[Dict[str, Any]]

    @property
    def t(self):
        return len(self.rewards)

    @classmethod
    def empty(cls):
        return cls(
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            infos=[],
        )


class SQIRL(BaseAlgorithm):
    policy_aliases = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
    }

    policy: SQIRLPolicy
    ema_policy: Optional[SQIRLPolicy]

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        intra_step_lr_decay: float = 1.0,
        ema_alpha: float = 0,
        buffer_size: int = 1_000_000,  # 1e6
        episodes_per_timestep: int = 1,
        planning_depth: int = 1,
        separate_networks: bool = False,
        batch_size: int = 256,
        gamma: float = 0.99,
        n_epochs: int = 1,
        max_grad_norm: float = 10,
        reward_scale: float = 1.0,
        use_huber_loss: bool = False,
        scale_losses: bool = False,
        loss_scale_smoothing: float = 0.99,
        num_atoms: int = 1,
        v_min: float = -10,
        v_max: float = 10,
        exploration_fraction: float = 0,
        replay_buffer_class: Optional[Type[ReplayBufferWithValueTargets]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        replay_priority_factor: float = 1.0,
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
        self.intra_step_lr_decay = intra_step_lr_decay
        self.ema_alpha = ema_alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.episodes_per_timestep = episodes_per_timestep
        self.planning_depth = planning_depth
        self.separate_networks = separate_networks
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale
        self.use_huber_loss = use_huber_loss
        self.scale_losses = scale_losses
        self.loss_scale_decay = loss_scale_smoothing
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.exploration_fraction = exploration_fraction
        self.replay_buffer: Optional[ReplayBufferWithValueTargets] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self.replay_priority_factor = replay_priority_factor
        self._episode_storage = None

        self._current_timestep = 0
        self._last_state: Optional[torch.Tensor] = None

        self.state_dicts: List[Dict[str, np.ndarray]] = []

        self._iterations_with_no_exploration = 0

        if self.scale_losses:
            self.loss_running_mean = [np.nan for _ in range(self.planning_depth)]

        if self.use_huber_loss and self.num_atoms > 1:
            raise ValueError("Huber loss not supported for distributional Q")

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            self.replay_buffer_class = ReplayBufferWithValueTargets

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=1,
                **replay_buffer_kwargs,
            )

        policy_kwargs = {
            **self.policy_kwargs,
            "planning_depth": self.planning_depth,
            "separate_networks": self.separate_networks,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

        assert issubclass(self.policy_class, SQIRLPolicy)
        assert isinstance(self.action_space, spaces.Discrete)
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if self.ema_alpha > 0:
            self.ema_policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **policy_kwargs,
            )
            self.ema_policy = self.ema_policy.to(self.device)
            self.ema_policy.load_state_dict(self.policy.state_dict())
            self.ema_policy.eval()

    def save_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(
            self.replay_buffer, ReplayBufferWithValueTargets
        ), "The replay buffer must inherit from ReplayBufferWithValueTargets class"

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

        assert self.env is not None and self.replay_buffer is not None

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                callback=callback,
                num_episodes=self.episodes_per_timestep,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            self.purely_exploring = (
                self.num_timesteps < self.exploration_fraction * total_timesteps
            )

            if rollout.continue_training is False:
                break

            gradient_steps = ceil(
                rollout.episode_timesteps * self.n_epochs / self.batch_size
            )
            self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            if not self.purely_exploring:
                # Freeze Q-network for this timestep
                self._add_state_dict(self._get_q_network_state_dict_copy())
                self._current_timestep += 1
                assert len(self.state_dicts) == self._current_timestep

                # Reduce priority of samples in replay buffer by the specified factor.
                self.replay_buffer.multiply_priorities(self.replay_priority_factor)

        callback.on_training_end()

        return self

    def _update_ema_parameters(self):
        assert self.ema_alpha > 0 and self.ema_policy is not None
        for param, ema_param in zip(
            self.policy.parameters(), self.ema_policy.parameters()
        ):
            ema_param.data.mul_(self.ema_alpha).add_((1 - self.ema_alpha) * param.data)
        for buffer, ema_buffer in zip(self.policy.buffers(), self.ema_policy.buffers()):
            ema_buffer.mul_(self.ema_alpha).add_((1 - self.ema_alpha) * buffer)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to schedule
        base_lr = self.lr_schedule(self._current_timestep)
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )
        update_learning_rate(self.policy.optimizer, base_lr)

        assert self.replay_buffer is not None
        assert isinstance(self.policy, SQIRLPolicy)

        losses = []
        losses_by_q_function: List[List[float]] = [
            [] for _ in range(self.planning_depth)
        ]
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = cast(
                ReplayBufferSamplesWithValueTargets,
                self.replay_buffer.sample(batch_size, env=self._vec_normalize_env),
            )
            target_q_values = replay_data.value_targets * self.reward_scale

            if self.num_atoms == 1:
                target_q_values = target_q_values[:, None]
                if self.planning_depth > 1:
                    with torch.no_grad():
                        next_q_values = self.policy.get_q_network_outputs(
                            replay_data.next_observations
                        )
                    next_q_values[replay_data.dones[:, 0] == True] = 0.0  # noqa: E712
                    target_q_values = torch.concat(
                        [
                            target_q_values,
                            self.gamma * next_q_values[:, :-1].max(dim=2).values
                            + replay_data.rewards * self.reward_scale,
                        ],
                        dim=1,
                    )
                target_q_values = target_q_values.detach()

                # Get current Q-values estimates
                current_q_values = self.policy.get_q_network_outputs(
                    replay_data.observations
                )

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = current_q_values[
                    torch.arange(batch_size), :, replay_data.actions[:, 0].long()
                ]

                if self.use_huber_loss:
                    unreduced_loss = F.smooth_l1_loss(
                        current_q_values, target_q_values, reduction="none"
                    )
                else:
                    unreduced_loss = (current_q_values - target_q_values) ** 2
            else:
                target_dist = torch.zeros(
                    batch_size, self.planning_depth, self.num_atoms, device=self.device
                )

                atom_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
                atom_values = torch.linspace(
                    self.v_min, self.v_max, self.num_atoms, device=self.device
                )

                target_q_values = target_q_values.clamp(self.v_min, self.v_max)
                b = (target_q_values - self.v_min) / atom_delta
                l, u = b.floor().long(), b.ceil().long()
                floor_equal_ceil = ((u - l) < 0.5).float()
                target_dist[range(batch_size), 0, l] += u - b + floor_equal_ceil
                target_dist[range(batch_size), 0, u] += b - l

                if self.planning_depth > 1:
                    next_dist_logits = self.policy.get_q_network_outputs(
                        replay_data.next_observations
                    )
                    next_dist_probs = next_dist_logits.softmax(dim=-1)
                    next_q_values = (next_dist_probs * atom_values).sum(dim=-1)
                    next_actions = next_q_values.argmax(dim=2)
                    batch_index = torch.arange(batch_size, device=self.device)[:, None]
                    depth_index = torch.arange(self.planning_depth, device=self.device)[
                        None, :
                    ]
                    next_best_dist = next_dist_probs[
                        batch_index, depth_index, next_actions
                    ]

                    tz = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * atom_values
                    )
                    tz = torch.clamp(tz, self.v_min, self.v_max)
                    b = (tz - self.v_min) / atom_delta
                    l, u = b.floor().long(), b.ceil().long()
                    floor_equal_ceil = ((u - l) < 0.5).float()
                    l_project = F.one_hot(l, self.num_atoms).float()
                    u_project = F.one_hot(u, self.num_atoms).float()
                    tl_delta = next_best_dist * (u - b + floor_equal_ceil)[:, None, :]
                    tu_delta = next_best_dist * (b - l)[:, None, :]
                    target_dist[:, 1:, :] += torch.sum(
                        u_project[:, None, :, :] * tu_delta[:, :-1, :, None],
                        dim=2,
                    )
                    target_dist[:, 1:, :] += torch.sum(
                        l_project[:, None, :, :] * tl_delta[:, :-1, :, None],
                        dim=2,
                    )

                current_dist_logits = self.policy.q_net(replay_data.observations)[
                    torch.arange(batch_size), :, replay_data.actions[:, 0].long()
                ]
                unreduced_loss = F.kl_div(
                    F.log_softmax(current_dist_logits, dim=-1),
                    target_dist.detach(),
                    reduction="none",
                ).sum(dim=-1)

            assert unreduced_loss.size() == (batch_size, self.planning_depth)
            for i in range(self.planning_depth):
                losses_by_q_function[i].append(unreduced_loss[:, i].mean().item())

            if self.scale_losses:
                for i in range(self.planning_depth):
                    current_loss = unreduced_loss[:, i].mean().item()
                    if np.isnan(self.loss_running_mean[i]):
                        self.loss_running_mean[i] = current_loss
                    else:
                        self.loss_running_mean[i] = (
                            self.loss_scale_decay * self.loss_running_mean[i]
                            + (1 - self.loss_scale_decay) * current_loss
                        )
                loss = (
                    unreduced_loss
                    / torch.tensor(self.loss_running_mean).to(self.device)
                ).mean()
            else:
                loss = unreduced_loss.mean()
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            if self.ema_alpha > 0:
                self._update_ema_parameters()

            # Update learning rate intra-step
            if not self.purely_exploring:
                lr = base_lr * self.intra_step_lr_decay ** (
                    (gradient_step + 1) / gradient_steps
                )
                update_learning_rate(self.policy.optimizer, lr)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for i in range(self.planning_depth):
            self.logger.record(
                f"train/loss_{i}",
                np.mean(losses_by_q_function[i]),
            )
        self.logger.record("train/loss", np.mean(losses))

    def _get_q_network_state_dict_copy(self) -> Dict[str, np.ndarray]:
        policy: SQIRLPolicy = self.policy
        if self.ema_alpha > 0:
            assert self.ema_policy is not None
            policy = self.ema_policy
        return policy.get_q_network_state_dict_copy()

    def _add_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.state_dicts.append(state_dict)

    def _get_state_dict_for_timestep(
        self, timestep: int
    ) -> Optional[Dict[str, np.ndarray]]:
        if timestep < len(self.state_dicts):
            return self.state_dicts[timestep]
        else:
            return None

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

        unique_timesteps = np.unique(timesteps)

        for t in unique_timesteps:
            mask = timesteps == t

            batch_observations = observation[mask]
            batch_states = (
                None
                if state is None
                else tuple(state_part[mask] for state_part in state)
            )
            batch_episode_starts = (
                None if episode_start is None else episode_start[mask]
            )

            if t < self._current_timestep or deterministic:
                state_dict: Optional[Dict[str, np.ndarray]]
                if t < self._current_timestep:
                    state_dict = self._get_state_dict_for_timestep(t)
                    assert state_dict is not None
                else:
                    state_dict = None
                batch_actions, _ = self.policy.predict(
                    batch_observations,
                    batch_states,
                    batch_episode_starts,
                    deterministic,
                    state_dict=state_dict,
                )
            else:
                batch_actions = np.array(
                    [self.action_space.sample() for _ in range(len(batch_observations))]
                )

            actions[mask] = batch_actions

        return actions, new_states

    def _sample_actions(
        self,
        t: int,
        n_envs: int = 1,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._last_obs is not None
        assert isinstance(self._last_obs, np.ndarray)
        obs = self._last_obs
        if mask is not None:
            obs = obs[mask]
        batch_size = obs.shape[0]
        action, state = self.predict(
            obs, (np.full(batch_size, t),), deterministic=False
        )
        exploring = np.full(batch_size, t >= self._current_timestep)
        return action, exploring

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

    def _store_transitions(
        self,
        episodes: List[Optional[Episode]],
        action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        assert isinstance(self._last_obs, np.ndarray)
        assert isinstance(new_obs_, np.ndarray)

        for episode_index, episode in enumerate(episodes):
            if episode is not None:
                episode.observations.append(self._last_obs[episode_index])
                episode.actions.append(action[episode_index])
                episode.rewards.append(float(reward_[episode_index]))
                episode.dones.append(dones[episode_index])
                episode.infos.append(infos[episode_index])

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for episode_index, (episode, done) in enumerate(zip(episodes, dones)):
            if done and infos[episode_index].get("terminal_observation") is not None:
                next_obs[episode_index] = infos[episode_index]["terminal_observation"]
                # VecNormalize normalizes the terminal observation
                if self._vec_normalize_env is not None:
                    next_obs[episode_index] = self._vec_normalize_env.unnormalize_obs(
                        next_obs[episode_index, :]
                    )
            if done and episode is not None:
                episode.observations.append(next_obs[episode_index])

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _store_episode(
        self, episode: Episode, replay_buffer: ReplayBufferWithValueTargets
    ):
        value_target = 0.0
        value_targets = np.zeros_like(episode.rewards)
        for t in reversed(range(episode.t)):
            value_target = episode.rewards[t] + self.gamma * value_target
            value_targets[t] = value_target

        assert episode.dones[-1]

        # We only save timesteps after self._current_timestep, since we only want to
        # train on timesteps where we acted randomly afterwards.
        for t in range(self._current_timestep, episode.t):
            replay_buffer.add(
                episode.observations[t][None],
                episode.observations[t + 1][None],
                episode.actions[t][None],
                np.array([episode.rewards[t]]),
                episode.dones[t][None],
                [episode.infos[t]],
                np.array([value_targets[t]]),
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        num_episodes: int,
        replay_buffer: ReplayBufferWithValueTargets,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"

        callback.on_rollout_start()
        continue_training = True

        explorings: List[bool] = []

        episodes: List[Optional[Episode]] = [
            Episode.empty() for _ in range(env.num_envs)
        ]

        self._last_obs = cast(np.ndarray, env.reset())

        while num_collected_episodes < num_episodes:
            # Select action randomly or according to policy
            ts = {episode.t for episode in episodes if episode is not None}
            assert len(ts) == 1
            (t,) = ts
            mask = np.array([episode is not None for episode in episodes])
            actions = np.full(env.num_envs, 0, dtype=int)
            actions[mask], exploring = self._sample_actions(t, env.num_envs, mask=mask)
            explorings.extend([exploring[0]] * mask.sum())

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            assert isinstance(new_obs, np.ndarray)

            self.num_timesteps += mask.sum()
            num_collected_steps += mask.sum()

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            for _ in range(mask.sum()):
                if callback.on_step() is False:
                    return RolloutReturn(
                        num_collected_steps,
                        num_collected_episodes,
                        continue_training=False,
                    )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(
                [
                    {} if episode is None else info
                    for episode, info in zip(episodes, infos)
                ],
                np.array(
                    [
                        False if episode is None else done
                        for episode, done in zip(episodes, dones)
                    ]
                ),
            )

            # Store data in episode buffers
            self._store_transitions(episodes, actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            for episode_index, (episode, done) in list(enumerate(zip(episodes, dones))):
                if done and episode is not None:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    # Save episode to replay buffer and start a new one
                    self._store_episode(episode, replay_buffer)
                    episodes[episode_index] = None

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

            if all(episode is None for episode in episodes):
                self._last_obs = cast(np.ndarray, env.reset())
                self._last_original_obs = deepcopy(self._last_obs)
                episodes = [Episode.empty() for _ in range(env.num_envs)]

                # Make sure not to collect more episodes than we need.
                if num_episodes - num_collected_episodes < env.num_envs:
                    for episode_index in range(
                        num_episodes - num_collected_episodes, env.num_envs
                    ):
                        episodes[episode_index] = None
        callback.on_rollout_end()

        exploration_rate = np.mean(explorings)
        self.logger.record("rollout/exploration_rate", exploration_rate)
        if exploration_rate == 0:
            self._iterations_with_no_exploration += 1
            if self._iterations_with_no_exploration > 10:
                print("No exploration for 10 iterations, stopping training")
                continue_training = False
        else:
            self._iterations_with_no_exploration = 0

        return RolloutReturn(
            num_collected_steps,
            num_collected_episodes,
            continue_training,
        )
