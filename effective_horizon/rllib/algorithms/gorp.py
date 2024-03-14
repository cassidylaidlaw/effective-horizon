import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune.registry import register_trainable

logger = logging.getLogger(__name__)


class GORPConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a PGConfig instance."""
        super().__init__(algo_class=algo_class or GORPAlgorithm)

        self.use_max = False
        self.action_seq_len = 1
        self.episodes_per_action_seq = 1
        self.num_sgd_iter = 1
        self.sgd_minibatch_size = 0

    def training(
        self,
        *,
        use_max=NotProvided,
        action_seq_len=NotProvided,
        episodes_per_action_seq=NotProvided,
        **kwargs,
    ) -> "GORPConfig":
        super().training(**kwargs)

        if use_max is not NotProvided:
            self.use_max = use_max
        if action_seq_len is not NotProvided:
            self.action_seq_len = action_seq_len
        if episodes_per_action_seq is not NotProvided:
            self.episodes_per_action_seq = episodes_per_action_seq

        self.num_sgd_iter = 1
        self.sgd_minibatch_size = 0

        return self


class GORPPolicy(Policy):
    actions: List[int]

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.actions = []

    def get_initial_state(self) -> List[TensorType]:
        return [np.zeros(1)]

    def is_recurrent(self) -> bool:
        return True

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        prev_reward_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[Any]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        if explore is None:
            explore = self.config["explore"]
        assert state_batches is not None
        actions = []
        for state in state_batches[0]:
            t = int(state[0])
            if t < len(self.actions):
                actions.append(self.actions[t])
            else:
                if explore:
                    actions.append(self.action_space.sample())
                else:
                    actions.append(0)
        return np.array(actions), [state_batches[0] + 1], {}

    def learn_on_batch(self, samples: SampleBatch):
        t = len(self.actions)
        at_timestep = samples[SampleBatch.T] == t
        actions_at_timestep = samples[SampleBatch.ACTIONS][at_timestep]
        returns_at_timestep = samples[Postprocessing.VALUE_TARGETS][at_timestep]
        assert isinstance(self.action_space, spaces.Discrete)

        optimal_return = float("-inf")
        optimal_action: Optional[int] = None
        for action in range(self.action_space.n):
            action_returns = returns_at_timestep[actions_at_timestep == action]
            action_return = (
                np.max(action_returns)
                if self.config["use_max"]
                else np.mean(action_returns)
            )
            reduction = "max" if self.config["use_max"] else "mean"
            logger.info(
                f"action {action}: {reduction} return = {action_return:.3f} ("
                + ", ".join(f"{action_return:.3f}" for action_return in action_returns)
                + ")"
            )
            if action_return > optimal_return:
                optimal_action = action
                optimal_return = action_return
        if optimal_action is None:
            optimal_action = 0
        self.actions.append(optimal_action)
        logger.info(self.actions)
        return {}  # return stats

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[str, Tuple[Policy, SampleBatch]]] = None,
        episode=None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        rewards_plus_v = np.concatenate(
            [sample_batch[SampleBatch.REWARDS], np.array([0])]
        )
        sample_batch[Postprocessing.VALUE_TARGETS] = discount_cumsum(
            rewards_plus_v, self.config["gamma"]
        )[:-1].astype(np.float32)
        return sample_batch

    def get_weights(self):
        return {"actions": self.actions}

    def set_weights(self, weights):
        self.actions = weights["actions"]


class GORPAlgorithm(Algorithm):
    config: GORPConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls):
        return GORPConfig()

    def get_default_policy_class(self, config):
        return GORPPolicy

    def training_step(self):
        # Custom sampling logic.
        assert self.workers is not None
        policy = self.workers.local_worker().get_policy()
        num_actions = policy.action_space.n
        episodes_to_sample: List[Sequence[int]] = []
        action_seq: Sequence[int]
        for action_seq in itertools.product(
            *[range(num_actions)] * self.config.action_seq_len
        ):
            episodes_to_sample.extend(
                [action_seq] * self.config.episodes_per_action_seq
            )

        actions = policy.get_weights()["actions"]
        num_workers = len(self.workers.remote_workers())
        assert self.config.batch_mode == "complete_episodes"
        train_batches: List[SampleBatch] = []
        while len(episodes_to_sample) > 0:
            next_episodes = episodes_to_sample[:num_workers]
            episodes_to_sample = episodes_to_sample[num_workers:]

            set_weights_refs = []
            for worker, action_seq in zip(self.workers.remote_workers(), next_episodes):
                set_weights_ref = worker.set_weights.remote(
                    {
                        DEFAULT_POLICY_ID: {
                            "actions": np.concatenate([actions, action_seq]).astype(int)
                        }
                    }
                )
                set_weights_refs.append(set_weights_ref)
            ray.get(set_weights_refs)

            sample_object_refs = []
            for worker, action_seq in zip(self.workers.remote_workers(), next_episodes):
                sample_object_refs.append(worker.sample.remote())
            train_batches.extend(ray.get(sample_object_refs))
        train_batch = SampleBatch.concat_samples(train_batches)

        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        train_results = train_one_step(
            self,
            train_batch,
        )

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("GORP", GORPAlgorithm)
