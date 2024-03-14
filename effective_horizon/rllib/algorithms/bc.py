import logging
import random
from typing import Dict, List, Optional, Set, Type, cast

import torch
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ResultDict, TensorType
from ray.tune.registry import register_trainable

logger = logging.getLogger(__name__)


class BCConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or BC)

        self.sgd_minibatch_size = 100
        self.num_rollout_workers = 0
        self.validation_prop = 0
        self.entropy_coeff = 0
        del self.exploration_config

    def training(
        self,
        *,
        sgd_minibatch_size=NotProvided,
        validation_prop=NotProvided,
        entropy_coeff=NotProvided,
        **kwargs,
    ) -> "BCConfig":
        super().training(**kwargs)

        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if validation_prop is not NotProvided:
            self.validation_prop = validation_prop
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff

        return self


class BCTorchPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, TorchModelV2)

        episode_ids: Set[int] = set(train_batch[SampleBatch.EPS_ID].tolist())
        episode_in_validation: Dict[int, bool] = {
            episode_id: random.Random(episode_id).random()
            < self.config["validation_prop"]
            for episode_id in episode_ids
        }
        validation_mask = torch.tensor(
            [
                episode_in_validation[episode_id.item()]
                for episode_id in train_batch[SampleBatch.EPS_ID]
            ],
            dtype=torch.bool,
            device=self.device,
        )

        model_out, _ = model(train_batch)
        action_dist: ActionDistribution = dist_class(model_out, model)
        actions = train_batch[SampleBatch.ACTIONS]
        logprobs = action_dist.logp(actions)

        bc_loss = -torch.mean(logprobs[~validation_mask])
        model.tower_stats["bc_loss"] = bc_loss
        model.tower_stats["accuracy"] = (
            (action_dist.deterministic_sample() == actions)[~validation_mask]
            .float()
            .mean()
        )

        entropy = action_dist.entropy().mean()
        model.tower_stats["entropy"] = entropy

        loss = bc_loss - self.config["entropy_coeff"] * entropy

        validation_cross_entropy: Optional[torch.Tensor]
        if torch.any(validation_mask):
            validation_cross_entropy = -logprobs[validation_mask].mean()
            model.tower_stats["validation_cross_entropy"] = validation_cross_entropy
        else:
            validation_cross_entropy = None
            model.tower_stats["validation_cross_entropy"] = torch.zeros(size=(0,))

        return loss

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = {
            "bc_loss": torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats("bc_loss")))
            ),
            "entropy": torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats("entropy")))
            ),
            "accuracy": torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats("accuracy")))
            ),
        }
        if self.get_tower_stats("validation_cross_entropy")[0] is not None:
            stats["validation/cross_entropy"] = torch.mean(
                torch.stack(
                    cast(
                        List[torch.Tensor],
                        self.get_tower_stats("validation_cross_entropy"),
                    )
                )
            )
        return cast(Dict[str, TensorType], convert_to_numpy(stats))


class BC(Algorithm):
    @classmethod
    def get_default_config(cls):
        return BCConfig()

    def get_default_policy_class(self, config) -> Type[Policy]:
        if config["framework"] == "torch":
            return BCTorchPolicy
        else:
            raise NotImplementedError()

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config["count_steps_by"] == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_agent_steps=self.config["train_batch_size"],
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Train
        train_results: ResultDict
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                from_worker = None
                self.workers.sync_weights(
                    from_worker=from_worker,
                    policies=list(train_results.keys()),
                    global_vars=global_vars,
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("BC", BC)
