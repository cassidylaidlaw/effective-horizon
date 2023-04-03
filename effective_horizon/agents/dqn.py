import logging
import queue
from queue import Queue
from threading import Thread
from typing import List

import ray
from ray._raylet import ObjectRef
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig, calculate_rr_weights
from ray.rllib.evaluation import MultiAgentBatch
from ray.rllib.execution.common import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
    MultiAgentPrioritizedReplayBuffer,
)
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict, SampleBatchType
from ray.tune.registry import register_trainable

logger = logging.getLogger(__name__)


@ray.remote
def decompress(sample_batch: SampleBatchType) -> SampleBatchType:
    sample_batch.decompress_if_needed()
    return sample_batch


class NoDecompressMixin(ReplayBuffer):
    def _encode_sample(self, idxes: List[int]):
        samples = []
        for i in idxes:
            self._hit_count[i] += 1
            samples.append(self._storage[i])

        if samples:
            # We assume all samples are of same type
            sample_type = type(samples[0])
            out = sample_type.concat_samples(samples)
        else:
            out = SampleBatch()

        return out


class NoDecompressPrioritizedReplayBuffer(NoDecompressMixin, PrioritizedReplayBuffer):
    pass


class NoDecompressMultiAgentPrioritizedReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    def __init__(self, *args, **kwargs):
        kwargs["underlying_buffer_config"] = {
            "type": NoDecompressPrioritizedReplayBuffer,
            "alpha": kwargs.get("prioritized_replay_alpha", 0.6),
            "beta": kwargs.get("prioritized_replay_beta", 0.4),
        }
        super().__init__(*args, **kwargs)


class FastDQN(DQN):
    def __init__(self, config, *args, **kwargs):
        DQN.__init__(self, DQNConfig.from_dict(config), *args, **kwargs)

        self.decompress_in_queue: Queue[SampleBatchType] = Queue()
        self.decompress_out_queue: Queue[SampleBatchType] = Queue()
        decompress_thread = Thread(
            target=self.run_decompress,
            daemon=True,
        )
        decompress_thread.start()
        self.decompresser_initialized = False

    def async_sample(self, queue: Queue[SampleBatchType], store_weight: int):
        for _ in range(int(store_weight)):
            # Sample (MultiAgentBatch) from workers.
            new_sample_batch = synchronous_parallel_sample(
                worker_set=self.workers, concat=True
            )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            queue.put(new_sample_batch)

    def run_decompress(self):
        batch_refs: List[ObjectRef[SampleBatchType]] = []
        while True:
            while True:
                try:
                    compressed_batch = self.decompress_in_queue.get(
                        block=len(batch_refs) == 0
                    )
                    batch_refs.append(decompress.remote(compressed_batch))
                except queue.Empty:
                    break
            ready_refs, batch_refs = ray.wait(batch_refs, num_returns=1, timeout=0.1)
            for batch_ref in ready_refs:
                self.decompress_out_queue.put(ray.get(batch_ref))

    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """

        train_results: dict = {}

        assert self.local_replay_buffer is not None

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        sample_queue: Queue[SampleBatchType] = Queue()
        sample_thread = Thread(
            target=self.async_sample,
            args=(sample_queue, store_weight),
            daemon=True,
        )
        sample_thread.start()

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED
            if self.config["count_steps_by"] == "agent_steps"
            else NUM_ENV_STEPS_SAMPLED
        ]

        if cur_ts > self.config["num_steps_sampled_before_learning_starts"]:
            if not self.decompresser_initialized:
                logger.info("putting initial batches into the decompress queue")
                batches_to_add_to_queue = 4 * int(sample_and_train_weight)
                self.decompresser_initialized = True
            else:
                batches_to_add_to_queue = 1 * int(sample_and_train_weight)

            # Sample training batch (MultiAgentBatch) from replay buffer.
            for _ in range(batches_to_add_to_queue):
                compressed_batch = self.local_replay_buffer.sample(
                    self.config["train_batch_size"]
                )
                self.decompress_in_queue.put(compressed_batch)

            for _ in range(int(sample_and_train_weight)):
                train_batch = self.decompress_out_queue.get()

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_batch_size = (
                        len(train_batch) // self.config["sgd_minibatch_size"]
                    ) * self.config["sgd_minibatch_size"]
                    assert isinstance(train_batch, MultiAgentBatch)
                    for policy_id in train_batch.policy_batches:
                        train_batch.policy_batches[
                            policy_id
                        ] = train_batch.policy_batches[policy_id].slice(
                            0, train_batch_size
                        )
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config["target_network_update_freq"]:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid: pid in to_update and p.update_target()
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        sample_thread.join()
        while not sample_queue.empty():
            new_sample_batch = sample_queue.get()
            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        # Update weights and global_vars - after learning on the local worker -
        # on all remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results


register_trainable("FastDQN", FastDQN)
