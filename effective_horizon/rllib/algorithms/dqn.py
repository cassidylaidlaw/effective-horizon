import logging
from queue import Queue
from threading import Thread

from ray.rllib.algorithms.dqn.dqn import DQN, calculate_rr_weights
from ray.rllib.execution.common import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict, SampleBatchType
from ray.tune.registry import register_trainable

from .replay_buffers import ParallelSampleBatchDecompressor

logger = logging.getLogger(__name__)


class FastDQN(DQN):
    def __init__(self, config, *args, **kwargs):
        DQN.__init__(self, config, *args, **kwargs)

        self.decompressor = ParallelSampleBatchDecompressor()
        self.decompressor_initialized = False

    def async_sample(self, queue: "Queue[SampleBatchType]", store_weight: int):
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

        sample_queue: "Queue[SampleBatchType]" = Queue()
        sample_thread = Thread(
            target=self.async_sample,
            args=(sample_queue, store_weight),
            daemon=True,
        )
        sample_thread.start()

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            (
                NUM_AGENT_STEPS_SAMPLED
                if self.config["count_steps_by"] == "agent_steps"
                else NUM_ENV_STEPS_SAMPLED
            )
        ]

        if cur_ts > self.config["num_steps_sampled_before_learning_starts"]:
            if not self.decompressor_initialized:
                logger.info("putting initial batches into the decompress queue")
                batches_to_add_to_queue = 4 * int(sample_and_train_weight)
                self.decompressor_initialized = True
            else:
                batches_to_add_to_queue = 1 * int(sample_and_train_weight)

            # Sample training batch (MultiAgentBatch) from replay buffer.
            for _ in range(batches_to_add_to_queue):
                compressed_batch = self.local_replay_buffer.sample(
                    self.config["train_batch_size"]
                )
                self.decompressor.decompress_in_queue.put(compressed_batch)

            for _ in range(int(sample_and_train_weight)):
                train_batch = self.decompressor.decompress_out_queue.get()

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                train_results = train_one_step(self, train_batch)

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
