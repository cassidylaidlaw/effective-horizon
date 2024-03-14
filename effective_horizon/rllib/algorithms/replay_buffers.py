import queue
from queue import Queue
from threading import Thread
from typing import List

import ray
from ray._raylet import ObjectRef
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.replay_buffers import (
    MultiAgentPrioritizedReplayBuffer,
    MultiAgentReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from ray.rllib.utils.typing import SampleBatchType


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


class NoDecompressReplayBuffer(NoDecompressMixin, ReplayBuffer):
    pass


class NoDecompressMultiAgentReplayBuffer(MultiAgentReplayBuffer):
    def __init__(self, *args, **kwargs):
        kwargs["underlying_buffer_config"] = {
            "type": NoDecompressReplayBuffer,
        }
        super().__init__(*args, **kwargs)


@ray.remote
def decompress(sample_batch: SampleBatchType) -> SampleBatchType:
    sample_batch.decompress_if_needed()
    return sample_batch


class ParallelSampleBatchDecompressor(object):
    def __init__(self):
        self.decompress_in_queue: "Queue[SampleBatchType]" = Queue()
        self.decompress_out_queue: "Queue[SampleBatchType]" = Queue()
        decompress_thread = Thread(
            target=self.run_decompress,
            daemon=True,
        )
        decompress_thread.start()

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
                self.decompress_in_queue.task_done()
