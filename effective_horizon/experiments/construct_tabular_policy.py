from logging import Logger
from queue import Queue
from threading import Thread
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import torch
import tqdm
from ray.rllib.evaluation import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from sacred import Experiment
from torch import nn

from ..mdp_utils import load_mdp
from ..training_utils import load_trainer

ex = Experiment("construct_tabular_policy")


@ex.config
def sacred_config(_log):
    run = "BC"  # noqa: F841
    checkpoint = ""  # noqa: F841
    mdp = ""  # noqa: F841
    horizon = 0  # noqa: F841
    out = ""  # noqa: F841
    batch_size = 500  # noqa: F841
    num_threads = 10  # noqa: F841


class AsynchronousScreenLoader(object):
    queue: Queue[Tuple[torch.Tensor, torch.Tensor]]
    screen_ids_with_state: torch.Tensor

    def __init__(
        self,
        screens: torch.Tensor,
        batch_size: int,
        device: torch.device,
        screen_mapping: np.ndarray,
        queue_size: Optional[int] = None,
        num_threads: int = 10,
    ):
        self.screens = screens
        num_states = screen_mapping.shape[0]
        if len(screen_mapping.shape) == 1:
            screen_mapping = screen_mapping[:, None]
        screen_ids_with_timestep = np.empty_like(
            screen_mapping, shape=(num_states, 1 + screen_mapping.shape[1])
        )
        screen_ids_with_timestep[:, 0] = np.arange(num_states)
        screen_ids_with_timestep[:, 1:] = screen_mapping
        self.screen_ids_with_timestep = torch.from_numpy(screen_ids_with_timestep)

        self.count = self.screen_ids_with_timestep.shape[0]
        self.batch_size = batch_size
        self.num_batches = (self.count + self.batch_size - 1) // self.batch_size

        self.device = device
        if queue_size is None:
            queue_size = num_threads * 10
        self.queue = Queue(maxsize=queue_size)
        self.num_threads = num_threads

    def load_loop(self, thread_index: int) -> None:
        for batch_index in range(thread_index, self.num_batches, self.num_threads):
            batch_slice = slice(
                batch_index * self.batch_size, (batch_index + 1) * self.batch_size
            )
            self.queue.put(self.load_batch(batch_slice))

    def load_batch(self, batch_slice: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = self.screen_ids_with_timestep[batch_slice]
        states = rows[:, 0]
        screen_ids = rows[:, 1:]

        if screen_ids.shape[1] == 1:
            screen_ids = screen_ids[:, 0]
            obs: torch.Tensor = self.screens[screen_ids]
        else:
            obs_flat = self.screens[screen_ids.reshape(-1)]
            obs = obs_flat.reshape(*screen_ids.size(), *obs_flat.size()[1:]).permute(
                0, 2, 3, 1
            )
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255

        assert obs.min() >= 0
        assert 1 / 255 < obs.max() <= 1
        assert obs.dtype == torch.float

        states.pin_memory()
        obs.pin_memory()

        return (
            states.to(self.device, non_blocking=True),
            obs.to(self.device, non_blocking=True),
        )

    def __iter__(self) -> "AsynchronousScreenLoader":
        self.threads: List[Thread] = []
        for thread_index in range(self.num_threads):
            thread = Thread(target=self.load_loop, args=(thread_index,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # If we've reached the number of batches to return
        # or the queue is empty and the threads are dead.
        done = (
            not any(thread.is_alive() for thread in self.threads) and self.queue.empty()
        )
        if done:
            self.queue.join()
            for thread in self.threads:
                thread.join()
            raise StopIteration
        # Otherwise return the next batch.
        out = self.queue.get()
        self.queue.task_done()
        return out

    def __len__(self) -> int:
        return self.num_batches


@ex.automain
def main(
    run: str,
    checkpoint: str,
    mdp: str,
    horizon: int,
    batch_size: int,
    num_threads: int,
    out: str,
    _log: Logger,
):
    # Load policy.
    trainer = load_trainer(
        checkpoint,
        run,
        {
            "input": "sampler",
            "num_workers": 0,
            "evaluation_num_workers": 0,
        },
    )
    policy: Any = trainer.get_policy()
    model = cast(TorchModelV2, policy.model)
    device = torch.device("cuda")
    cast(nn.Module, model).to(device).eval()

    mdp_arrays = np.load(mdp)
    screen_mapping = mdp_arrays["screen_mapping"]
    index_dtype = screen_mapping.dtype
    transitions, rewards = load_mdp(mdp)
    num_states, num_actions = transitions.shape
    _log.info("loading screens")
    screens = torch.from_numpy(mdp_arrays["screens"])
    # Set the done state screen to an arbitrary one.
    screen_mapping = np.concatenate(
        [screen_mapping, np.zeros((1,) + screen_mapping.shape[1:], dtype=index_dtype)]
    )

    loader = AsynchronousScreenLoader(
        screens,
        screen_mapping=screen_mapping,
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
    )

    _log.info("calculating policy")
    tabular_policy = torch.full((horizon, num_states, num_actions), np.nan)
    for states, obs in tqdm.tqdm(loader):
        action_dist_inputs: torch.Tensor
        action_dist_inputs, _ = model({SampleBatch.OBS: obs})
        action_dist = action_dist_inputs.softmax(dim=-1)
        tabular_policy[:, states, :] = action_dist.detach().cpu()[None, :, :]

    tabular_policy_numpy = tabular_policy.cpu().numpy()
    _log.info(f"saving tabular policy to {out}...")
    np.save(out, tabular_policy_numpy)
