import zipfile
from logging import Logger
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from numpy.lib import format as npy_format
from sacred import Experiment

from effective_horizon.image_utils import resize, rgb2gray

ex = Experiment("convert_atari_mdp_to_framestack")


@ex.config
def sacred_config(_log):
    mdp = ""
    horizon = 0  # noqa: F841
    out = f"{mdp[:-4]}_framestack.npz"  # noqa: F841
    num_threads = 10  # noqa: F841


def preprocess_screens_thread(
    screens: np.ndarray,
    queue: "Queue[Tuple[slice, np.ndarray]]",
    done_event: Event,
    scaled_size: int,
):
    while not done_event.is_set():
        try:
            screens_slice, screens_batch = queue.get(timeout=0.01)
        except Empty:
            continue

        batch_size = screens_batch.shape[0]
        resized_batch = np.empty((batch_size, scaled_size, scaled_size))

        for screen_index in range(batch_size):
            screen = screens_batch[screen_index]
            screen = rgb2gray(screen)
            screen = resize(screen, scaled_size, scaled_size)
            resized_batch[screen_index] = screen

        screens[screens_slice] = torch.from_numpy(resized_batch)  # * 255).byte()
        queue.task_done()


def load_grayscale_screens(
    mdp_path: str, batch_size=1000, scaled_size=84, num_threads: int = 10
) -> torch.Tensor:
    with zipfile.ZipFile(mdp_path, mode="r") as mdp_zip:
        with mdp_zip.open("screens.npy", mode="r") as screens_file:
            version = npy_format.read_magic(screens_file)

            read_array_header = getattr(
                npy_format,
                "_read_array_header",
                getattr(npy_format, "read_array_header_1_0", None),
            )
            assert read_array_header is not None

            shape, fortran_order, dtype = read_array_header(screens_file, version)
            assert fortran_order is False

            num_screens, height, width, _ = shape
            screens = torch.empty(
                (num_screens, scaled_size, scaled_size), dtype=torch.uint8
            )

            done_event = Event()
            queue: "Queue[Tuple[slice, np.ndarray]]" = Queue(maxsize=num_threads * 10)
            threads: List[Thread] = []
            for thread_index in range(num_threads):
                thread = Thread(
                    target=preprocess_screens_thread,
                    args=(
                        screens,
                        queue,
                        done_event,
                        scaled_size,
                    ),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)

            # Work in batches.
            batch_start_index = 0
            remaining_screens = num_screens
            for _ in tqdm.trange((remaining_screens + batch_size - 1) // batch_size):
                num_screens_to_read = min(batch_size, remaining_screens)
                screens_batch = np.frombuffer(
                    screens_file.read(num_screens_to_read * width * height * 3),
                    dtype=dtype,
                )
                screens_batch = screens_batch.reshape(
                    (num_screens_to_read, height, width, 3)
                )
                screens_slice = slice(
                    batch_start_index, batch_start_index + num_screens_to_read
                )
                queue.put((screens_slice, screens_batch))
                remaining_screens -= num_screens_to_read
                batch_start_index += num_screens_to_read
            assert remaining_screens == 0

            queue.join()
            done_event.set()
            for thread in threads:
                thread.join()

    return screens


@ex.automain
def main(
    mdp: str,
    horizon: int,
    num_threads: int,
    out: str,
    _log: Logger,
):
    mdp_arrays = np.load(mdp)
    transitions = mdp_arrays["transitions"]
    num_states, num_actions = transitions.shape
    rewards = mdp_arrays["rewards"]
    index_dtype = transitions.dtype
    screen_mapping = mdp_arrays["screen_mapping"]

    _log.info("loading screens and converting to grayscale")
    grayscale_screens = load_grayscale_screens(mdp, num_threads=num_threads)

    _log.info("calculating screens in framestack")
    states_with_screen_ids = np.empty((1, 5), dtype=index_dtype)
    states_with_screen_ids[0, 0] = 0
    states_with_screen_ids[0, 1:] = screen_mapping[0]

    for timestep in tqdm.trange(1, horizon):
        prev_num_states = states_with_screen_ids.shape[0]
        state_actions_with_screen_ids = np.repeat(
            states_with_screen_ids, num_actions, axis=0
        )
        state_actions_with_screen_ids[:, 0] = state_actions_with_screen_ids[
            :, 0
        ] * num_actions + np.tile(
            np.arange(num_actions, dtype=index_dtype),
            prev_num_states,
        )
        next_states = transitions.flat[state_actions_with_screen_ids[:, 0]]
        next_screen_ids = screen_mapping[next_states]
        next_terminal = next_states == -1

        next_states_with_screen_ids = np.empty(
            (prev_num_states * num_actions, 5), dtype=index_dtype
        )
        next_states_with_screen_ids[:, 0] = next_states
        next_states_with_screen_ids[:, 1:4] = state_actions_with_screen_ids[:, 2:5]
        next_states_with_screen_ids[:, 4] = next_screen_ids
        next_states_with_screen_ids[next_terminal, :] = states_with_screen_ids[:1]

        combined_states_with_screen_ids = np.concatenate(
            [states_with_screen_ids, next_states_with_screen_ids],
            axis=0,
        )
        unique_states_with_screen_ids, unique_inverse = np.unique(
            combined_states_with_screen_ids, axis=0, return_inverse=True
        )
        new_num_states = unique_states_with_screen_ids.shape[0]
        new_prev_state_ids = unique_inverse[:prev_num_states]

        assert np.all(
            unique_inverse[prev_num_states:][next_terminal] == new_prev_state_ids[0]
        )

        if timestep == horizon - 1:
            new_transitions = np.full(
                (new_num_states, num_actions), -1, dtype=index_dtype
            )
            new_transitions_for_prev_states = unique_inverse[prev_num_states:]
            new_transitions_for_prev_states[next_terminal] = -1
            new_transitions[new_prev_state_ids] = (
                new_transitions_for_prev_states.reshape((prev_num_states, num_actions))
            )

        states_with_screen_ids = unique_states_with_screen_ids

    new_rewards = rewards[states_with_screen_ids[:, 0]]

    _log.info(f"saving to {out}")
    np.savez_compressed(
        out,
        transitions=new_transitions,
        rewards=new_rewards,
        screens=grayscale_screens,
        screen_mapping=states_with_screen_ids[:, 1:],
        prev_state_mapping=states_with_screen_ids[:, 0],
    )
