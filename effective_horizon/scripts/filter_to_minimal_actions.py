import glob
import json
import os
from logging import Logger
from typing import List

from ale_py.env.gym import AtariEnv
from ray.rllib.evaluation import SampleBatch
from ray.rllib.offline.json_reader import from_json_data
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import concat_samples
from sacred import Experiment

ex = Experiment("filter_to_minimal_actions")


@ex.config
def sacred_config(_log):
    in_dir = None  # noqa: F841
    out_dir = None  # noqa: F841
    rom = None  # noqa: F841


@ex.automain
def main(
    in_dir: str,
    out_dir: str,
    rom: str,
    _log: Logger,
):
    env = AtariEnv(game=rom)
    actions = env.ale.getMinimalActionSet()

    writer = JsonWriter(out_dir)
    for in_fname in glob.glob(os.path.join(in_dir, "*.json")):
        _log.info(f"reading {in_fname}...")
        with open(in_fname, "r") as in_file:
            for in_line in in_file:
                if in_line.strip() == "":
                    continue
                in_json = json.loads(in_line)
                in_batch = from_json_data(in_json, worker=None)
                out_batch_timesteps: List[SampleBatch] = []
                for t in range(len(in_batch)):
                    in_action = in_batch[SampleBatch.ACTIONS][t]
                    if in_action in actions:
                        out_action = actions.index(in_action)
                        timestep_batch = in_batch[t : t + 1]
                        timestep_batch[SampleBatch.ACTIONS][0] = out_action
                        out_batch_timesteps.append(timestep_batch)
                if len(out_batch_timesteps) > 0:
                    writer.write(concat_samples(out_batch_timesteps))
