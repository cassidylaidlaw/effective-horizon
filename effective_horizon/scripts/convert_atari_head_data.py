import base64
import json
from io import BytesIO
from logging import Logger
from typing import List, Literal, Optional

import numpy as np
from ale_py.env.gym import AtariEnv
from sacred import Experiment

from effective_horizon.atari_head import AtariHeadDataset

ex = Experiment("convert_atari_head_data")


@ex.config
def sacred_config(_log):
    data_dir = None  # noqa: F841
    frameskip: int = 30  # noqa: F841
    out_path = None  # noqa: F841
    rom = None  # noqa: F841
    minimal_actions = False  # noqa: F841
    out_format: Literal["rllib", "json"] = "json"  # noqa: F841


@ex.automain
def main(
    data_dir: str,
    frameskip: int,
    out_path: str,
    rom: str,
    minimal_actions: bool,
    out_format: Literal["rllib", "dict"],
    _log: Logger,
):
    action_set: Optional[List[int]] = None
    if minimal_actions:
        env = AtariEnv(game=rom)
        action_set = [int(action) for action in env.ale.getMinimalActionSet()]

    dataset = AtariHeadDataset(data_dir, frameskip=frameskip, action_set=action_set)
    if out_format == "rllib":
        from ray.rllib.evaluation import SampleBatch
        from ray.rllib.offline.json_writer import JsonWriter

        writer = JsonWriter(out_path)
        for batch in dataset.get_human_samples():
            writer.write(SampleBatch(**batch))
    elif out_format == "json":
        with open(out_path, "w") as out_file:
            for episode_dict in dataset.get_human_samples():
                with BytesIO() as obs_bytes:
                    np.save(obs_bytes, episode_dict["obs"])
                    obs_base64 = base64.b64encode(obs_bytes.getvalue()).decode("utf-8")

                json_dict = {
                    "obs": obs_base64,
                    "actions": episode_dict["actions"].tolist(),
                    "rewards": episode_dict["rewards"].tolist(),
                    "episode_ids": episode_dict["eps_id"].tolist(),
                }
                out_file.write(json.dumps(json_dict) + "\n")
