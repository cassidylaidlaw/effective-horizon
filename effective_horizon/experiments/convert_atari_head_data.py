from logging import Logger

from ray.rllib.offline.json_writer import JsonWriter
from sacred import Experiment

from ..atari_head import AtariHeadDataset

ex = Experiment("convert_atari_head_data")


@ex.config
def sacred_config(_log):
    data_dir = None  # noqa: F841
    frameskip: int = 30  # noqa: F841
    out_dir = None  # noqa: F841


@ex.automain
def main(
    data_dir: str,
    frameskip: int,
    out_dir: str,
    _log: Logger,
):
    dataset = AtariHeadDataset(data_dir, frameskip=frameskip)
    writer = JsonWriter(out_dir)
    for batch in dataset.get_human_samples():
        writer.write(batch)
