import csv
import glob
import logging
import os
import tarfile
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, TypedDict, cast

import imageio
import numpy as np
import tqdm
from ray.rllib.evaluation import SampleBatch, SampleBatchBuilder
from ray.rllib.utils.images import resize, rgb2gray

logger = logging.getLogger(__name__)


class AtariHeadDataRow(TypedDict):
    frame_id: str
    episode_id: str
    score: str
    unclipped_reward: str
    action: str


class AtariHeadDataset(object):
    recent_frames: Deque[np.ndarray]

    def __init__(self, data_dir: str, frameskip: int, image_size=84, framestack=4):
        self.data_dir = data_dir
        self.frameskip = frameskip
        self.image_size = image_size
        self.framestack = framestack

    def get_human_samples(self) -> Iterable[SampleBatch]:
        self.builder = SampleBatchBuilder()
        data_fnames = glob.glob(os.path.join(self.data_dir, "*.txt"))
        for data_fname in data_fnames:
            logger.info(f"processing {data_fname}...")
            self.trial_name = os.path.basename(data_fname)[:-4]
            self.trial_id = int(self.trial_name.split("_")[0])
            frame_data_fname = data_fname[:-4] + ".tar.bz2"
            self._reset_episode()
            self.rows_for_this_timestep: List[AtariHeadDataRow] = []
            with open(data_fname, "r") as data_file, tarfile.open(
                frame_data_fname, "r"
            ) as self.frame_data_file:
                self.frame_data_tar_members = {
                    info.name: info for info in self.frame_data_file.getmembers()
                }
                data_csv = csv.DictReader(data_file)
                for row in cast(Sequence[AtariHeadDataRow], tqdm.tqdm(data_csv)):
                    if row["episode_id"] == "null":
                        continue
                    if (
                        self.rows_for_this_timestep
                        and row["episode_id"]
                        != self.rows_for_this_timestep[0]["episode_id"]
                    ):
                        self._record_timestep()
                        yield self.builder.build_and_reset()
                        self._reset_episode()
                    self.rows_for_this_timestep.append(row)
                    if len(self.rows_for_this_timestep) >= self.frameskip:
                        self._record_timestep()
                if self.rows_for_this_timestep:
                    self._record_timestep()
            yield self.builder.build_and_reset()

    def _reset_episode(self):
        self.recent_frames = deque([], maxlen=self.framestack)

    def _add_frame(self, frame_id):
        frame_file = self.frame_data_file.extractfile(
            self.frame_data_tar_members[f"{self.trial_name}/{frame_id}.png"]
        )
        assert frame_file is not None
        with frame_file as frame_file:
            frame = imageio.imread(frame_file.read())
        frame = rgb2gray(frame)[:, :, None]
        frame = resize(frame, height=84, width=84)
        self.recent_frames.append(frame)
        while len(self.recent_frames) < self.framestack:
            self.recent_frames.append(frame)

    def _get_obs(self):
        return np.concatenate(self.recent_frames, axis=2)

    def _record_timestep(self):
        self._add_frame(self.rows_for_this_timestep[0]["frame_id"])
        # actions = Counter(int(row["action"]) for row in self.rows_for_this_timestep)
        # ((most_common_action, _),) = actions.most_common(1)
        reward = 0
        episode_id: Optional[int] = None
        for row in self.rows_for_this_timestep:
            if row["unclipped_reward"] != "null":
                reward += int(row["unclipped_reward"])
            if row["episode_id"] != "null":
                episode_id = int(row["episode_id"])
        assert episode_id is not None
        for row in self.rows_for_this_timestep:
            self.builder.add_values(
                **{
                    SampleBatch.OBS: self._get_obs(),
                    SampleBatch.ACTIONS: int(row["action"]),
                    SampleBatch.REWARDS: reward,
                    SampleBatch.EPS_ID: self.trial_id * 1000 + episode_id,
                }
            )
        self.rows_for_this_timestep.clear()
