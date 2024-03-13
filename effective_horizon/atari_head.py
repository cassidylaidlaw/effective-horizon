import csv
import glob
import logging
import os
import tarfile
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, TypedDict, cast

import imageio
import numpy as np
import tqdm

from .image_utils import resize, rgb2gray

logger = logging.getLogger(__name__)


class AtariHeadDataRow(TypedDict):
    frame_id: str
    episode_id: str
    score: str
    unclipped_reward: str
    action: str


class Episode(TypedDict):
    obs: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    episode_ids: List[int]


def empty_episode() -> Episode:
    return Episode(obs=[], actions=[], rewards=[], episode_ids=[])


class AtariHeadDataset(object):
    recent_frames: Deque[np.ndarray]

    def __init__(
        self,
        data_dir: str,
        frameskip: int,
        image_size=84,
        framestack=4,
        action_set: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.frameskip = frameskip
        self.image_size = image_size
        self.framestack = framestack
        self.action_set = action_set

    def get_human_samples(self) -> Iterable[Dict[str, np.ndarray]]:
        data_fnames = glob.glob(os.path.join(self.data_dir, "*.txt"))
        for data_fname in data_fnames:
            logger.info(f"processing {data_fname}...")
            self.trial_name = os.path.basename(data_fname)[:-4]
            self.trial_id = int(self.trial_name.split("_")[0])
            frame_data_fname = data_fname[:-4] + ".tar.bz2"
            self._reset_episode()
            self.rows_for_this_timestep: List[AtariHeadDataRow] = []
            with (
                open(data_fname, "r") as data_file,
                tarfile.open(frame_data_fname, "r") as self.frame_data_file,
            ):
                self.frame_data_tar_members = {
                    info.name: info for info in self.frame_data_file.getmembers()
                }
                data_csv = csv.DictReader(data_file)
                for row in cast(Sequence[AtariHeadDataRow], tqdm.tqdm(data_csv)):
                    if row["action"] == "null":
                        continue
                    if (
                        self.rows_for_this_timestep
                        and row["episode_id"]
                        != self.rows_for_this_timestep[0]["episode_id"]
                    ):
                        self._record_timestep()
                        yield self._get_episode_dict()
                        self._reset_episode()
                    # elif np.random.random() < 0.01:
                    #     yield self._get_episode_dict()
                    #     self._reset_episode()
                    #     break
                    self.rows_for_this_timestep.append(row)
                    if len(self.rows_for_this_timestep) >= self.frameskip:
                        self._record_timestep()
                if self.rows_for_this_timestep:
                    self._record_timestep()
            yield self._get_episode_dict()

    def _reset_episode(self):
        self.recent_frames = deque([], maxlen=self.framestack)
        self.episode = empty_episode()

    def _add_frame(self, frame_id):
        frame_file = self.frame_data_file.extractfile(
            self.frame_data_tar_members[f"{self.trial_name}/{frame_id}.png"]
        )
        assert frame_file is not None
        with frame_file as frame_file:
            frame = imageio.imread(frame_file.read())
        frame = rgb2gray(frame)[:, :, None]
        frame = resize(frame, height=self.image_size, width=self.image_size)
        self.recent_frames.append(frame)
        while len(self.recent_frames) < self.framestack:
            self.recent_frames.append(frame)

    def _get_obs(self):
        if self.recent_frames[0].ndim == 2:
            return np.stack(self.recent_frames, axis=2)
        else:
            return np.concatenate(self.recent_frames, axis=2)

    def _record_timestep(self):
        self._add_frame(self.rows_for_this_timestep[0]["frame_id"])
        reward = 0
        episode_id: int = 0
        for row in self.rows_for_this_timestep:
            if row["unclipped_reward"] != "null":
                reward += int(row["unclipped_reward"])
            if row["episode_id"] != "null":
                episode_id = int(row["episode_id"])

        for row in self.rows_for_this_timestep:
            action = int(row["action"])
            if self.action_set is not None:
                try:
                    action = self.action_set.index(action)
                except ValueError:
                    continue

            self.episode["obs"].append(self._get_obs())
            self.episode["actions"].append(action)
            self.episode["rewards"].append(reward)
            self.episode["episode_ids"].append(self.trial_id * 1000 + episode_id)
        self.rows_for_this_timestep.clear()

    def _get_episode_dict(self) -> Dict[str, np.ndarray]:
        if len(self.episode["obs"]) == 0:
            obs = np.zeros((0, 84, 84, 4))
        else:
            obs = np.stack(self.episode["obs"], axis=0)
        return {
            "obs": obs,
            "actions": np.array(self.episode["actions"]),
            "rewards": np.array(self.episode["rewards"]),
            "eps_id": np.array(self.episode["episode_ids"]),
        }
