import os

import numpy as np
import pytest

from effective_horizon.scripts.construct_tabular_policy import (
    ex as construct_tabular_policy_ex,
)
from effective_horizon.scripts.convert_atari_head_data import (
    ex as convert_atari_head_data_ex,
)
from effective_horizon.scripts.convert_atari_mdp_to_framestack import (
    ex as convert_atari_mdp_to_framestack_ex,
)


def test_convert_atari_head_data_one_step(tmp_path):
    convert_atari_head_data_ex.run(
        config_updates={
            "data_dir": "data/atari-head/freeway",
            "out_path": str(tmp_path / "json"),
            "out_format": "json",
            "rom": "freeway",
            "minimal_actions": True,
        }
    )


@pytest.mark.uses_rllib
def test_convert_atari_head_data_two_steps(tmp_path):
    from effective_horizon.scripts.filter_to_minimal_actions import (
        ex as filter_to_minimal_actions_ex,
    )

    convert_atari_head_data_ex.run(
        config_updates={
            "data_dir": "data/atari-head/freeway",
            "out_path": str(tmp_path / "rllib_complete"),
            "out_format": "rllib",
        }
    )
    filter_to_minimal_actions_ex.run(
        config_updates={
            "in_dir": str(tmp_path / "rllib_complete"),
            "out_dir": str(tmp_path / "rllib_complete_minimal_actions"),
            "rom": "freeway",
        }
    )


def test_convert_atari_mdp_to_framestack(tmp_path):
    convert_atari_mdp_to_framestack_ex.run(
        config_updates={
            "mdp": "data/mdps_with_exploration_policy_sb3/freeway_10_fs30/consolidated.npz",
            "horizon": 10,
            "out": str(tmp_path / "consolidated_framestack.npz"),
        }
    )

    framestack_mdp = np.load(tmp_path / "consolidated_framestack.npz")
    expected_framestack_mdp = np.load(
        "data/mdps_with_exploration_policy_sb3/freeway_10_fs30/consolidated_framestack.npz"
    )
    for key in framestack_mdp:
        np.testing.assert_array_equal(framestack_mdp[key], expected_framestack_mdp[key])


@pytest.mark.uses_rllib
@pytest.mark.uses_cuda
def test_construct_tabular_policy_rllib(tmp_path):
    os.makedirs(os.path.expanduser("~/ray_results/BCTrainer_mdps"), exist_ok=True)

    for mdp_fname, checkpoint_fname, run, horizon in [
        (
            "data/mdps_with_exploration_policy_rllib/freeway_10_fs30/consolidated_framestack.npz",
            "data/logs/bc/mdps/freeway_200_fs30-v0/complete/entropy_0.1/2022-12-30_16-10-41/checkpoint_000100/checkpoint-100",
            "BC",
            10,
        ),
        (
            "data/mdps_with_exploration_policy_rllib/maze_easy_l0_30_fs1/consolidated.npz",
            "data/logs/PPO/procgen/maze/entropy_0.1/2022-12-16_07-33-02/checkpoint_002500/checkpoint-2500",
            "PPO",
            30,
        ),
    ]:
        mdp_name = mdp_fname.split("/")[-2]
        out_dir = tmp_path / mdp_name
        os.makedirs(out_dir, exist_ok=True)
        construct_tabular_policy_ex.run(
            config_updates={
                "rllib_checkpoint": checkpoint_fname,
                "mdp": mdp_fname,
                "horizon": horizon,
                "run": run,
                "out": str(out_dir / "exploration_policy.npy"),
            }
        )

        exploration_policy = np.load(out_dir / "exploration_policy.npy")
        expected_exploration_policy = np.load(
            f"{os.path.dirname(mdp_fname)}/exploration_policy.npy"
        )
        np.testing.assert_array_almost_equal(
            exploration_policy,
            expected_exploration_policy,
            decimal=3,
        )


@pytest.mark.uses_sb3
@pytest.mark.uses_cuda
def test_construct_tabular_policy_sb3(tmp_path):
    for mdp_fname, policy_fname, horizon in [
        (
            "data/mdps_with_exploration_policy_sb3/freeway_10_fs30/consolidated_framestack.npz",
            "data/logs_sb3/BC/BRIDGE/freeway_200_fs30-v0/CnnPolicy/entropy_0.1/2023-10-18_23-17-48/1/final_policy.pt",
            10,
        ),
        (
            "data/mdps_with_exploration_policy_sb3/maze_easy_l0_30_fs1/consolidated.npz",
            "data/logs_sb3/PPO/procgen/maze/easy/CnnPolicy/entropy_0.1/2023-10-18_17-32-12/1/final_policy.pt",
            30,
        ),
    ]:
        mdp_name = mdp_fname.split("/")[-2]
        print(mdp_name)
        out_dir = tmp_path / mdp_name
        os.makedirs(out_dir, exist_ok=True)
        construct_tabular_policy_ex.run(
            config_updates={
                "sb3_policy": policy_fname,
                "mdp": mdp_fname,
                "horizon": horizon,
                "out": str(out_dir / "exploration_policy.npy"),
            }
        )

        exploration_policy = np.load(out_dir / "exploration_policy.npy")
        expected_exploration_policy = np.load(
            f"{os.path.dirname(mdp_fname)}/exploration_policy.npy"
        )
        np.testing.assert_array_almost_equal(
            exploration_policy,
            expected_exploration_policy,
            decimal=3,
        )
