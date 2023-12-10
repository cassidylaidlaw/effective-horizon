import os
import random
from typing import List, Tuple

import pytest

from effective_horizon.envs.deterministic_registration import (
    GYM_NAMESPACE,
    register_all,
)

try:
    from effective_horizon.sb3.train import ex as train_ex
except ImportError:
    pass


# For some reason, pytest and the gym registry don't play nicely together so we
# have to re-register all the environments before each test.
@pytest.fixture(autouse=True)
def register_envs():
    register_all()


mdps_for_testing: List[Tuple[str, int]] = [
    (f"{GYM_NAMESPACE}/freeway_10_fs30-v0", 1),
    (f"{GYM_NAMESPACE}/maze_easy_l2_30_fs1-v0", 10),
    (f"{GYM_NAMESPACE}/MiniGrid-Empty-5x5-v0", 1),
]


@pytest.mark.uses_sb3
def test_train_ppo(tmp_path):
    mdp, optimal_reward = random.choice(mdps_for_testing)
    train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": mdp,
            "algo": "PPO",
            "timesteps": 4_000,
            "stop_on_eval_reward": optimal_reward,
        }
    )


@pytest.mark.uses_sb3
def test_train_dqn(tmp_path):
    mdp, optimal_reward = random.choice(mdps_for_testing)
    train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": mdp,
            "algo": "DQN",
            "timesteps": 4_000,
            "algo_args": {"exploration_fraction": 1.0, "learning_starts": 0},
            "stop_on_eval_reward": optimal_reward,
        }
    )


@pytest.mark.uses_sb3
def test_train_sqirl(tmp_path):
    train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": "BRIDGE/Atari-v0",
            "rom_file": "breakout",
            "done_on_life_lost": True,
            "fire_on_reset": True,
            "horizon": 27_000,
            "frameskip": 4,
            "gamma": 0.99,
            "deterministic": False,
            "algo": "SQIRL",
            "use_impala_cnn": True,
            "timesteps": 4_000,
            "algo_args": {
                "n_epochs": 1,
                "episodes_per_timestep": 1,
                "planning_depth": 2,
            },
        }
    )


@pytest.mark.uses_sb3
def test_train_gorp(tmp_path):
    train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": "BRIDGE/Procgen-v0",
            "procgen_env_name": "coinrun",
            "start_level": 0,
            "num_levels": 1,
            "rand_seed": 0,
            "algo": "GORP",
            "timesteps": 10_000,
            "algo_args": {
                "episodes_per_action_sequence": 1,
                "planning_depth": 2,
            },
        }
    )


@pytest.mark.uses_sb3
def test_pretrain_atari_bc(tmp_path):
    env_name = f"{GYM_NAMESPACE}/freeway_10_fs30-v0"

    ex = train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": env_name,
            "bc_data_fname": "data/atari_human_data/json/freeway.json",
            "bc_epochs": 1,
            "algo": "BC",
            "algo_args": {
                "ent_weight": 0.1,
                "batch_size": 4,
                "minibatch_size": 4,
            },
        }
    )

    assert ex.result is not None
    policy_path = ex.result["policy_path"]
    assert os.path.exists(policy_path)

    train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": env_name,
            "algo": "PPO",
            "timesteps": 10_000,
            "stop_on_eval_reward": 1,
            "pretrained_policy_fname": policy_path,
        }
    )
