import random
import time

import pytest

from effective_horizon.envs.atari import AtariEnv
from effective_horizon.envs.deterministic_registration import (
    GYM_NAMESPACE,
    register_all,
)
from effective_horizon.envs.procgen import ProcgenEnv

try:
    from effective_horizon.rllib.train import ex as train_ex
    from effective_horizon.rllib.train_bc import ex as train_bc_ex
except ImportError:
    pass


# For some reason, pytest and the Ray registry don't play nicely together so we
# have to re-register all the environments before each test.
@pytest.fixture(autouse=True)
def register_envs():
    register_all()
    try:
        from ray.tune.registry import register_env
    except ImportError:
        pass
    else:
        register_env("procgen", ProcgenEnv)
        register_env("atari", AtariEnv)


@pytest.fixture
def random_seed():
    random.seed(time.time_ns())


@pytest.mark.uses_rllib
def test_pretrain_atari_bc(random_seed, tmp_path):
    run = train_bc_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": f"{GYM_NAMESPACE}/freeway_10_fs30-v0",
            "input": "data/atari_human_data/rllib_complete_minimal_actions/freeway",
            "num_workers": 0,
            "entropy_coeff": 0.1,
            "num_training_iters": 10,
            "train_batch_size": 100,
        }
    )
    assert run.result is not None
    assert (
        run.result["info"]["learner"]["default_policy"]["learner_stats"]["bc_loss"] < 1
    )


@pytest.mark.uses_rllib
def test_pretrain_procgen(random_seed, tmp_path):
    run = train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": "procgen",
            "procgen_env_name": "coinrun",
            "run": "PPO",
            "train_batch_size": 100,
            "rollout_fragment_length": 100,
            "sgd_minibatch_size": 100,
            "num_workers": 1,
            "num_sgd_iter": 1,
            "num_training_iters": 1,
            "entropy_coeff": 0.1,
        }
    )
    assert run.result is not None
    assert run.result["episode_reward_mean"] >= 0


@pytest.mark.uses_rllib
@pytest.mark.slow
def test_train_ppo(random_seed, tmp_path):
    mdp, optimal_reward = random.choice(
        [
            (f"{GYM_NAMESPACE}/freeway_10_fs30-v0", 1),
            (f"{GYM_NAMESPACE}/maze_easy_l0_30_fs1-v0", 10),
            (f"{GYM_NAMESPACE}/MiniGrid-KeyCorridorS3R1-v0", 1),
        ]
    )
    run = train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": mdp,
            "run": "PPO",
            "train_batch_size": 100,
            "rollout_fragment_length": 100,
            "sgd_minibatch_size": 100,
            "num_sgd_iter": 1,
            "num_workers": 1,
            "num_training_iters": 1,
            "stop_on_eval_reward": optimal_reward,
        }
    )
    assert run.result is not None
    assert run.result["episode_reward_mean"] >= 0


@pytest.mark.uses_rllib
@pytest.mark.slow
def test_train_dqn(random_seed, tmp_path):
    mdp, optimal_reward = random.choice(
        [
            (f"{GYM_NAMESPACE}/freeway_10_fs30-v0", 1),
            (f"{GYM_NAMESPACE}/maze_easy_l0_30_fs1-v0", 10),
            (f"{GYM_NAMESPACE}/MiniGrid-KeyCorridorS3R1-v0", 1),
        ]
    )
    run = train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": mdp,
            "run": "FastDQN",
            "train_batch_size": 80,
            "rollout_fragment_length": 10,
            "sgd_minibatch_size": 80,
            "num_sgd_iter": 1,
            "num_workers": 1,
            "epsilon_timesteps": 1000,
            "dueling": True,
            "double_q": True,
            "prioritized_replay": True,
            "replay_buffer_capacity": 1000,
            "learning_starts": 0,
            "num_training_iters": 1,
            "stop_on_eval_reward": optimal_reward,
        }
    )
    assert run.result is not None
    assert run.result["episode_reward_mean"] >= 0


@pytest.mark.uses_rllib
def test_train_gorp(random_seed, tmp_path):
    run = train_ex.run(
        config_updates={
            "log_dir": tmp_path / "logs",
            "env_name": "atari",
            "horizon": 27000,
            "frameskip": 4,
            "deterministic": True,
            "rom_file": "pong",
            "reward_scale": 1,
            "done_on_life_lost": True,
            "gamma": 0.99,
            "run": "GORP",
            "episodes_per_action_seq": 1,
            "num_training_iters": 2,
            "stop_on_timesteps": 1000,
        }
    )
    assert run.result is not None
    assert run.result["episode_reward_mean"] >= -21
