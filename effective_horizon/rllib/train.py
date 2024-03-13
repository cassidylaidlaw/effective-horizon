import copy
import faulthandler
import multiprocessing
import os
import signal
import tempfile
from logging import Logger
from typing import Optional

import numpy as np
import ray
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from sacred import SETTINGS as sacred_settings
from sacred import Experiment
from typing_extensions import Literal

from effective_horizon.envs.atari import AtariEnvConfig
from effective_horizon.envs.deterministic_registration import (
    procgen_frameskips,
    procgen_num_actions,
)
from effective_horizon.envs.procgen import ProcgenEnv, ProcgenEnvConfig
from effective_horizon.os_utils import available_cpu_count

from .algorithms.gorp import GORPConfig
from .algorithms.replay_buffers import NoDecompressMultiAgentPrioritizedReplayBuffer
from .training_utils import build_logger_creator, load_policies_from_checkpoint

ex = Experiment("train")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")


# Useful for debugging in Kubernetes.
faulthandler.register(signal.SIGUSR1)


@ex.config
def sacred_config(_log):  # noqa
    run = "PPO"
    config: AlgorithmConfig = get_trainable_cls(run).get_default_config()

    # Environment
    env_name = "MiniGrid-Empty-5x5-v0"
    env_config: dict = {}
    horizon: Optional[int] = None
    reward_scale: float = 1
    clip_rewards = False

    # AtariEnv settings
    if env_name == "atari":
        rom_file: str = ""
        horizon = 5
        done_on_reward: bool = False
        done_on_life_lost: bool = False
        no_done_reward: float = 0
        noops_after_horizon: int = 0
        frameskip: int = 5
        repeat_action_probability = 0.25
        deterministic = True
        fire_on_reset = False
        clip_atari_rewards = False
        atari_env_config: AtariEnvConfig = {
            "rom_file": rom_file,
            "horizon": horizon,
            "done_on_reward": done_on_reward,
            "done_on_life_lost": done_on_life_lost,
            "no_done_reward": no_done_reward,
            "noops_after_horizon": noops_after_horizon,
            "frameskip": frameskip,
            "repeat_action_probability": repeat_action_probability,
            "deterministic": deterministic,
            "reward_scale": reward_scale,
            "fire_on_reset": fire_on_reset,
            "clip_rewards": clip_atari_rewards,
        }
        env_config.update(atari_env_config)

    if env_name == "procgen":
        procgen_env_name = "coinrun"
        distribution_mode: Literal["easy", "hard", "exploration"] = "easy"
        frameskip = procgen_frameskips[procgen_env_name]
        framestack = 1
        num_actions = procgen_num_actions[procgen_env_name]
        start_level = 100
        num_levels = 500
        rand_seed = 0
        procgen_env_config: ProcgenEnvConfig = {
            "env_name": procgen_env_name,
            "distribution_mode": distribution_mode,
            "frameskip": frameskip,
            "framestack": framestack,
            "num_actions": num_actions,
            "start_level": start_level,
            "num_levels": num_levels,
            "rand_seed": rand_seed,
        }
        env_config.update(procgen_env_config)

    try:
        env = _global_registry.get(ENV_CREATOR, env_name)(env_config)
    except KeyError:
        env = None

    config.environment(
        env=env_name,
        env_config=env_config,
        clip_rewards=clip_rewards,
    )

    # Training
    num_workers = 2
    num_envs_per_worker = 1
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    max_gpu_memory: float = np.inf  # noqa: F841
    compress_observations = True
    train_batch_size = 2000
    count_batch_size_by = "timesteps"
    batch_mode = (
        "complete_episodes"
        if count_batch_size_by == "episodes"
        else "truncate_episodes"
    )
    sgd_minibatch_size = 512
    rollout_fragment_length = 200
    num_training_iters = 500  # noqa: F841
    stop_on_timesteps = None  # noqa: F841
    stop_on_eval_reward = None
    stop_on_kl = None  # noqa: F841
    lr = 0.001 if "DQN" in run else 2e-4
    grad_clip = None
    gamma = 1
    num_sgd_iter = 6
    step_exploration = False
    epsilon_timesteps = 1000 if step_exploration else 250000
    final_epsilon = 0.01
    input = "sampler"

    config.framework("torch")
    config.rollouts(
        num_rollout_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode=batch_mode,
        compress_observations=compress_observations,
    )
    config.offline_data(input_=input)
    config.resources(num_gpus=num_gpus)
    config.debugging(seed=seed)

    # Model
    custom_model = None
    vf_share_layers = run in ["GORP", "ntk_gorp"] or custom_model == "impala_cnn"
    max_seq_len = 1
    model_config: ModelConfigDict = {
        "custom_model": custom_model,
        "custom_model_config": {},
        "vf_share_layers": vf_share_layers,
        "max_seq_len": max_seq_len,
    }
    if custom_model == "action_sequence_model":
        model_config["custom_model_config"].update(
            {
                "horizon": horizon,
            }
        )
    if custom_model == "impala_cnn":
        model_config["custom_model_config"].update(
            {
                "hidden_size": 256,
            }
        )
    if isinstance(env, ProcgenEnv):
        model_config["conv_filters"] = [
            (16, (8, 8), 4),
            (32, (4, 4), 2),
            (256, (8, 8), 1),
        ]
    small_vision_net = False
    if small_vision_net:
        model_config["conv_filters"] = [
            (16, (8, 8), 4),
            (16, (4, 4), 2),
            (16, (4, 4), 2),
            (16, (3, 3), 2),
            (16, (3, 3), 1),
        ]
    if run == "ntk_gorp":
        model_config["custom_model_config"].update(copy.deepcopy(model_config))
        model_config["custom_model_config"]["vf_share_layers"] = True
        model_config["custom_model"] = "ntk_model"

    # GORP
    if run == "GORP":
        action_seq_len = 1
        episodes_per_action_seq = 1
        use_max = False
        assert isinstance(config, GORPConfig)
        config.training(
            gamma=gamma,
            action_seq_len=action_seq_len,
            episodes_per_action_seq=episodes_per_action_seq,
            use_max=use_max,
        ).rollouts(
            rollout_fragment_length=1,
            batch_mode="complete_episodes",
        )

    if run == "PPO":
        gae_lambda = 1.0
        vf_loss_coeff = 1e-4
        vf_clip_param = 10.0**2
        entropy_coeff = 0
        entropy_coeff_start = entropy_coeff
        entropy_coeff_end = entropy_coeff
        entropy_coeff_horizon = 1e5
        kl_coeff = 0.2
        kl_target = 0.01
        clip_param = 0.3
        assert isinstance(config, PPOConfig)
        config.training(
            gamma=gamma,
            lr=lr,
            grad_clip=grad_clip,
            num_sgd_iter=num_sgd_iter,
            sgd_minibatch_size=sgd_minibatch_size,
            train_batch_size=train_batch_size,
            model=model_config,
            lambda_=gae_lambda,
            vf_loss_coeff=vf_loss_coeff,
            vf_clip_param=vf_clip_param,
            entropy_coeff_schedule=[
                (0, entropy_coeff_start),
                (entropy_coeff_horizon, entropy_coeff_end),
            ],
            kl_coeff=kl_coeff,
            kl_target=kl_target,
            clip_param=clip_param,
        )

    # DQN
    if "DQN" in run:
        double_q = False
        dueling = False
        learning_starts = 80000
        replay_buffer_capacity = 1_000_000
        target_network_update_freq = 8000
        prioritized_replay = False
        n_step = 1
        replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": replay_buffer_capacity,
        }
        if prioritized_replay:
            replay_buffer_config.update(
                {
                    "type": NoDecompressMultiAgentPrioritizedReplayBuffer,
                    "prioritized_replay_alpha": 0.5,
                }
            )

        exploration_config = {
            "type": "EpsilonGreedy",
            "epsilon_timesteps": epsilon_timesteps,
            "final_epsilon": final_epsilon,
        }

        assert isinstance(config, DQNConfig)
        config.training(
            gamma=gamma,
            lr=lr,
            grad_clip=grad_clip,
            train_batch_size=rollout_fragment_length * num_workers * 8,
            model=model_config,
            target_network_update_freq=target_network_update_freq,
            replay_buffer_config=replay_buffer_config,
            num_steps_sampled_before_learning_starts=learning_starts,
            dueling=dueling,
            double_q=double_q,
            n_step=n_step,
            hiddens=[512],
            adam_epsilon=0.00015,
        )
        config.reporting(
            min_train_timesteps_per_iteration=train_batch_size,
        )
        config.exploration(exploration_config=exploration_config)

    # Evaluation
    evaluation_interval = None if stop_on_eval_reward is None else 1
    evaluation_duration = 1
    evaluation_duration_unit = "episodes"
    evaluation_num_workers = 0
    evaluation_config = {
        "explore": False,
        "input": "sampler",
    }
    config.evaluation(
        evaluation_interval=evaluation_interval,
        evaluation_config=evaluation_config,
        evaluation_num_workers=evaluation_num_workers,
        evaluation_duration=evaluation_duration,
        evaluation_duration_unit=evaluation_duration_unit,
    )

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs"  # noqa: F841
    experiment_tag = None
    experiment_name_parts = [run, env_name]
    if env_name == "atari":
        experiment_name_parts.append(rom_file)
    if env_name == "procgen":
        experiment_name_parts.extend([procgen_env_name, distribution_mode])
        if framestack != 1:
            experiment_name_parts.append(f"framestack_{framestack}")
    if custom_model is not None:
        experiment_name_parts.append(custom_model)
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
    checkpoint_path = None  # noqa: F841
    checkpoint_to_load_policies = None  # noqa: F841


@ex.named_config
def rainbow():
    dueling = True  # noqa: F841
    double_q = True  # noqa: F841
    prioritized_replay = True  # noqa: F841
    n_step = 3  # noqa: F841


@ex.automain
def main(
    config: AlgorithmConfig,
    log_dir,
    experiment_name,
    run,
    num_training_iters,
    stop_on_timesteps: Optional[int],
    stop_on_eval_reward: Optional[float],
    stop_on_kl: Optional[float],
    save_freq,
    checkpoint_path: Optional[str],
    checkpoint_to_load_policies: Optional[str],
    max_gpu_memory: float,
    num_workers: int,
    evaluation_num_workers: int,
    _log: Logger,
):
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        max_gpu_frac = min(max_gpu_memory / total_gpu_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(max_gpu_frac)
        _log.info(
            f"set GPU memory fraction to {max_gpu_frac:.2f} "
            f"({max_gpu_memory}/{total_gpu_memory} GB)"
        )

    temp_dir = tempfile.mkdtemp()
    ray.init(
        num_cpus=max(available_cpu_count(), num_workers + evaluation_num_workers),
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=temp_dir,
    )

    AlgorithmClass = get_trainable_cls(run)
    trainer: Algorithm = AlgorithmClass(
        config,
        logger_creator=build_logger_creator(log_dir, experiment_name),
    )

    if checkpoint_to_load_policies is not None:
        _log.info(f"Initializing policies from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(checkpoint_to_load_policies, trainer)

    if checkpoint_path is not None:
        _log.info(f"Restoring checkpoint at {checkpoint_path}")
        trainer.restore(checkpoint_path)

    num_iters_below_kl = 0
    num_iters_without_exploration = 0

    result = None
    for train_iter in range(num_training_iters):
        _log.info(f"Starting training iteration {trainer.iteration}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            _log.info(f"Saved checkpoint to {checkpoint}")

        if stop_on_eval_reward is not None:
            if (
                result["evaluation"]["episode_reward_mean"]
                >= stop_on_eval_reward - 1e-4
            ):
                break
        if stop_on_kl is not None:
            kl = result["info"]["learner"][DEFAULT_POLICY_ID]["learner_stats"].get(
                "kl", np.inf
            )
            if kl < stop_on_kl:
                num_iters_below_kl += 1
            else:
                num_iters_below_kl = 0
            if num_iters_below_kl >= 10:
                break
        if stop_on_timesteps is not None:
            if result["timesteps_total"] >= stop_on_timesteps:
                break
        exploit_prop = result["info"]["learner"][DEFAULT_POLICY_ID][
            "learner_stats"
        ].get("exploit_prop", 0)
        if exploit_prop == 1:
            num_iters_without_exploration += 1
        else:
            num_iters_without_exploration = 0
        if num_iters_without_exploration >= 10:
            break

        episode_lengths = result["sampler_results"]["hist_stats"]["episode_lengths"]
        if len(episode_lengths) > 0:
            max_episode_len = max(episode_lengths)
            if run == "GORP" and trainer.iteration > max_episode_len:
                break

    checkpoint = trainer.save()
    _log.info(f"Saved final checkpoint to {checkpoint}")

    # Symlink final checkpoint to checkpoint_final
    os.symlink(
        os.path.basename(checkpoint),
        os.path.join(os.path.dirname(checkpoint), "checkpoint_final"),
    )

    return result
