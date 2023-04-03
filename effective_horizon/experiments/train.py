import faulthandler
import multiprocessing
import os
import signal
from logging import Logger
from typing import Optional

import numpy as np
import ray
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import AlgorithmConfigDict, ModelConfigDict
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from sacred import SETTINGS as sacred_settings
from sacred import Experiment
from typing_extensions import Literal

from ..agents.dqn import NoDecompressMultiAgentPrioritizedReplayBuffer
from ..envs.atari import AtariEnvConfig
from ..envs.deterministic_registration import procgen_frameskips, procgen_num_actions
from ..envs.procgen import ProcgenEnv, ProcgenEnvConfig
from ..training_utils import build_logger_creator, load_policies_from_checkpoint

ex = Experiment("train")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")


# Useful for debugging.
faulthandler.register(signal.SIGUSR1)


@ex.config
def sacred_config(_log):  # noqa
    # Environment
    env_name = "MiniGrid-Empty-5x5-v0"
    env_config: dict = {}
    horizon: Optional[int] = None
    reward_scale: float = 1
    # TODO: implement for nondeterministic Atari

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
        }
        env_config.update(atari_env_config)

    if env_name == "procgen":
        procgen_env_name = "coinrun"
        distribution_mode: Literal["easy", "hard", "exploration"] = "easy"
        frameskip = procgen_frameskips[procgen_env_name]
        num_actions = procgen_num_actions[procgen_env_name]
        start_level = 100
        num_levels = 500
        procgen_env_config: ProcgenEnvConfig = {
            "env_name": procgen_env_name,
            "distribution_mode": distribution_mode,
            "frameskip": frameskip,
            "num_actions": num_actions,
            "start_level": start_level,
            "num_levels": num_levels,
        }
        env_config.update(procgen_env_config)

    env = _global_registry.get(ENV_CREATOR, env_name)(env_config)

    # Training
    run = "PPO"
    num_workers = 2
    num_envs_per_worker = 1
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    simple_optimizer = False
    compress_observations = True
    train_batch_size = 2000
    count_batch_size_by = "timesteps"
    sgd_minibatch_size = 500
    rollout_fragment_length = 200
    num_training_iters = 500  # noqa: F841
    stop_on_timesteps = None  # noqa: F841
    stop_on_eval_reward = None
    stop_on_kl = None  # noqa: F841
    lr = 0.001 if "DQN" in run else 2e-4
    grad_clip = None
    gamma = 1
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
    clip_rewards = False
    use_max = False
    num_sgd_iter = 6
    reward_scale = 1

    # Deterministic GORP
    action_seq_len = 1
    episodes_per_action_seq = 1

    # DQN
    double_q = False
    dueling = False
    learning_starts = 80000
    replay_buffer_capacity = 1_000_000
    target_network_update_freq = 8000
    epsilon_timesteps = 250000
    final_epsilon = 0.01
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

    # Model
    custom_model = None
    vf_share_layers = False
    model_config: ModelConfigDict = {
        "custom_model": custom_model,
        "custom_model_config": {},
        "vf_share_layers": vf_share_layers,
        "max_seq_len": 1,
    }
    if isinstance(env, ProcgenEnv):
        model_config["conv_filters"] = [
            (16, (8, 8), 4),
            (32, (4, 4), 2),
            (256, (8, 8), 1),
        ]

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs"  # noqa: F841
    experiment_tag = None
    experiment_name_parts = [run, env_name]
    if env_name == "atari":
        experiment_name_parts.append(rom_file)
    if env_name == "procgen":
        experiment_name_parts.append(procgen_env_name)
    if custom_model is not None:
        experiment_name_parts.append(custom_model)
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
    checkpoint_path = None  # noqa: F841
    checkpoint_to_load_policies = None  # noqa: F841

    config: AlgorithmConfigDict = {  # noqa: F841
        "env": env_name,
        "env_config": env_config,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "num_gpus": num_gpus,
        "simple_optimizer": simple_optimizer,
        "compress_observations": compress_observations,
        "train_batch_size": train_batch_size,
        "horizon": horizon,
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "complete_episodes"
        if count_batch_size_by == "episodes"
        else "truncate_episodes",
        "seed": seed,
        "clip_rewards": clip_rewards,
        "gamma": gamma,
        "model": model_config,
        "framework": "torch",
        "framework_str": "torch",
        "evaluation_interval": None if stop_on_eval_reward is None else 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False,
        },
    }
    if run == "GORP":
        config.update(
            {
                "batch_mode": "complete_episodes",
                "action_seq_len": action_seq_len,
                "rollout_fragment_length": 1,
                "episodes_per_action_seq": episodes_per_action_seq,
                "use_max": use_max,
            }
        )
    if run != "GORP":
        config.update(
            {
                "grad_clip": grad_clip,
                "lr": lr,
                "sgd_minibatch_size": sgd_minibatch_size,
                "num_sgd_iter": num_sgd_iter,
            }
        )
    if run not in ["GORP", "DQN", "FastDQN"]:
        config.update(
            {
                "vf_loss_coeff": vf_loss_coeff,
                "vf_clip_param": vf_clip_param,
            }
        )
    if "PPO" in run:
        config.update(
            {
                "lambda": gae_lambda,
                "kl_coeff": kl_coeff,
                "kl_target": kl_target,
                "clip_param": clip_param,
                "entropy_coeff_schedule": [
                    (0, entropy_coeff_start),
                    (entropy_coeff_horizon, entropy_coeff_end),
                ],
            }
        )
    if "DQN" in run:
        config.update(
            {
                "double_q": double_q,
                "dueling": dueling,
                "replay_buffer_config": replay_buffer_config,
                "num_steps_sampled_before_learning_starts": learning_starts,
                "target_network_update_freq": target_network_update_freq,
                "n_step": n_step,
                "exploration_config": {
                    "epsilon_timesteps": epsilon_timesteps,
                    "final_epsilon": final_epsilon,
                },
                "train_batch_size": rollout_fragment_length * num_workers * 8,
                "timesteps_per_iteration": train_batch_size,
                "hiddens": [512],
                "adam_epsilon": 0.00015,
            }
        )
        del config["num_sgd_iter"]
    if run == "DQN":
        del config["sgd_minibatch_size"]


@ex.named_config
def rainbow():
    dueling = True  # noqa: F841
    double_q = True  # noqa: F841
    prioritized_replay = True  # noqa: F841
    n_step = 3  # noqa: F841


@ex.automain
def main(
    config,
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
    _log: Logger,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    AlgorithmClass = get_trainable_cls(run)
    trainer: Algorithm = AlgorithmClass(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    if checkpoint_to_load_policies is not None:
        _log.info(f"Initializing policies from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(checkpoint_to_load_policies, trainer)

    if checkpoint_path is not None:
        _log.info(f"Restoring checkpoint at {checkpoint_path}")
        trainer.restore(checkpoint_path)

    num_iters_below_kl = 0

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
