import base64
import faulthandler
import json
import multiprocessing
import os
import signal
from datetime import datetime
from io import BytesIO
from logging import Logger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import torch.nn as nn
from sacred import SETTINGS as sacred_settings
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from typing_extensions import Literal

from effective_horizon.envs.atari import AtariEnvConfig
from effective_horizon.envs.deterministic_registration import (
    procgen_frameskips,
    procgen_num_actions,
)
from effective_horizon.envs.procgen import ProcgenEnvConfig

from .algorithms.gorp import GORP
from .algorithms.sqirl import SQIRL
from .models import ImpalaCNNFeaturesExtractor, MiniGridCNN

if TYPE_CHECKING:
    from imitation.algorithms.bc import BC  # noqa: F401

ex = Experiment("train_sb3")
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")


# Useful for debugging in Kubernetes.
faulthandler.register(signal.SIGUSR1)


class LogMedianRewardCallback(BaseCallback):
    def __init__(self, stop_on_median_reward: Optional[float] = None, verbose=0):
        super().__init__(verbose)
        self.stop_on_median_reward = stop_on_median_reward

    def _on_step(self) -> bool:
        ep_info_buffer = self.model.ep_info_buffer
        assert ep_info_buffer is not None
        if len(ep_info_buffer) > 0:
            median_reward: float = np.median(
                [ep_info["r"] for ep_info in ep_info_buffer]
            )
            self.logger.record("rollout/ep_rew_median", median_reward)
        else:
            return True

        if (
            self.stop_on_median_reward is None
            or len(ep_info_buffer) < self.model._stats_window_size
        ):
            return True
        else:
            continue_training = bool(median_reward < self.stop_on_median_reward)
            if self.verbose >= 1 and not continue_training:
                print(
                    f"Stopping training because the median reward {median_reward:.2f} "
                    f"is above the threshold {self.stop_on_median_reward}"
                )
            if not continue_training:
                # Make sure we log the final median reward.
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(self.num_timesteps)
            return continue_training


@ex.config
def sacred_config(_log):  # noqa
    # Environment
    env_name = "MiniGrid-Empty-5x5-v0"
    env_config: dict = {}
    horizon: Optional[int] = None
    reward_scale: float = 1

    is_atari = False  # noqa: F841
    atari_wrapper_kwargs = {}  # noqa: F841

    # AtariEnv settings
    if env_name == "BRIDGE/Atari-v0":
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
        clip_rewards = False
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
            "clip_rewards": clip_rewards,
        }
        env_config.update(atari_env_config)

    if env_name == "BRIDGE/Procgen-v0":
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

    policy = "CnnPolicy"
    if "MiniGrid" in env_name and not env_name.endswith("-Sticky-v0"):
        policy = "MlpPolicy"
    policy_kwargs: dict = {  # noqa: F841
        "features_extractor_kwargs": {},
        "activation_fn": nn.ReLU,
    }
    policy_str = policy

    use_impala_cnn = False
    if use_impala_cnn and policy == "CnnPolicy":
        policy_kwargs["features_extractor_class"] = ImpalaCNNFeaturesExtractor
        policy_kwargs["net_arch"] = [256, 256]
        policy_str = "ImpalaCNN_256"
    elif policy == "CnnPolicy" and "MiniGrid" in env_name:
        policy_kwargs["features_extractor_class"] = MiniGridCNN
    dueling = False
    if dueling:
        policy_kwargs["dueling"] = True
        policy_str += "_dueling"

    algo = "PPO"
    gamma = 1 if env_name.startswith("BRIDGE/") else 0.99
    n_envs = 8  # noqa: F841
    use_subproc = False  # noqa: F841
    timesteps = 5_000_000  # noqa: F841
    seed = 0  # noqa: F841

    n_eval_episodes = 1
    eval_freq = 10_000
    deterministic_eval = True
    stop_on_eval_reward = None
    stop_on_median_reward = stop_on_eval_reward  # noqa: F841

    # For BC training on Atari HEAD data.
    bc_data_fname = ""  # noqa: F841
    bc_epochs = 1  # noqa: F841

    # For loading from a pretrained policy.
    pretrained_policy_fname = None  # noqa: F841

    algo_args: Dict[str, Any] = {}
    if algo == "PPO":
        algo_args = {  # noqa: F841
            "n_steps": 128,
            "n_epochs": 4,
            "batch_size": 256,
            "gamma": gamma,
            "learning_rate": 2.5e-4,
            "clip_range": 0.1,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "seed": seed,
        }
        eval_freq = int(max(eval_freq, algo_args["n_steps"] * n_envs))
    elif algo == "DQN":
        algo_args = {  # noqa: F841
            "buffer_size": 100_000,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gamma": gamma,
            "learning_starts": 100_000,
            "target_update_interval": 1000,
            "train_freq": 4,
            "gradient_steps": 1,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
            "optimize_memory_usage": False,
            "seed": seed,
        }
        n_envs = 1
    elif algo == "GORP":
        algo_args = {
            "episodes_per_action_sequence": 1,
            "planning_depth": 1,
            "gamma": gamma,
        }
        policy = "ActionSequencePolicy"
        policy_str = ""
        n_envs = 1
    elif algo == "SQIRL":
        algo_args = {
            "buffer_size": 1_000_000,
            "learning_rate": 1e-4,
            "ema_alpha": 0,
            "batch_size": 128,
            "gamma": gamma,
            "n_epochs": 10,
            "episodes_per_timestep": 1,
            "planning_depth": 1,
            "separate_networks": False,
            "reward_scale": 1.0,
            "use_huber_loss": False,
            "scale_losses": True,
            "loss_scale_smoothing": 0.99,
            "num_atoms": 1,
            "replay_buffer_kwargs": {
                "optimize_memory_usage": True,
                "handle_timeout_termination": False,
            },
            "exploration_fraction": 0,
            "replay_priority_factor": 1.0,
        }
    elif algo == "BC":
        algo_args = {
            "batch_size": 256,
            "minibatch_size": 256,
            "optimizer_kwargs": {"lr": 1e-4},
            "ent_weight": 0,
        }
    else:
        raise ValueError(f"Unsupported algorithm {algo}")

    eval_args = {  # noqa: F841
        "n_eval_episodes": n_eval_episodes,
        "eval_freq": max(eval_freq if algo == "SQIRL" else eval_freq // n_envs, 1),
        "deterministic": deterministic_eval,
    }
    n_eval_envs = n_envs  # noqa: F841

    max_gpu_memory = np.inf  # noqa: F841

    # Logging
    log_dir = "data/logs"  # noqa: F841
    experiment_tag = None
    experiment_name_parts = [algo, env_name]
    if env_name == "atari":
        experiment_name_parts.append(rom_file)
    if env_name == "procgen":
        experiment_name_parts.extend([procgen_env_name, distribution_mode])
        if framestack != 1:
            experiment_name_parts.append(f"framestack_{framestack}")
    experiment_name_parts.append(policy_str)
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    experiment_name_parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    experiment_dir = os.path.join(log_dir, *experiment_name_parts)
    observer = FileStorageObserver(experiment_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    env_name: str,
    env_config,
    is_atari: bool,
    atari_wrapper_kwargs: dict,
    policy: str,
    policy_kwargs: dict,
    pretrained_policy_fname: Optional[str],
    algo: str,
    algo_args,
    n_envs: int,
    n_eval_envs: int,
    use_subproc: bool,
    seed: int,
    timesteps: int,
    stop_on_eval_reward: Optional[float],
    stop_on_median_reward: Optional[float],
    eval_args: dict,
    bc_data_fname: str,
    bc_epochs: int,
    max_gpu_memory: float,
    observer,
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

    if algo == "SQIRL":
        n_envs = min(
            n_envs,
            algo_args["episodes_per_timestep"],
        )

    vec_env_cls: Union[Type[SubprocVecEnv], Type[DummyVecEnv]] = (
        SubprocVecEnv if use_subproc else DummyVecEnv
    )

    eval_env: Union[gym.Env, VecEnv]
    if is_atari:
        env = make_atari_env(
            env_name,
            n_envs=n_envs,
            seed=seed,
            wrapper_kwargs=atari_wrapper_kwargs,
            vec_env_cls=vec_env_cls,
        )
        env = VecFrameStack(env, n_stack=4)
        eval_env = make_atari_env(
            env_name,
            n_envs=n_eval_envs,
            seed=seed,
            wrapper_kwargs=atari_wrapper_kwargs,
            vec_env_cls=vec_env_cls,
        )
        eval_env = VecFrameStack(eval_env, n_stack=4)
    else:

        def env_creator(env_config=env_config):
            import effective_horizon  # noqa: F401

            if env_config:
                return gym.make(env_name, config=env_config)
            else:
                return gym.make(env_name)

        env = make_vec_env(
            env_creator, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls
        )
        eval_env = make_vec_env(
            env_creator, n_envs=n_eval_envs, seed=seed, vec_env_cls=vec_env_cls
        )

    model: Union[BaseAlgorithm, "BC"]
    if algo == "BC":
        import imitation.util.logger as imitation_logger
        from imitation.algorithms.bc import BC  # noqa: F811
        from imitation.data.types import Trajectory
        from imitation.scripts.ingredients.policy_evaluation import eval_policy

        env = VecTransposeImage(env)

        # Get a policy from PPO.
        ppo_model = stable_baselines3.PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
        )

        # Fix for imitation bug.
        ppo_policy = ppo_model.policy
        old_evaluate_actions = ppo_policy.evaluate_actions
        ppo_policy.evaluate_actions = lambda obs, actions: old_evaluate_actions(  # type: ignore[method-assign]
            obs,
            actions.to(ppo_policy.device),
        )

        print(f"Loading BC data from {bc_data_fname}...", end="", flush=True)
        atari_head_trajectories: List[Trajectory] = []
        with open(bc_data_fname, "r") as bc_data_file:
            for line in bc_data_file:
                trajectory_json = json.loads(line)

                # Decode base64 to numpy array for observations
                obs_bytes = BytesIO(base64.b64decode(trajectory_json["obs"]))
                obs: np.ndarray = np.load(obs_bytes)
                obs_bytes.close()
                actions = np.array(trajectory_json["actions"])

                if len(obs) > 0:
                    atari_head_trajectories.append(
                        Trajectory(
                            obs=obs.transpose(0, 3, 1, 2),
                            acts=actions[:-1],
                            # acts=torch.from_numpy(actions[:-1]).to(ppo_model.device),
                            infos=None,
                            terminal=False,
                        )
                    )
                    print(".", end="", flush=True)

            print()

        model = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=np.random.default_rng(seed),
            policy=ppo_policy,
            demonstrations=atari_head_trajectories,
            custom_logger=imitation_logger.configure(
                observer.dir,
                ["stdout", "csv", "tensorboard"],
            ),
            **algo_args,
        )

        def on_epoch_end():
            eval_results = eval_policy(
                model.policy,
                eval_env,
                n_episodes_eval=100,
            )
            for key, value in eval_results.items():
                model.logger.record(f"eval/{key}", value)

        model.train(n_epochs=bc_epochs, on_epoch_end=on_epoch_end)

        ppo_policy.evaluate_actions = old_evaluate_actions  # type: ignore[method-assign]
    else:
        algo_class: Type[BaseAlgorithm]
        if algo == "GORP":
            algo_class = GORP
        elif algo == "SQIRL":
            algo_class = SQIRL
        else:
            algo_class = getattr(stable_baselines3, algo)
        model = algo_class(policy, env, policy_kwargs=policy_kwargs, **algo_args)

        if pretrained_policy_fname is not None:
            print(f"Loading pretrained policy from {pretrained_policy_fname}...")
            pretrained_policy: nn.Module = torch.load(pretrained_policy_fname)
            model.policy.load_state_dict(pretrained_policy.state_dict())

        stop_training_callback: Optional[StopTrainingOnRewardThreshold] = None
        if stop_on_eval_reward is not None:
            stop_training_callback = StopTrainingOnRewardThreshold(
                reward_threshold=stop_on_eval_reward,
                verbose=1,
            )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=stop_training_callback,
            best_model_save_path=observer.dir,
            log_path=observer.dir,
            render=False,
            verbose=1,
            **eval_args,
        )
        median_reward_callback = LogMedianRewardCallback(
            stop_on_median_reward, verbose=1
        )
        callbacks = CallbackList([eval_callback, median_reward_callback])

        model.set_logger(
            sb3_logger.configure(
                observer.dir,
                ["stdout", "csv", "tensorboard"],
            )
        )
        model.learn(total_timesteps=timesteps, callback=callbacks)

    if hasattr(model, "policy"):
        policy_path = os.path.join(observer.dir, "policy.pt")
        print(f"Saving final policy to {policy_path}...")
        torch.save(model.policy, policy_path)
        return {"policy_path": policy_path}
    else:
        return {}
