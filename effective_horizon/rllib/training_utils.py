import os
from datetime import datetime
from io import BytesIO
from pickle import Unpickler
from typing import Any, Callable, Dict, Type, Union, cast

import cloudpickle
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import AlgorithmConfigDict, PolicyID
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls

from effective_horizon.envs.deterministic_registration import GYM_NAMESPACE


def build_logger_creator(log_dir: str, experiment_name: str):
    experiment_dir = os.path.join(
        log_dir,
        experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in
        <log_dir>/<experiment_name>_<timestamp>
        """

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        return UnifiedLogger(config, experiment_dir)

    return custom_logger_creator


class DummyObject(object):
    def __init__(self, *args, **kwargs):
        pass


class OldCheckpointUnpickler(Unpickler):
    """
    A bit of a hacky workaround for loading old config files that might have pickled
    representations of functions that no longer exist.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("rl_theory."):
            module = "effective_horizon." + module[len("rl_theory.") :]
        if module.startswith("effective_horizon.agents."):
            module = (
                "effective_horizon.rllib.algorithms."
                + module[len("effective_horizon.agents.") :]
            )
        if module == "effective_horizon.training_utils":
            module = "effective_horizon.rllib.training_utils"
        if (module, name) == (
            "gym.utils.seeding",
            "RandomNumberGenerator._generator_ctor",
        ):
            module, name = "effective_horizon.rllib.training_utils", "DummyObject"
        return super().find_class(module, name)


def load_trainer_config(checkpoint_path: str) -> AlgorithmConfigDict:
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory!"
        )
    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config: AlgorithmConfigDict = OldCheckpointUnpickler(f).load()

    if config["env"].startswith("mdps/"):
        config["env"] = f"{GYM_NAMESPACE}/{config['env'][len('mdps/'):]}"

    return config


def load_trainer(
    checkpoint_path: str,
    run: Union[str, Type[Algorithm]],
    config_updates: dict = {},
) -> Algorithm:
    config = load_trainer_config(checkpoint_path)
    if isinstance(config, AlgorithmConfig):
        config = config.to_dict()
    config_updates.setdefault("num_workers", 0)
    config = Algorithm.merge_trainer_configs(
        config, config_updates, _allow_unknown_configs=True
    )

    # Create the Trainer from config.
    if isinstance(run, str):
        cls = cast(Type[Algorithm], get_trainable_cls(run))
    else:
        cls = run
    trainer: Algorithm = cls(config=config)

    # Fix the metadata is necessary using OldCheckpointUnpickler.
    metadata_fname = os.path.join(os.path.dirname(checkpoint_path), ".tune_metadata")
    if os.path.exists(metadata_fname):
        with open(metadata_fname, "rb") as metadata_file:
            metadata = OldCheckpointUnpickler(metadata_file).load()
        with open(metadata_fname, "wb") as metadata_file:
            cloudpickle.dump(metadata, metadata_file)

    # ...and fix the checkpoint itself similarly.
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint_data = OldCheckpointUnpickler(checkpoint_file).load()
            worker_data_pickled = checkpoint_data["worker"]
            worker_data_file = BytesIO(worker_data_pickled)
            worker_data = OldCheckpointUnpickler(worker_data_file).load()
            checkpoint_data["worker"] = cloudpickle.dumps(worker_data)
        with open(checkpoint_path, "wb") as checkpoint_file:
            cloudpickle.dump(checkpoint_data, checkpoint_file)

    # Load state from checkpoint.
    trainer.restore(checkpoint_path)

    return trainer


def load_policies_from_checkpoint(
    checkpoint_fname: str,
    trainer: Algorithm,
    policy_map: Callable[[PolicyID], PolicyID] = lambda policy_id: policy_id,
):
    """
    Load policy model weights from a checkpoint and copy them into the given
    trainer.
    """

    with open(checkpoint_fname, "rb") as checkpoint_file:
        checkpoint_data = cloudpickle.load(checkpoint_file)
    policy_states: Dict[str, Any] = cloudpickle.loads(checkpoint_data["worker"])[
        "state"
    ]

    policy_weights = {
        policy_map(policy_id): policy_state["weights"]
        for policy_id, policy_state in policy_states.items()
    }

    def copy_policy_weights(policy: Policy, policy_id: PolicyID):
        if policy_id in policy_weights:
            policy.set_weights(policy_weights[policy_id])

    workers: WorkerSet = cast(Any, trainer).workers
    workers.foreach_policy(copy_policy_weights)
