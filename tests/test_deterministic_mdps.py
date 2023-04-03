import glob
import logging
import os
import random
import re
from typing import Any, List, cast

import gymnasium as gym
import numpy as np

from effective_horizon.mdp_utils import get_sparse_mdp, load_mdp, run_value_iteration

logger = logging.getLogger(__name__)


def test_random_deterministic_mdp():
    mdp_fnames = glob.glob("data/mdps/*/consolidated.npz")
    mdp_fnames.sort()
    mdp_fname = random.choice(mdp_fnames)

    mdp_name = os.path.basename(os.path.dirname(mdp_fname))

    logger.info(f"loading {mdp_fname}...")
    transitions, rewards = load_mdp(mdp_fname)
    num_states, num_actions = transitions.shape
    done_state = num_states - 1
    sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)

    mdp_name = os.path.basename(os.path.dirname(mdp_fname))
    env_id = f"mdps/{mdp_name}"
    if not re.match(r".*-v\d+", env_id):
        env_id += "-v0"
    logger.info(f"creating gym env {env_id}")
    env = gym.make(env_id)
    horizon: int = cast(Any, env).horizon
    reward_scale = getattr(env, "reward_scale", 1)

    if num_states * horizon >= 100_000_000:
        return

    logger.info("running value iteration")
    vi = run_value_iteration(sparse_transitions, rewards_vector, horizon)

    num_episodes = 100
    logger.info(f"running {num_episodes} episodes")
    for _ in range(num_episodes):
        if random.random() < 0.2:
            guide_qs = -vi.worst_qs
        else:
            guide_qs = vi.optimal_qs
        eps = random.random()

        state = 0
        actions: List[int] = []
        env.reset()
        total_reward = 0
        for t in range(horizon):
            if random.random() < eps:
                action = random.randrange(num_actions)
            else:
                state_qs = guide_qs[t, state]
                action = random.choice(
                    np.arange(num_actions)[state_qs == state_qs.max()]
                )
            actions.append(action)

            expected_reward = rewards[state, action] * reward_scale
            _, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if abs(env_reward - expected_reward) > 1e-4:
                raise ValueError(
                    f"expected reward {expected_reward} but received {env_reward} "
                    f"from following actions {actions}"
                )
            total_reward += expected_reward

            state = int(transitions[state, action])
            if done:
                if not (
                    state == done_state or np.all(transitions[state] == done_state)
                ):
                    raise ValueError(
                        f"expected done state {done_state} but reached {state} "
                        f"with transitions {transitions[state].astype(int)} "
                        f"from following actions {actions}"
                    )
                break

        logger.info(f"actions {' '.join(map(str, actions))} => reward {total_reward}")


def test_atari_framestack():
    mdp_fnames = glob.glob(
        "data/mdps_with_exploration_policy/*/consolidated_framestack.npz"
    )
    mdp_fnames.sort()
    mdp_fname = random.choice(mdp_fnames)

    mdp_name = os.path.basename(os.path.dirname(mdp_fname))

    logger.info(f"loading {mdp_fname}...")
    transitions, rewards = load_mdp(mdp_fname)
    num_states, num_actions = transitions.shape
    done_state = num_states - 1
    mdp = np.load(mdp_fname)
    screens = mdp["screens"]
    screen_ids = mdp["screen_mapping"]

    mdp_name = os.path.basename(os.path.dirname(mdp_fname))
    env_id = f"mdps/{mdp_name}-v0"
    logger.info(f"creating gym env {env_id}")
    env = gym.make(env_id)
    horizon: int = cast(Any, env).horizon
    reward_scale: float = cast(Any, env).reward_scale

    num_episodes = 100
    logger.info(f"running {num_episodes} episodes")
    for _ in range(num_episodes):
        state = 0
        actions: List[int] = []
        obs, _ = env.reset()
        for t in range(horizon):
            action = random.randrange(num_actions)
            actions.append(action)

            expected_screen = screens[screen_ids[state]].transpose(1, 2, 0)
            np.testing.assert_allclose(
                expected_screen,
                obs,
                rtol=0,
                atol=0.6 / 255,
                err_msg=f"mismatched screens from following actions {actions}",
            )

            expected_reward = rewards[state, action] * reward_scale
            obs, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if env_reward != expected_reward:
                raise ValueError(
                    f"expected reward {expected_reward} but received {env_reward} "
                    f"from following actions {actions}"
                )

            state = int(transitions[state, action])
            if done:
                if not (
                    state == done_state or np.all(transitions[state] == done_state)
                ):
                    raise ValueError(
                        f"expected done state {done_state} but reached {state} "
                        f"with transitions {transitions[state].astype(int)} "
                        f"from following actions {actions}"
                    )
                break

        logger.info(f"actions {' '.join(map(str, actions))}")
