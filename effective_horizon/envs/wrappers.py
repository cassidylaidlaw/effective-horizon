from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium.utils import seeding


class StickyActionsWrapper(gym.Wrapper):
    def __init__(self, env, repeat_action_probability=0.25):
        super().__init__(env)
        self.repeat_action_probability = repeat_action_probability

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Don't use the np_random of the wrapped env, because it may be set
        # deterministically for BRIDGE environments.
        self.sticky_np_random, _ = seeding.np_random(seed)
        self.last_action = None
        return self.env.reset(seed=seed, options=options, **kwargs)

    def step(self, action):
        if (
            self.last_action is not None
            and self.sticky_np_random.uniform() < self.repeat_action_probability
        ):
            action = self.last_action
        self.last_action = action
        return self.env.step(action)
