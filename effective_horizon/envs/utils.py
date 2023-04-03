from typing import Any, Union, cast

from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces


def convert_gym_space(
    space: Union[gym_spaces.Space, gymnasium_spaces.Space]
) -> gymnasium_spaces.Space:
    if isinstance(space, gymnasium_spaces.Space):
        return space
    if isinstance(space, gym_spaces.Box):
        return gymnasium_spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=cast(Any, space.dtype),
        )
    elif isinstance(space, gym_spaces.Discrete):
        return gymnasium_spaces.Discrete(n=space.n)
    elif isinstance(space, gym_spaces.MultiDiscrete):
        return gymnasium_spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym_spaces.MultiBinary):
        return gymnasium_spaces.MultiBinary(n=space.n)
    elif isinstance(space, gym_spaces.Tuple):
        return gymnasium_spaces.Tuple(
            spaces=tuple(map(convert_gym_space, space.spaces))
        )
    elif isinstance(space, gym_spaces.Dict):
        return gymnasium_spaces.Dict(
            spaces={
                key: convert_gym_space(value) for key, value in space.spaces.items()
            }
        )
    else:
        raise ValueError(f"Unknown space type: {type(space)}")
