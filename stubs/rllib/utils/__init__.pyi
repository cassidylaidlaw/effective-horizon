import contextlib

from _typeshed import Incomplete
from ray.rllib.utils.annotations import DeveloperAPI as DeveloperAPI
from ray.rllib.utils.annotations import PublicAPI as PublicAPI
from ray.rllib.utils.annotations import override as override
from ray.rllib.utils.deprecation import deprecation_warning as deprecation_warning
from ray.rllib.utils.filter import Filter as Filter
from ray.rllib.utils.filter_manager import FilterManager as FilterManager
from ray.rllib.utils.framework import try_import_tf as try_import_tf
from ray.rllib.utils.framework import try_import_tfp as try_import_tfp
from ray.rllib.utils.framework import try_import_torch as try_import_torch
from ray.rllib.utils.numpy import LARGE_INTEGER as LARGE_INTEGER
from ray.rllib.utils.numpy import MAX_LOG_NN_OUTPUT as MAX_LOG_NN_OUTPUT
from ray.rllib.utils.numpy import MIN_LOG_NN_OUTPUT as MIN_LOG_NN_OUTPUT
from ray.rllib.utils.numpy import SMALL_NUMBER as SMALL_NUMBER
from ray.rllib.utils.numpy import fc as fc
from ray.rllib.utils.numpy import lstm as lstm
from ray.rllib.utils.numpy import one_hot as one_hot
from ray.rllib.utils.numpy import relu as relu
from ray.rllib.utils.numpy import sigmoid as sigmoid
from ray.rllib.utils.numpy import softmax as softmax
from ray.rllib.utils.pre_checks.env import check_env as check_env
from ray.rllib.utils.schedules import ConstantSchedule as ConstantSchedule
from ray.rllib.utils.schedules import ExponentialSchedule as ExponentialSchedule
from ray.rllib.utils.schedules import LinearSchedule as LinearSchedule
from ray.rllib.utils.schedules import PiecewiseSchedule as PiecewiseSchedule
from ray.rllib.utils.schedules import PolynomialSchedule as PolynomialSchedule
from ray.rllib.utils.test_utils import check as check
from ray.rllib.utils.test_utils import (
    check_compute_single_action as check_compute_single_action,
)
from ray.rllib.utils.test_utils import check_train_results as check_train_results
from ray.rllib.utils.test_utils import framework_iterator as framework_iterator
from ray.tune.utils import deep_update as deep_update
from ray.tune.utils import merge_dicts as merge_dicts

def add_mixins(base, mixins, reversed: bool = ...): ...
def force_list(elements: Incomplete | None = ..., to_tuple: bool = ...): ...

class NullContextManager(contextlib.AbstractContextManager):
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

force_tuple: Incomplete
