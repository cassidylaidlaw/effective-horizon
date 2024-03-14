from typing import Dict, List, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from effective_horizon.impala_cnn import ImpalaConvSequence


class ImpalaCNN(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        conv_channels: List[int] = [16, 32, 32],
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if not model_config["vf_share_layers"]:
            raise ValueError("ImpalaCNN does not support separate value network.")

        assert isinstance(obs_space, spaces.Box)
        obs_shape: Tuple[int, ...] = obs_space.shape
        assert len(obs_shape) == 3

        width, height, in_channels = obs_shape
        self.conv_sequences = nn.ModuleList()
        for out_channels in conv_channels:
            self.conv_sequences.append(ImpalaConvSequence(in_channels, out_channels))
            in_channels = out_channels
            width, height = (width + 1) // 2, (height + 1) // 2

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * width * height, 256)

        self.action_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        out = cast(torch.Tensor, input_dict["obs"]).float().permute(0, 3, 1, 2)

        for conv_seq in self.conv_sequences:
            out = conv_seq(out)

        out = self.flatten(out)
        out = F.relu(out)
        out = self.fc(out)
        out = F.relu(out)
        self._backbone_out: torch.Tensor = out

        return self.action_head(out), state

    def value_function(self) -> TensorType:
        vf_out: torch.Tensor = self.value_head(self._backbone_out)
        return vf_out.squeeze(1)


ModelCatalog.register_custom_model("impala_cnn", ImpalaCNN)
