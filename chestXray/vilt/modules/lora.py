from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        down_hidden_states = self.down(hidden_states)
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states



class LoRAConv1dLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Conv1d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,device=device,dtype=dtype)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv1d(rank, out_features, kernel_size=1, stride=1, bias=False,device=device,dtype=dtype)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        input = hidden_states.permute(0,2,1)

        down_hidden_states = self.down(input)
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.permute(0,2,1)
    




class LoRAMissingLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.text_missing_up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.image_missing_up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)


        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.text_missing_up.weight)
        nn.init.zeros_(self.image_missing_up.weight)


    def forward(self, hidden_states: torch.Tensor,missing_type) -> torch.Tensor:

        down_hidden_states = self.down(hidden_states)

        # lora_tensor = []
        # for b in range(down_hidden_states.shape[0]):
        #     temp = down_hidden_states[b:b+1]
        #     if missing_type[b] == 0:
        #         temp_lora = self.image_missing_up(temp) + self.text_missing_up(temp)
        #     elif missing_type[b] == 1:
        #         temp_lora = self.text_missing_up(temp)
        #     else:
        #         temp_lora = self.image_missing_up(temp)
        #     lora_tensor.append(temp_lora)
        # lora_tensor = torch.cat(lora_tensor,dim=0)
        # up_hidden_states = lora_tensor

        if missing_type == 0:
            up_hidden_states = self.image_missing_up(down_hidden_states) + self.text_missing_up(down_hidden_states)
        elif missing_type == 1:
            up_hidden_states = self.text_missing_up(down_hidden_states)
            #up_hidden_states = self.image_missing_up(down_hidden_states) + self.text_missing_up(down_hidden_states)
        else:
            up_hidden_states = self.image_missing_up(down_hidden_states)
            #up_hidden_states = self.image_missing_up(down_hidden_states) + self.text_missing_up(down_hidden_states)



        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states
    
    def up_weights(self):
        return [self.text_missing_up.weight, self.image_missing_up.weight]
