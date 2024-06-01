
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from prettytable import PrettyTable

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                 ) -> None:

        super(LoRALinear, self).__init__(in_features, out_features, bias)

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / lora_rank  # alpha / r

            self.lora_A = nn.Parameter(self.weight.new_zeros(lora_rank, out_features))
            self.lora_B = nn.Parameter(self.weight.new_zeros(in_features, lora_rank))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_lora() and not self.has_weights_merged:
            self.weight.data += self.lora_scaling * torch.matmul(self.lora_B, self.lora_A).transpose(0,1)
            self.has_weights_merged = True
            return F.linear(input, weight=self.weight, bias=self.bias)
        else:
            return super(LoRALinear, self).forward(input)
        raise NotImplementedError

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self=self)
        if self.has_weights_merged:
            self.weight.data -= self.lora_scaling * torch.matmul(self.lora_B, self.lora_A).transpose(0,1)
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval()
        if self.has_weights_merged == False:
            self.weight.data += self.lora_scaling * torch.matmul(self.lora_B, self.lora_A).transpose(0,1)
            self.has_weights_merged = True
        return self

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out


def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
            table.add_row([name, parameter.numel()])
    print(table)
    print(f"Total params: {total_params}")
    print(f"Total trainable Params: {trainable_params}")
    return model
    raise NotImplementedError 
