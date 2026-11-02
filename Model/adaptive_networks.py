from __future__ import annotations

"""Small sine-activated networks used by AdaptiveInvariantPINN."""

from torch import Tensor, nn


class LayerCountError(Exception):
    def __init__(self, layers_num: int) -> None:
        self.layers_num = layers_num
        super().__init__()

    def __str__(self) -> str:
        return f"layers_num must be at least 2, received {self.layers_num}"


class Sine(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.sin()


class SineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_num: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers_num < 2:
            raise LayerCountError(layers_num)
        layers = []
        for index in range(layers_num):
            if index == 0:
                layers.extend((nn.Linear(input_dim, hidden_dim), Sine()))
            elif index == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend(
                    (nn.Linear(hidden_dim, hidden_dim), Sine(), nn.Dropout(dropout))
                )
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class SOHPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, 32),
            Sine(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)
