"""
Model definitions for surrogate training.

Contains:
- FNN
- HNN
- create_model()
"""

from flax import linen as nn
import numpy as np

class FNN(nn.Module):
    hidden_dims: tuple
    out_dims: int
    nodes: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h, use_bias=True)(x)
            x = nn.gelu(x)

        y_amp = nn.Dense(self.out_dims, use_bias=True)(x)
        return y_amp


class HNN(nn.Module):
    hidden_dims: int
    out_dims: int
    nodes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dims, use_bias=True)(x)

        # For even node counts, nodes / 2 is integer-valued.
        phi = x ** (self.nodes / 2)

        y_amp = nn.Dense(self.out_dims, use_bias=True)(phi)
        return y_amp


def create_model(model_name, hidden_dims, out_dims, nodes):
    model_name = model_name.upper()

    if model_name == "FNN":
        if isinstance(hidden_dims, int):
            raise ValueError(
                "For FNN, hidden_dims should be a tuple, e.g. (2000, 2000, 2000)."
            )

        return FNN(
            hidden_dims=tuple(hidden_dims),
            out_dims=out_dims,
            nodes=nodes,
        )

    if model_name == "HNN":
        if isinstance(hidden_dims, (tuple, list)):
            if len(hidden_dims) != 1:
                raise ValueError(
                    "For HNN, hidden_dims should be an integer, e.g. 1000."
                )
            hidden_dims = int(hidden_dims[0])

        return HNN(
            hidden_dims=int(hidden_dims),
            out_dims=out_dims,
            nodes=nodes,
        )

    raise ValueError(f"Unknown model_name: {model_name}. Use 'HNN' or 'FNN'.")
