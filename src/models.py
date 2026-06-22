"""
Model definitions for surrogate training.

Contains:
- FNN  (Feedforward Neural Network, multi-layer with GELU activations)
- PNN  (Polynomial Neural Network, single hidden layer with monomial activation)
- create_model()
"""

from flax import linen as nn


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


class PNN(nn.Module):
    """Polynomial Neural Network (PNN).

    One hidden layer of size `hidden_dims` followed by a monomial activation
    x^(nodes/2).  The output is the unnormalised quantum-state amplitude vector.
    """
    hidden_dims: int
    out_dims: int
    nodes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dims, use_bias=True)(x)

        # Monomial activation: x^(nodes/2).
        phi = x ** (self.nodes / 2)

        y_amp = nn.Dense(self.out_dims, use_bias=True)(phi)
        return y_amp


def create_model(model_name, hidden_dims, out_dims, nodes):
    model_name = model_name.upper()

    # Backward compatibility: "HNN" was the previous name for PNN.
    if model_name == "HNN":
        print("Warning: model_name='HNN' is deprecated. Use 'PNN' instead.")
        model_name = "PNN"

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

    if model_name == "PNN":
        if isinstance(hidden_dims, (tuple, list)):
            if len(hidden_dims) != 1:
                raise ValueError(
                    "For PNN, hidden_dims should be an integer, e.g. 1000."
                )
            hidden_dims = int(hidden_dims[0])
        return PNN(
            hidden_dims=int(hidden_dims),
            out_dims=out_dims,
            nodes=nodes,
        )

    raise ValueError(f"Unknown model_name: {model_name}. Use 'PNN' or 'FNN'.")
