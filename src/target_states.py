"""
Target quantum states for surrogate inverse optimisation.

Supported targets:
- GHZ
- W
- LINEAR_CLUSTER / CLUSTER / LINEAR
- SINGLE
- ZERO
"""

import numpy as np


def ghz_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float32)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = 1.0 / np.sqrt(2.0)
    return state


def w_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float32)
    norm = 1.0 / np.sqrt(float(n))

    for i in range(n):
        bitstring = ["0"] * n
        bitstring[i] = "1"
        idx = int("".join(bitstring), 2)
        state[idx] = norm

    return state


def linear_cluster_state(n: int) -> np.ndarray:
    dim = 2 ** n
    norm = 1.0 / np.sqrt(float(dim))
    state = np.zeros(dim, dtype=np.float32)

    for i in range(dim):
        bitstring = format(i, f"0{n}b")
        bits = np.array(list(bitstring), dtype=np.int32)
        exponent = int(np.sum(bits[:-1] * bits[1:]))
        state[i] = ((-1) ** exponent) * norm

    return state


def single_state(n: int) -> np.ndarray:
    """
    Original helper state from the earlier optimiser code.
    For n < 4 this state is not valid because indices 5 and 10 do not exist.
    """
    dim = 2 ** n
    if dim <= 10:
        raise ValueError("single_state requires 2**n > 10, e.g. n >= 4.")

    state = np.zeros(dim, dtype=np.float32)
    state[5] = 1.0 / np.sqrt(2.0)
    state[10] = 1.0 / np.sqrt(2.0)
    return state


def zero_state(n: int) -> np.ndarray:
    return np.zeros(2 ** n, dtype=np.float32)


def get_target_state(target_name: str, n: int) -> np.ndarray:
    target_name = target_name.upper()

    if target_name == "GHZ":
        return ghz_state(n)

    if target_name == "W":
        return w_state(n)

    if target_name in {"LINEAR_CLUSTER", "CLUSTER", "LINEAR"}:
        return linear_cluster_state(n)

    if target_name == "SINGLE":
        return single_state(n)

    if target_name == "ZERO":
        return zero_state(n)

    raise ValueError(f"Unknown target state: {target_name}")
