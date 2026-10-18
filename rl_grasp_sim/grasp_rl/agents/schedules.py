import numpy as np


def epsilon_at(episode, start, end, decay, n):
    """ε for a given episode: linear or exponential decay from start to end over n episodes."""
    if n <= 0:
        return end
    frac = min(episode / n, 1.0)
    if decay == "linear":
        return start + (end - start) * frac
    if decay == "exp":
        return end + (start - end) * np.exp(-5.0 * frac)  # reaches ~end after n episodes
    raise ValueError(f"unknown epsilon decay {decay!r}")
