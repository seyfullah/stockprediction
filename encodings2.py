from typing import Optional

import torch
import numpy as np
from torch.distributions.log_normal import LogNormal

def lognormal(
    datum: torch.Tensor,
    time: Optional[int] = None,
    dt: float = 1.0,
    device="cpu",
    **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Generates Lognormal-distributed spike trains based on input intensity. Inputs must
    be non-negative. Spikes correspond to successful Lognormal trials, with success
    probability equal to (normalized in [0, 1]) input value.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Lognormal spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Lognormal-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Lognormal trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get("max_prob", 1.0)

    assert 0 <= max_prob <= 1, "Maximum firing probability must be in range [0, 1]"
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()

    if time is not None:
        time = int(time / dt)

    # Normalize inputs and rescale (spike probability proportional to input intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Lognormal sampling.
    if time is None:
        spikes = LogNormal(max_prob * datum, torch.tensor([1.0])).to(device)
        spikes = spikes.view(*shape)
    else:
        spikes = LogNormal(max_prob * datum.repeat([time, 1]), torch.tensor([1.0]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()
