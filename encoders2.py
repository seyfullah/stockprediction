from bindsnet.encoding.encoders import Encoder
from encodings2 import lognormal

class LognormalEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable ``LognormalEncoder`` which encodes as defined in
        :code:`bindsnet.encoding.bernoulli`

        :param time: Length of Lognormal spike train per input variable.
        :param dt: Simulation time step.

        Keyword arguments:

        :param float max_prob: Maximum probability of spike per time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = lognormal
