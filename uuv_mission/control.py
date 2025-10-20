# control.py
from dataclasses import dataclass

@dataclass
class PDController:
    KP: float = 0.15
    KD: float = 0.6
    u_min: float | None = None   # optional actuator lower limit
    u_max: float | None = None   # optional actuator upper limit

    def __post_init__(self):
        # previous error initialization
        self.e_prev = 0.0
        self._first_step = True

    def reset(self):
        """Reset stored state before a new simulation or mission."""
        self.e_prev = 0.0
        self._first_step = True

    def compute(self, y: float, r: float, dt: float | None = None) -> float:
        """
        Compute control action using discrete PD:
            e[t] = r - y
            u[t] = KP * e[t] + KD * (e[t] - e[t-1])

        If dt is provided and you later want to use derivative per-second,
        you could divide the difference by dt. For the coursework brief we
        use the discrete-form (difference) only.
        """
        e = float(r) - float(y)

        if self._first_step:
            de = 0.0
            self._first_step = False
        else:
            de = e - self.e_prev

        u = self.KP * e + self.KD * de

        # Optional saturation
        if (self.u_min is not None) and (u < self.u_min):
            u = self.u_min
        if (self.u_max is not None) and (u > self.u_max):
            u = self.u_max

        # save for next step
        self.e_prev = e
        return float(u)