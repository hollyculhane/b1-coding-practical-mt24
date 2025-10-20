# control.py
from dataclasses import dataclass

@dataclass
class PDController:
    KP: float = 0.15
    KD: float = 0.6
    u_min: float | None = None  # optional saturation limits
    u_max: float | None = None

    def __post_init__(self):
        self.e_prev = 0.0
        self.initialized = False

    def reset(self):
        """Reset internal state (call before each new simulation/mission)."""
        self.e_prev = 0.0
        self.initialized = False

    def compute(self, y: float, r: float, dt: float | None = None) -> float:
        """
        Compute control action given measurement y and reference r.
        dt is optional (if you later want derivative scaling by time).
        """
        e = r - y
        if not self.initialized:
            # first step: assume derivative is zero (or use e for derivative)
            de = 0.0
            self.initialized = True
        else:
            de = e - self.e_prev

        # If you have non-uniform sampling and want derivative per second:
        # if dt is not None and dt > 0:
        #     de = (e - self.e_prev) / dt

        u = self.KP * e + self.KD * de

        # apply optional saturation
        if (self.u_min is not None) and (u < self.u_min):
            u = self.u_min
        if (self.u_max is not None) and (u > self.u_max):
            u = self.u_max

        self.e_prev = e
        return float(u)