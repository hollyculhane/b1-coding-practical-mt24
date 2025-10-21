# uuv_mission/control.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PDController:
    KP: float = 0.15
    KD: float = 0.6
    u_min: Optional[float] = None
    u_max: Optional[float] = None

    def __post_init__(self):
        self.e_prev = 0.0
        self._first_step = True

    def reset(self):
        self.e_prev = 0.0
        self._first_step = True

    def compute(self, y: float, r: float, dt: Optional[float] = None) -> float:
        """
        Discrete PD:
          e[t] = r - y
          u[t] = KP*e[t] + KD*(e[t] - e[t-1])
        """
        e = float(r) - float(y)

        if self._first_step:
            de = 0.0
            self._first_step = False
        else:
            de = e - self.e_prev

        u = self.KP * e + self.KD * de

        # optional saturation
        if (self.u_min is not None) and (u < self.u_min):
            u = self.u_min
        if (self.u_max is not None) and (u > self.u_max):
            u = self.u_max

        self.e_prev = e
        return float(u)