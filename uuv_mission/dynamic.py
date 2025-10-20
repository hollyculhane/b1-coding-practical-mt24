from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """Create a Mission instance from a CSV file.

        The CSV is expected to contain columns (case-insensitive):
        'reference', 'cave_height', 'cave_depth'.
        Returns a Mission with numpy arrays for each column.
        """
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("pandas is required to load mission CSV") from e

        try:
            df = pd.read_csv(file_name)
        except Exception as e:
            raise FileNotFoundError(f"Could not read mission file '{file_name}': {e}") from e

        # Normalize column names to allow flexible header formatting
        col_map = {c.strip().lower(): c for c in df.columns}
        required = ["reference", "cave_height", "cave_depth"]
        missing = [r for r in required if r not in col_map]
        if missing:
            raise ValueError(f"Missing required columns in mission CSV: {missing}. Found columns: {list(df.columns)}")

        # Extract columns preserving the original names
        reference = df[col_map["reference"]].to_numpy(dtype=float)
        cave_height = df[col_map["cave_height"]].to_numpy(dtype=float)
        cave_depth = df[col_map["cave_depth"]].to_numpy(dtype=float)

        if not (len(reference) == len(cave_height) == len(cave_depth)):
            raise ValueError("Columns 'reference', 'cave_height' and 'cave_depth' must have the same length")

        return cls(reference, cave_height, cave_depth)


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self, mission: Mission, disturbances: np.ndarray) -> Trajectory:
        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            reference_t = mission.reference[t]

            # --- Controller call (inserted here) ---
            try:
                # supports named arguments (recommended for your PDController)
                actions[t] = self.controller.compute(y=observation_t, r=reference_t)
            except TypeError:
                # fallback if controller expects positional args
                actions[t] = self.controller.compute(observation_t, reference_t)
            # --------------------------------------

            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
