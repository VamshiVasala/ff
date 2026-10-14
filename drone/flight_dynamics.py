"""
Calibrated 5-inch FPV Drone 6-DOF Flight Dynamics Model.

Specifications:
- Mass: 1.5 kg
- Top Speed: 150 km/h (41.67 m/s)
- Thrust-to-Weight Ratio: ~7:1 (Max thrust ~95 N, Hover ~14.71 N)
- Calibrated critically-damped attitude integration and aerodynamic drag.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class DroneState:
    x: float = 0.0          # Lateral X (meters)
    y: float = 5.0          # Vertical Altitude Y (meters)
    z: float = -35.0        # Longitudinal Depth Z (meters)
    vx: float = 0.0         # Velocity X (m/s)
    vy: float = 0.0         # Velocity Y (m/s)
    vz: float = 0.0         # Velocity Z (m/s)
    roll: float = 0.0       # Bank angle (rad)
    pitch: float = 0.0      # Pitch angle (rad)
    yaw: float = 0.0        # Heading angle (rad, 0 = forward along +Z)
    yaw_rate: float = 0.0   # Angular rate (rad/s)
    thrust_n: float = 14.71 # Collective thrust (N)
    speed_kmh: float = 0.0  # Speed in km/h

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "vz": round(self.vz, 3),
            "roll": round(math.degrees(self.roll), 2),
            "pitch": round(math.degrees(self.pitch), 2),
            "yaw": round(math.degrees(self.yaw), 2),
            "thrust_n": round(self.thrust_n, 2),
            "speed_kmh": round(self.speed_kmh, 2),
        }


class FPVDroneModel:
    """
    6-DOF simulation model for an aggressive 5-inch FPV drone.
    Calibrated for stable gate trajectory following up to 150 km/h.
    """

    def __init__(
        self,
        mass_kg: float = 1.5,
        max_speed_kmh: float = 150.0,
        dt: float = 0.02,  # 50 Hz control loop
    ):
        self.mass = mass_kg
        self.gravity = 9.81
        self.hover_thrust = self.mass * self.gravity  # 14.715 N
        self.max_speed_ms = max_speed_kmh / 3.6       # 41.67 m/s
        self.max_thrust = self.hover_thrust * 6.5     # ~95.6 N
        self.dt = dt

        # Calibrated quadratic drag coefficient
        # At full pitch (50 deg) and max thrust, drag matches forward component
        self.drag_coeff_forward = (self.max_thrust * math.sin(math.radians(50))) / (self.max_speed_ms ** 2)
        self.drag_coeff_lateral = self.drag_coeff_forward * 1.4  # higher lateral profile drag
        self.drag_coeff_vertical = 0.18

        self.state = DroneState()

    def reset(self, x: float = 0.0, y: float = 5.0, z: float = -35.0, yaw: float = 0.0):
        self.state = DroneState(x=x, y=y, z=z, yaw=yaw)

    def step(
        self,
        forward_throttle: float,   # [0.0, 1.0] from P9 descending neurons
        steering_yaw_cmd: float,   # [-1.0, 1.0] from DNa01/02 descending neurons
        altitude_target: float = 5.0,
    ) -> DroneState:
        """
        Step flight dynamics forward by self.dt (50 Hz).
        """
        s = self.state

        # 1. Calibrated Yaw & Coordinated Roll Banking (DNa01/02 steering)
        max_yaw_rate = math.radians(200)  # Up to 200 deg/s agile rate
        target_yaw_rate = steering_yaw_cmd * max_yaw_rate

        # Critically damped angular rate filter
        s.yaw_rate += (target_yaw_rate - s.yaw_rate) * 0.40
        s.yaw = (s.yaw + s.yaw_rate * self.dt) % (2 * math.pi)

        # Coordinated banking into turns: bank angle proportional to turn rate and speed
        speed_ms = math.sqrt(s.vx**2 + s.vy**2 + s.vz**2)
        speed_norm = min(1.0, speed_ms / 25.0)
        target_roll = -steering_yaw_cmd * math.radians(40) * (0.3 + 0.7 * speed_norm)
        s.roll += (target_roll - s.roll) * 0.30

        # 2. Forward Pitch from P9 Locomotion Command
        # Smooth pitch-down for acceleration (up to 50 degrees forward tilt)
        target_pitch = forward_throttle * math.radians(50)
        s.pitch += (target_pitch - s.pitch) * 0.25

        # 3. Altitude Hold & Thrust Coupling
        # Compensates for gravity and rotor tilt (cos(pitch)*cos(roll))
        alt_error = altitude_target - s.y
        alt_p = float(np.clip(alt_error * 14.0, -30.0, 45.0))
        alt_d = -s.vy * 5.0  # critically damped vertical velocity

        # Pure quadcopter equilibrium thrust:
        # Tilt angle naturally scales total thrust to provide forward acceleration while holding altitude
        tilt_cos = max(0.25, math.cos(s.pitch) * math.cos(s.roll))
        total_thrust = (self.hover_thrust + alt_p + alt_d) / tilt_cos
        s.thrust_n = float(np.clip(total_thrust, 1.0, self.max_thrust))

        # 4. Forces in Body Frame and Rotation to World Frame
        # In this coordinate frame:
        # +Z is Forward, +X is Right/Lateral, +Y is Vertical Altitude (Up)
        # Body thrust vector tilted by pitch (around X) and roll (around Z):
        f_forward_body = s.thrust_n * math.sin(s.pitch)
        f_lateral_body = -s.thrust_n * math.sin(s.roll)
        f_vertical = s.thrust_n * math.cos(s.pitch) * math.cos(s.roll) - (self.mass * self.gravity)

        # Rotate horizontal forces by heading yaw (around Y):
        fx_world = f_forward_body * math.sin(s.yaw) + f_lateral_body * math.cos(s.yaw)
        fz_world = f_forward_body * math.cos(s.yaw) - f_lateral_body * math.sin(s.yaw)

        # 5. Aerodynamic Drag Forces
        drag_x = self.drag_coeff_lateral * s.vx * abs(s.vx)
        drag_y = self.drag_coeff_vertical * s.vy * abs(s.vy)
        drag_z = self.drag_coeff_forward * s.vz * abs(s.vz)

        ax = (fx_world - drag_x) / self.mass
        ay = (f_vertical - drag_y) / self.mass
        az = (fz_world - drag_z) / self.mass

        # 6. Semi-implicit Euler Integration
        s.vx += ax * self.dt
        s.vy += ay * self.dt
        s.vz += az * self.dt

        s.x += s.vx * self.dt
        s.y = max(0.5, s.y + s.vy * self.dt)
        s.z += s.vz * self.dt

        s.speed_kmh = math.sqrt(s.vx**2 + s.vy**2 + s.vz**2) * 3.6

        return s
