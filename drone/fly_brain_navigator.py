"""
Calibrated Drosophila Central Complex Navigation & Gate Guidance System.

Replaces traditional SLAM and ROS with:
1. EPG Ring Attractor Compass (Ellipsoid Body / EB) for heading tracking.
2. Fan-Shaped Body (FB) phase-shifted Goal Vector calculator with PD damping.
3. Descending Neurons (DNa01/02 for steering, P9 for forward locomotion and gate speed regulation).
"""

import math
from typing import Dict, List, Tuple, Any
import numpy as np


# Standard Cube Obstacle with Two Open Gates
# Cube dimensions: 24m wide x 10m high x 30m long, centered at (0, 5, 0)
# Entrance Gate: z = -15m
# Exit Gate: z = +15m
# Calibrated Symmetric Racetrack around and through the 24m x 10m x 30m Cube Arena
# Gate 1: Entrance at (0, 5, -15)
# Gate 2: Exit at (0, 5, +15)
CUBE_GATE_WAYPOINTS = [
    ( 0.0, 5.0, -45.0),   # WP 0: Runway Alignment Entry
    ( 0.0, 5.0, -30.0),   # WP 1: Runway Pre-gate Lineup
    ( 0.0, 5.0, -15.0),   # WP 2: Gate 1 Center (Entrance Gate)
    ( 0.0, 5.0,   0.0),   # WP 3: Cube Interior Center
    ( 0.0, 5.0,  15.0),   # WP 4: Gate 2 Center (Exit Gate)
    ( 0.0, 5.5,  30.0),   # WP 5: North Arc Entry (Straight Exit)
    ( 3.5, 5.5,  38.5),   # WP 6: North Arc Q1
    (12.0, 5.5,  42.0),   # WP 7: North Arc Apex
    (20.5, 5.5,  38.5),   # WP 8: North Arc Q3
    (24.0, 5.5,  30.0),   # WP 9: Return Leg North (Outside Cube Wall X=12)
    (24.0, 5.5,   0.0),   # WP 10: Return Leg Mid-span
    (24.0, 5.0, -45.0),   # WP 11: South Arc Entry
    (20.5, 5.0, -53.5),   # WP 12: South Arc Q1
    (12.0, 5.0, -57.0),   # WP 13: South Arc Apex
    ( 3.5, 5.0, -53.5),   # WP 14: South Arc Q3
]


class FlyBrainMissionNavigator:
    """
    Biological Drone Mission Planner & Guidance Computer based on
    the Drosophila Fruit Fly Central Complex connectome.
    Calibrated for precision gate navigation, high-speed cruise, and continuous racetrack laps.
    """

    def __init__(self, num_epg_sectors: int = 16, loop: bool = True):
        self.num_sectors = num_epg_sectors
        self.loop = loop
        self.sector_angles = np.linspace(0, 2 * math.pi, num_epg_sectors, endpoint=False)

        # Internal EPG Ring Attractor activity bump (normalized [0, 1])
        self.epg_ring_activity = np.zeros(num_epg_sectors)

        # Descending neuron state
        self.dna_left_hz: float = 0.0
        self.dna_right_hz: float = 0.0
        self.p9_forward_hz: float = 0.0

        # Mission state
        self.current_waypoint_idx: int = 0
        self.waypoints: List[Tuple[float, float, float]] = CUBE_GATE_WAYPOINTS
        self.laps_completed: int = 0

    def set_mission(self, waypoints: List[Tuple[float, float, float]], loop: bool = True):
        self.waypoints = list(waypoints)
        self.current_waypoint_idx = 0
        self.laps_completed = 0
        self.loop = loop

    def compute_guidance(
        self,
        drone_pos: Tuple[float, float, float],   # (x, y, z)
        drone_yaw: float,                       # current heading (rad)
        drone_speed_kmh: float,
        drone_yaw_rate: float = 0.0,            # current yaw rate for PD damping
    ) -> Dict[str, Any]:
        """
        Calibrated biological guidance loop:
        Inputs GPS waypoint -> EPG Ring Attractor -> FB Phase -> DNa/P9 Descending Neurons.
        """
        if not self.waypoints:
            return {
                "forward_throttle": 0.0,
                "steering_yaw_cmd": 0.0,
                "altitude_target": drone_pos[1],
                "mission_complete": True,
                "neural_telemetry": {},
            }

        target_wp = self.waypoints[self.current_waypoint_idx]
        dx = target_wp[0] - drone_pos[0]
        dy = target_wp[1] - drone_pos[1]  # vertical altitude delta
        dz = target_wp[2] - drone_pos[2]  # longitudinal delta
        dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)

        # Calibrated Waypoint / Gate Arrival Logic:
        is_gate_1 = (self.current_waypoint_idx == 2)
        is_gate_2 = (self.current_waypoint_idx == 4)
        is_runway = self.current_waypoint_idx in [0, 1, 2, 3, 4, 5]
        
        passed = (dist_3d < 5.0)
        # Advance along runway when crossing waypoint depth plane
        if is_runway and drone_pos[2] >= target_wp[2]:
            passed = True

        mission_done = False
        if passed:
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
            elif self.loop:
                self.current_waypoint_idx = 0
                self.laps_completed += 1
            else:
                mission_done = True

            target_wp = self.waypoints[self.current_waypoint_idx]
            dx = target_wp[0] - drone_pos[0]
            dy = target_wp[1] - drone_pos[1]
            dz = target_wp[2] - drone_pos[2]
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)

        # 1. Update Biological EPG Compass Ring Attractor
        drone_yaw_norm = drone_yaw % (2 * math.pi)
        angle_diffs = np.arctan2(
            np.sin(self.sector_angles - drone_yaw_norm),
            np.cos(self.sector_angles - drone_yaw_norm),
        )
        self.epg_ring_activity = np.exp(-(angle_diffs ** 2) / (2 * (0.35 ** 2)))
        self.epg_ring_activity /= max(1e-5, np.max(self.epg_ring_activity))

        # 2. Fan-Shaped Body (FB) Goal Vector Phase Calculation
        goal_azimuth = math.atan2(dx, dz) % (2 * math.pi)
        heading_error = math.atan2(
            math.sin(goal_azimuth - drone_yaw_norm),
            math.cos(goal_azimuth - drone_yaw_norm),
        )  # in [-pi, pi]

        # 3. Calibrated Descending Neurons (DNa01/02 Steering)
        kp_steer = 2.60
        kd_steer = 0.32
        raw_steer = (heading_error * kp_steer) - (drone_yaw_rate * kd_steer)
        steering_cmd = float(np.clip(raw_steer, -1.0, 1.0))

        if heading_error > 0:  # Target is to the right
            self.dna_right_hz = min(180.0, abs(heading_error) * 115.0 + 8.0)
            self.dna_left_hz = max(2.0, 8.0 - abs(heading_error) * 15.0)
        else:  # Target is to the left
            self.dna_left_hz = min(180.0, abs(heading_error) * 115.0 + 8.0)
            self.dna_right_hz = max(2.0, 8.0 - abs(heading_error) * 15.0)

        # 4. Calibrated P9 Forward Locomotion (Adaptive Speed Regulation)
        # Precision speed in gates (~50 km/h), cornering speed in turns (~32 km/h), full cruise in straights (~60 km/h)
        alignment = max(0.0, math.cos(heading_error))
        target_speed = 32.0 + 25.0 * (alignment ** 3)
        speed_error = target_speed - drone_speed_kmh
        throttle = float(np.clip(0.35 + (speed_error * 0.025), 0.12, 0.85))

        self.p9_forward_hz = float(throttle * 180.0)

        return {
            "forward_throttle": throttle,
            "steering_yaw_cmd": steering_cmd,
            "altitude_target": target_wp[1],
            "target_waypoint": target_wp,
            "target_waypoint_idx": self.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "distance_to_target_m": round(dist_3d, 2),
            "heading_error_deg": round(math.degrees(heading_error), 1),
            "mission_complete": mission_done,
            "laps_completed": self.laps_completed,
            "neural_telemetry": {
                "epg_ring": [round(float(v), 2) for v in self.epg_ring_activity],
                "p9_forward_hz": round(self.p9_forward_hz, 1),
                "dna_left_hz": round(self.dna_left_hz, 1),
                "dna_right_hz": round(self.dna_right_hz, 1),
                "goal_azimuth_deg": round(math.degrees(goal_azimuth), 1),
            },
        }
