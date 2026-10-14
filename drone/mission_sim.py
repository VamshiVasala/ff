"""
Mission Simulator & 3D Visualizer Launcher for Fly Brain FPV Drone.

Usage:
    # 1. Run headless mission test in terminal / Colab:
    python drone/mission_sim.py --headless --steps 300

    # 2. Launch interactive 3D WebGL simulator in browser:
    python drone/mission_sim.py --browser

    # 3. Inside a Google Colab Notebook cell:
    from drone.mission_sim import show_3d_simulation
    show_3d_simulation()
"""

import os
import sys
import json
import math
import argparse
import webbrowser
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure drone package is accessible
DRONE_DIR = Path(__file__).resolve().parent
if str(DRONE_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_DIR))

from flight_dynamics import FPVDroneModel, DroneState
from fly_brain_navigator import FlyBrainMissionNavigator, CUBE_GATE_WAYPOINTS


DEFAULT_WAYPOINTS = CUBE_GATE_WAYPOINTS


def run_mission_simulation(
    waypoints: List[Tuple[float, float, float]] = None,
    total_steps: int = 500,
    dt: float = 0.02,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes a high-speed FPV drone mission guided entirely by the Fly Central Complex.
    """
    if waypoints is None:
        waypoints = DEFAULT_WAYPOINTS

    drone = FPVDroneModel(mass_kg=1.5, max_speed_kmh=150.0, dt=dt)
    navigator = FlyBrainMissionNavigator()
    navigator.set_mission(waypoints)

    drone.reset(x=0.0, y=5.0, z=-45.0, yaw=0.0)

    trajectory = []
    if verbose:
        print("\n" + "=" * 70)
        print(" FlyBrain 5\" FPV Drone Mission Planner (Biological Central Complex)")
        print(" Mission Arena: 30m Cube with Two Open Gates")
        print(f" Drone Mass: 1.5 kg | Max Speed: 150 km/h | Control Loop: {int(1/dt)} Hz")
        print(f" Waypoints: {len(waypoints)} targets")
        print("=" * 70 + "\n")

    for step in range(1, total_steps + 1):
        # 1. Guidance from Fly Brain Navigator
        guidance = navigator.compute_guidance(
            drone_pos=(drone.state.x, drone.state.y, drone.state.z),
            drone_yaw=drone.state.yaw,
            drone_speed_kmh=drone.state.speed_kmh,
            drone_yaw_rate=drone.state.yaw_rate,
        )

        # 2. Step 6-DOF FPV Flight Dynamics
        drone.step(
            forward_throttle=guidance["forward_throttle"],
            steering_yaw_cmd=guidance["steering_yaw_cmd"],
            altitude_target=guidance["altitude_target"],
        )

        # 3. Log Telemetry
        telemetry = {
            "step": step,
            "time_sec": round(step * dt, 2),
            "state": drone.state.to_dict(),
            "target_wp_idx": guidance["target_waypoint_idx"],
            "dist_to_wp": guidance["distance_to_target_m"],
            "p9_forward_hz": guidance["neural_telemetry"]["p9_forward_hz"],
            "dna_l_hz": guidance["neural_telemetry"]["dna_left_hz"],
            "dna_r_hz": guidance["neural_telemetry"]["dna_right_hz"],
        }
        trajectory.append(telemetry)

        # Print progress every 50 steps (~1 sec)
        if verbose and (step % 50 == 0 or guidance["mission_complete"]):
            s = drone.state
            print(
                f"[T={step*dt:4.1f}s] "
                f"WP {guidance['target_waypoint_idx']+1}/{len(waypoints)} | "
                f"Dist: {guidance['distance_to_target_m']:5.1f}m | "
                f"Spd: {s.speed_kmh:5.1f} km/h | "
                f"Alt: {s.y:4.1f}m | "
                f"Thrust: {s.thrust_n:5.1f}N | "
                f"P9: {guidance['neural_telemetry']['p9_forward_hz']:4.0f}Hz | "
                f"DNa: L{guidance['neural_telemetry']['dna_left_hz']:3.0f}/R{guidance['neural_telemetry']['dna_right_hz']:3.0f}Hz"
            )

        if guidance["mission_complete"]:
            if verbose:
                print(f"\n✓ Mission Completed Successfully at T={step*dt:.2f}s!")
            break

    return {
        "waypoints": waypoints,
        "total_steps": len(trajectory),
        "flight_time_sec": round(len(trajectory) * dt, 2),
        "max_speed_kmh": round(max(t["state"]["speed_kmh"] for t in trajectory), 1),
        "trajectory": trajectory,
    }


def show_3d_simulation(width: int = 1000, height: int = 650):
    """
    Renders the interactive 3D WebGL mission planner inside Google Colab or Jupyter.
    """
    html_path = DRONE_DIR / "simulator_3d.html"
    if not html_path.exists():
        raise FileNotFoundError(f"3D simulator HTML not found at {html_path}")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    try:
        import base64
        from IPython.display import HTML, display
        # Use Base64 Data URI so Colab embeds Three.js cleanly without quote escaping bugs
        b64_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
        iframe_html = f"""
        <iframe src="data:text/html;base64,{b64_html}" 
                width="{width}" height="{height}" 
                style="border: 1px solid #1e293b; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.6);" 
                frameborder="0" allowfullscreen></iframe>
        """
        display(HTML(iframe_html))
    except Exception:
        print("[Notice] Running outside Jupyter/Colab notebook. Opening in default web browser...")
        webbrowser.open(f"file://{html_path.resolve()}")


def open_in_browser():
    """Opens the 3D WebGL simulator directly in your desktop browser."""
    html_path = (DRONE_DIR / "simulator_3d.html").resolve()
    print(f"Opening 3D Mission Planner in browser: file://{html_path}")
    webbrowser.open(f"file://{html_path}")


def main():
    parser = argparse.ArgumentParser(description="FlyBrain 5\" FPV Drone Mission Planner")
    parser.add_argument("--browser", action="store_true", help="Launch interactive 3D simulator in web browser")
    parser.add_argument("--headless", action="store_true", help="Run fast headless flight dynamics simulation")
    parser.add_argument("--steps", type=int, default=400, help="Number of flight simulation steps (default: 400)")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step in seconds (default: 0.02s / 50Hz)")

    args = parser.parse_args()

    if args.browser:
        open_in_browser()
    else:
        results = run_mission_simulation(total_steps=args.steps, dt=args.dt)
        print(f"\n[Summary] Flight Time: {results['flight_time_sec']}s | Max Speed Achieved: {results['max_speed_kmh']} km/h")


if __name__ == "__main__":
    main()
