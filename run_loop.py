"""Run many LSODA Calisto flights in a loop, for sampling-profiler analysis."""
import sys

from benchmark_solver import build_scenario
from rocketpy import Flight

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80

for _ in range(N):
    env, rocket = build_scenario()
    Flight(
        rocket=rocket, environment=env,
        rail_length=5.2, inclination=85, heading=0,
        ode_solver="LSODA", rtol=1e-6, atol=1e-6, time_overshoot=True,
    )
