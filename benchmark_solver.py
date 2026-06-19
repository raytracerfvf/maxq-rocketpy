"""Rigorous solver benchmarks for RocketPy using pyperf.

Builds the standard Calisto 6-DOF scenario once per worker process and times
only the ``Flight`` simulation (setup excluded). pyperf handles warmup,
calibration, multiple worker processes and reports mean +/- stdev.

Usage:
    python benchmark_solver.py -o results.json        # full suite
    python benchmark_solver.py --fast                 # fewer values, quick
    python benchmark_solver.py -b flight_LSODA        # single benchmark
"""

import pyperf

from rocketpy import Environment, SolidMotor, Rocket, Flight


def build_scenario():
    env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
    env.set_atmospheric_model(type="standard_atmosphere")

    motor = SolidMotor(
        thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    rocket = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.255)
    rocket.set_rail_buttons(
        upper_button_position=0.0818, lower_button_position=-0.618
    )
    rocket.add_nose(length=0.55829, kind="von karman", position=1.278)
    rocket.add_trapezoidal_fins(
        n=4, root_chord=0.120, tip_chord=0.060, span=0.110,
        position=-1.04956, cant_angle=0.5,
    )
    rocket.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )
    rocket.add_parachute(
        "Main", cd_s=10.0, trigger=800,
        sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
    )
    rocket.add_parachute(
        "Drogue", cd_s=1.0, trigger="apogee",
        sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
    )
    return env, rocket


ENV, ROCKET = build_scenario()


def simulate(ode_solver="LSODA", rtol=1e-6, atol=1e-6, time_overshoot=True):
    Flight(
        rocket=ROCKET, environment=ENV,
        rail_length=5.2, inclination=85, heading=0,
        ode_solver=ode_solver, rtol=rtol, atol=atol,
        time_overshoot=time_overshoot,
    )


if __name__ == "__main__":
    runner = pyperf.Runner()

    # Canonical workload + solver comparison (default tolerances)
    for solver in ["LSODA", "RK45", "DOP853", "BDF", "Radau"]:
        runner.bench_func(
            f"flight_{solver}", simulate, solver, 1e-6, 1e-6, True
        )

    # Tolerance sweep (LSODA) -- the accuracy/speed dial
    for rtol in [1e-6, 1e-5, 1e-4, 1e-3]:
        runner.bench_func(
            f"flight_LSODA_rtol{rtol:.0e}", simulate, "LSODA", rtol, rtol, True
        )
