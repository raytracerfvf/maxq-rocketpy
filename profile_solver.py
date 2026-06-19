"""Profile a representative 6-DOF Calisto flight to locate solver hot spots."""
import cProfile
import pstats
import io
import time

from rocketpy import Environment, SolidMotor, Rocket, Flight


def build_flight(ode_solver="LSODA"):
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

    return env, rocket, ode_solver


def run(ode_solver="LSODA"):
    env, rocket, solver = build_flight(ode_solver)
    return Flight(
        rocket=rocket, environment=env,
        rail_length=5.2, inclination=85, heading=0,
        ode_solver=solver,
    )


def time_solver(name, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        f = run(name)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, f


if __name__ == "__main__":
    # Warm caches / JIT-free baseline
    run("LSODA")

    print("=== Wall-time comparison (best of 3) ===")
    results = {}
    for name in ["LSODA", "RK45", "DOP853", "BDF", "Radau"]:
        try:
            dt, f = time_solver(name)
            results[name] = dt
            print(f"{name:8s}: {dt*1000:8.1f} ms   "
                  f"function_evals={sum(f.function_evaluations)}")
        except Exception as e:  # noqa
            print(f"{name:8s}: FAILED ({e})")

    print("\n=== cProfile of a single LSODA flight ===")
    pr = cProfile.Profile()
    pr.enable()
    run("LSODA")
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    print("\n=== Top by total (self) time ===")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(25)
    print(s2.getvalue())
