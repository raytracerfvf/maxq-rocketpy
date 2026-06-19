"""Regression tests for Flight's cooperative wall-clock watchdog."""

import pytest

from rocketpy import Environment, Flight
import rocketpy.simulation.flight as flight_module

from tests.unit.test_short_burn_step_clamp import (
    _make_light_rocket,
    _make_short_burn_motor,
)


def test_flight_max_wall_time_aborts_after_solver_step(monkeypatch):
    """A configured wall-clock budget should abort once solver control returns."""
    ticks = iter([100.0, 101.5])
    monkeypatch.setattr(flight_module.time, "monotonic", lambda: next(ticks))

    motor = _make_short_burn_motor()
    rocket = _make_light_rocket(motor)

    with pytest.raises(TimeoutError, match="max_wall_time=1.000 s"):
        Flight(
            environment=Environment(),
            rocket=rocket,
            rail_length=2.0,
            inclination=90,
            heading=0,
            terminate_on_apogee=True,
            max_wall_time=1.0,
        )


def test_flight_max_wall_time_must_be_positive():
    motor = _make_short_burn_motor()
    rocket = _make_light_rocket(motor)

    with pytest.raises(ValueError, match="max_wall_time"):
        Flight(
            environment=Environment(),
            rocket=rocket,
            rail_length=2.0,
            inclination=90,
            heading=0,
            terminate_on_apogee=True,
            max_wall_time=0,
        )
