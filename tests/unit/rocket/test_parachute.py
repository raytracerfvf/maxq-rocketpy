"""Unit tests for the Parachute class.

Covers:
- Trigger function signatures (3, 4, or 5 parameters; time argument).
- Radius and drag_coefficient parameters introduced in PR #889.
"""

import numpy as np
import pytest

from rocketpy import Parachute


class TestParachuteTriggerSignatures:
    """Test that parachute triggers work with different parameter counts."""

    def test_trigger_with_5_parameters_receives_time(self):
        """Test that 5-parameter triggers receive the time argument."""
        deploy_time = 5.0
        received_time = None

        def time_trigger(p, h, y, sensors, time):
            nonlocal received_time
            received_time = time
            return time >= deploy_time

        parachute = Parachute(
            name="test",
            cd_s=1.0,
            trigger=time_trigger,
            sampling_rate=100,
        )

        # Should not trigger before deploy time
        assert not parachute.triggerfunc(0, 0, [0] * 13, [], 4.0)
        assert received_time == 4.0

        # Should trigger at deploy time
        assert parachute.triggerfunc(0, 0, [0] * 13, [], 5.0)
        assert received_time == 5.0

        # Should trigger after deploy time
        assert parachute.triggerfunc(0, 0, [0] * 13, [], 10.0)
        assert received_time == 10.0

    def test_trigger_with_4_parameters_backward_compatible(self):
        """Test that 4-parameter triggers still work (backward compatibility)."""

        def old_trigger(p, h, y, sensors):
            return h < 100

        parachute = Parachute(
            name="test",
            cd_s=1.0,
            trigger=old_trigger,
            sampling_rate=100,
        )

        # Time parameter is passed but ignored by wrapper
        assert parachute.triggerfunc(0, 50, [0] * 13, [], 10.0)
        assert not parachute.triggerfunc(0, 150, [0] * 13, [], 10.0)

    def test_trigger_with_3_parameters_backward_compatible(self):
        """Test that 3-parameter legacy triggers still work."""

        def legacy_trigger(p, h, y):
            return h < 100 and y[5] < 0  # altitude check with descending

        parachute = Parachute(
            name="test",
            cd_s=1.0,
            trigger=legacy_trigger,
            sampling_rate=100,
        )

        # Create state vector with negative vz (descending)
        state = [0] * 13
        state[5] = -10  # vz = -10 m/s (descending)

        # Time and sensors parameters are passed but ignored by wrapper
        assert parachute.triggerfunc(0, 50, state, [], 10.0)

        # Not descending - should not trigger
        state[5] = 10  # vz = 10 m/s (ascending)
        assert not parachute.triggerfunc(0, 50, state, [], 10.0)

    def test_trigger_with_invalid_parameter_count_raises(self):
        """Test that triggers with invalid parameter counts raise ValueError."""

        def bad_trigger(p, h):  # Only 2 parameters
            return h < 100

        with pytest.raises(ValueError, match="must have 3, 4, or 5 parameters"):
            Parachute(
                name="test",
                cd_s=1.0,
                trigger=bad_trigger,
                sampling_rate=100,
            )

    def test_altitude_trigger_includes_time_parameter(self):
        """Test that altitude (float) triggers have correct signature."""
        parachute = Parachute(
            name="test",
            cd_s=1.0,
            trigger=100.0,  # Altitude trigger
            sampling_rate=100,
        )

        # Create state vector with negative vz (descending)
        state = [0] * 13
        state[5] = -10  # vz = -10 m/s (descending)

        # Should trigger when below altitude and descending
        # Time parameter is accepted but not used
        assert parachute.triggerfunc(0, 50, state, [], 5.0)
        assert not parachute.triggerfunc(0, 150, state, [], 5.0)

    def test_apogee_trigger_includes_time_parameter(self):
        """Test that apogee triggers have correct signature."""
        parachute = Parachute(
            name="test",
            cd_s=1.0,
            trigger="apogee",
            sampling_rate=100,
        )

        # Create state vector
        state = [0] * 13

        # Should trigger when descending (vz < 0)
        state[5] = -1  # Descending
        assert parachute.triggerfunc(0, 1000, state, [], 10.0)

        # Should not trigger when ascending (vz > 0)
        state[5] = 1  # Ascending
        assert not parachute.triggerfunc(0, 1000, state, [], 10.0)


class TestTimeTriggerUseCases:
    """Test practical time-based trigger use cases."""

    def test_burnout_plus_delay_trigger(self):
        """Test a trigger that fires at burnout time + delay."""
        burnout_time = 3.5
        delay = 2.0
        deploy_time = burnout_time + delay

        def burnout_delay_trigger(p, h, y, sensors, time):
            return time >= deploy_time

        parachute = Parachute(
            name="drogue",
            cd_s=1.0,
            trigger=burnout_delay_trigger,
            sampling_rate=100,
        )

        state = [0] * 13

        # Before deploy time
        assert not parachute.triggerfunc(0, 1000, state, [], 4.0)
        assert not parachute.triggerfunc(0, 1000, state, [], 5.0)

        # At deploy time
        assert parachute.triggerfunc(0, 1000, state, [], 5.5)

        # After deploy time
        assert parachute.triggerfunc(0, 1000, state, [], 10.0)

    def test_absolute_flight_time_trigger(self):
        """Test a trigger that fires at absolute flight time."""
        flight_time = 15.0

        def flight_time_trigger(p, h, y, sensors, time):
            return time >= flight_time

        parachute = Parachute(
            name="main",
            cd_s=10.0,
            trigger=flight_time_trigger,
            sampling_rate=100,
        )

        state = [0] * 13

        assert not parachute.triggerfunc(0, 500, state, [], 10.0)
        assert parachute.triggerfunc(0, 500, state, [], 15.0)
        assert parachute.triggerfunc(0, 500, state, [], 20.0)


def _make_parachute(**kwargs):
    defaults = {
        "name": "test",
        "cd_s": 10.0,
        "trigger": "apogee",
        "sampling_rate": 100,
    }
    defaults.update(kwargs)
    return Parachute(**defaults)


class TestParachuteRadiusEstimation:
    """Tests for auto-computed radius from cd_s and drag_coefficient."""

    def test_radius_auto_computed_from_cd_s_default_drag_coefficient(self):
        """When radius is not provided the radius is estimated using the
        default drag_coefficient of 1.4 and the formula R = sqrt(cd_s / (Cd * pi))."""
        cd_s = 10.0
        parachute = _make_parachute(cd_s=cd_s)
        expected_radius = np.sqrt(cd_s / (1.4 * np.pi))
        assert parachute.radius == pytest.approx(expected_radius, rel=1e-9)

    def test_radius_auto_computed_uses_custom_drag_coefficient(self):
        """When drag_coefficient is provided and radius is not, the radius
        must be estimated using the given drag_coefficient."""
        cd_s = 10.0
        custom_cd = 0.75
        parachute = _make_parachute(cd_s=cd_s, drag_coefficient=custom_cd)
        expected_radius = np.sqrt(cd_s / (custom_cd * np.pi))
        assert parachute.radius == pytest.approx(expected_radius, rel=1e-9)

    def test_explicit_radius_overrides_estimation(self):
        """When radius is explicitly provided, it must be used directly and
        drag_coefficient must be ignored for the radius calculation."""
        explicit_radius = 2.5
        parachute = _make_parachute(radius=explicit_radius, drag_coefficient=0.5)
        assert parachute.radius == explicit_radius

    def test_drag_coefficient_stored_on_instance(self):
        """drag_coefficient must be stored as an attribute regardless of
        whether radius is provided or not."""
        parachute = _make_parachute(drag_coefficient=0.75)
        assert parachute.drag_coefficient == 0.75

    def test_drag_coefficient_default_is_1_4(self):
        """Default drag_coefficient must be 1.4 for backward compatibility."""
        parachute = _make_parachute()
        assert parachute.drag_coefficient == pytest.approx(1.4)

    def test_drogue_radius_smaller_than_main(self):
        """A drogue (cd_s=1.0) must have a smaller radius than a main (cd_s=10.0)
        when using the same drag_coefficient."""
        main = _make_parachute(cd_s=10.0)
        drogue = _make_parachute(cd_s=1.0)
        assert drogue.radius < main.radius

    def test_drogue_radius_approximately_0_48(self):
        """For cd_s=1.0 and drag_coefficient=1.4, the estimated radius
        must be approximately 0.48 m (fixes the previous hard-coded 1.5 m)."""
        drogue = _make_parachute(cd_s=1.0)
        assert drogue.radius == pytest.approx(0.476, abs=1e-3)

    def test_main_radius_approximately_1_51(self):
        """For cd_s=10.0 and drag_coefficient=1.4, the estimated radius
        must be approximately 1.51 m, matching the old hard-coded value."""
        main = _make_parachute(cd_s=10.0)
        assert main.radius == pytest.approx(1.508, abs=1e-3)


class TestParachuteSerialization:
    """Tests for to_dict / from_dict round-trip including drag_coefficient."""

    def test_to_dict_includes_drag_coefficient(self):
        """to_dict must include the drag_coefficient key."""
        parachute = _make_parachute(drag_coefficient=0.75)
        data = parachute.to_dict()
        assert "drag_coefficient" in data
        assert data["drag_coefficient"] == 0.75

    def test_from_dict_round_trip_preserves_drag_coefficient(self):
        """A Parachute serialized to dict and restored must have the same
        drag_coefficient."""
        original = _make_parachute(cd_s=5.0, drag_coefficient=0.75)
        data = original.to_dict()
        restored = Parachute.from_dict(data)
        assert restored.drag_coefficient == pytest.approx(0.75)
        assert restored.radius == pytest.approx(original.radius, rel=1e-9)

    def test_from_dict_defaults_drag_coefficient_to_1_4_when_absent(self):
        """Dicts serialized before drag_coefficient was added (no key) must
        fall back to 1.4 for backward compatibility."""
        data = {
            "name": "legacy",
            "cd_s": 10.0,
            "trigger": "apogee",
            "sampling_rate": 100,
            "lag": 0,
            "noise": (0, 0, 0),
            # no drag_coefficient key — simulates old serialized data
        }
        parachute = Parachute.from_dict(data)
        assert parachute.drag_coefficient == pytest.approx(1.4)
