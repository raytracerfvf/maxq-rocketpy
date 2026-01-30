"""Unit tests for Parachute trigger functionality."""

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
