"""LSODA-only benchmark (for A/B comparisons via `pyperf compare_to`)."""
import pyperf

from benchmark_solver import make_bench

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("flight_LSODA", make_bench("LSODA"))
