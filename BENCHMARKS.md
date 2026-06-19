# Solver performance benchmarks (LSODA focus)

Rigorous measurements via [`pyperf`](https://pyperf.readthedocs.io/) (multi-process,
calibrated warmup, mean ± stdev). Workload: standard Calisto 6-DOF flight
(powered ascent + drogue + main), built fresh per iteration with setup
excluded from the timer (`benchmark_solver.py`).

Reproduce:

```bash
pip install pyperf
python benchmark_solver.py -o bench_current.json --processes 6 --values 5 --warmups 1
python -m pyperf show bench_current.json
```

## 1. LSODA is the right default

Same flight, default tolerances (`rtol=atol=1e-6`):

| Solver  | Mean ± stdev |
|---------|--------------|
| **LSODA** | **829 ms ± 13 ms** |
| DOP853  | 1188 ms ± 36 ms |
| BDF     | 1331 ms ± 23 ms |
| Radau   | 2499 ms ± 51 ms |

LSODA is the fastest scipy option; no integrator swap beats it.
(RK45 is omitted: it produced a spurious ~62 ms result under the pyperf
worker that does not reproduce in direct timing — a measurement artifact,
not a real speed.)

## 2. The LSODA algorithm core is ~1% of runtime

From `profile_solver.py` (cProfile): `scipy.integrate._odepack.lsoda` self-time
is ~0.015 s out of ~1.6 s. The cost is the Python derivative
(`u_dot_generalized`, ~56%) and the parachute overshoot processing, which the
solver merely drives. A faster *LSODA implementation* (e.g. numbalsoda) cannot
help unless the derivative itself is compiled.

## 3. Tolerance is the biggest dial (accuracy tradeoff)

LSODA, varying `rtol=atol`:

| rtol=atol | Mean ± stdev | vs 1e-6 |
|-----------|--------------|---------|
| 1e-6 (default) | 838 ms ± 53 ms | — |
| 1e-5 | 692 ms ± 28 ms | −17% |
| 1e-4 | 620 ms ± 21 ms | −26% |
| 1e-3 | 523 ms ± 14 ms | **−38%** |

Apogee changes by <0.01% across this range for this flight. The default
`rtol=1e-6` is conservative for apogee/trajectory estimates; recovery/landing
studies may need it.

## 4. The committed micro-optimization (index cache)

Caching the interpolation interval index (`_cached_bisect_left`) vs. the
previous per-call `bisect_left`, LSODA, `pyperf compare_to`:

```
flight_LSODA: 868 ms -> 831 ms: 1.04x faster
Significant (t=5.18)
```

A real, statistically significant **~4%**. Modest because `bisect_left` was
only ~3.7% of runtime to begin with. A closure-binding follow-up that removed
per-call attribute lookups measured at ~0% — confirming that per-call Python
overhead is not the bottleneck.

## Bottom line

There is no large *free* win. The expensive parts are an accuracy tradeoff
(tolerance, ~38%), load-bearing (the overshoot machinery is 6× faster than
disabling it), or irreducible in pure Python (the derivative). The only
results-preserving multi-fold gain would be compiling `u_dot_generalized`
(numba/Cython), which is a rewrite.
