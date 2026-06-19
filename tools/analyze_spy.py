import re
from collections import defaultdict

TOTAL_MS = 829.0  # rigorous pyperf LSODA mean (setup excluded)
lines = open("/tmp/spy_folded.txt").read().splitlines()

def parse(line):
    i = line.rfind(" ")
    return line[:i].split(";"), int(line[i + 1:])

def subsystem(fr):
    if "vector_matrix.py" in fr:
        return "Vector/Matrix algebra"
    if any(k in fr for k in ("spline_interpolation", "linear_interpolation",
            "akima_interpolation", "polynomial_interpolation",
            "constant_extrapolation", "natural_extrapolation",
            "_cached_bisect", "bisect")):
        return "Interpolation kernels"
    if ("__get_value_opt" in fr or "get_value_opt" in fr or
            ("function.py" in fr and "__call__" in fr) or
            re.search(r"function\.py.*get_value", fr) or "funcify" in fr):
        return "Function dispatch/overhead"
    if "differentiate" in fr:
        return "Differentiation (complex-step)"
    if "rocket.py" in fr or "/motors/" in fr:
        return "Rocket/motor properties"
    if "aero_surface" in fr or "fins.py" in fr:
        return "Aerodynamic surfaces"
    if "environment.py" in fr:
        return "Environment lookups"
    if "parachute.py" in fr or "triggerfunc" in fr:
        return "Parachute trigger eval"
    if "__calculate_and_save_pressure_signals" in fr:
        return "Pressure-signal calc"
    if any(k in fr for k in ("lsoda.py", "_ode.py", "/_ivp/", "odepack",
                             "dense_output", "_call_impl")):
        return "scipy solver internals"
    if any(k in fr for k in ("TimeNode", "add_parachutes", "add_node",
                             "merge", "time_iterator", "__process_overshoot",
                             "__check_overshoot")):
        return "Overshoot/node bookkeeping"
    if any(k in fr for k in ("u_dot_generalized", "udot", "u_dot")):
        return "Derivative self / EOM assembly"
    if "flight.py" in fr:
        return "Flight loop bookkeeping"
    return None

total = setup = sim = 0
activity = defaultdict(int)
sub = {"STEPPING": defaultdict(int), "OVERSHOOT": defaultdict(int),
       "OTHER": defaultdict(int)}

for line in lines:
    frames, cnt = parse(line)
    total += cnt
    if any("build_scenario" in f for f in frames):
        setup += cnt
        continue
    if not (any("flight.py" in f for f in frames)):
        continue
    sim += cnt
    if any("__process_overshoot" in f for f in frames):
        act = "OVERSHOOT"
    elif any(("_step_impl" in f or "base.py:201" in f or "solve_ivp" in f)
             for f in frames):
        act = "STEPPING"
    else:
        act = "OTHER"
    activity[act] += cnt
    cat = None
    for fr in reversed(frames):
        cat = subsystem(fr)
        if cat:
            break
    sub[act][cat or "other/unattributed"] += cnt

print(f"Total samples {total} | setup {100*setup/total:.1f}% | "
      f"simulation {100*sim/total:.1f}% | "
      f"process startup/import {100*(total-setup-sim)/total:.1f}%")
print(f"\nSimulation anchored to pyperf mean = {TOTAL_MS:.0f} ms\n")
print("=== TOP-LEVEL ACTIVITY (clean partition by call-site) ===")
for k in ("STEPPING", "OVERSHOOT", "OTHER"):
    v = activity[k]
    print(f"  {v/sim*100:5.1f}%  {v/sim*TOTAL_MS:6.1f} ms   {k}")
for k in ("STEPPING", "OVERSHOOT", "OTHER"):
    print(f"\n--- {k}: {activity[k]/sim*100:.1f}% ({activity[k]/sim*TOTAL_MS:.0f} ms) "
          f"-- subsystem breakdown ---")
    for c, v in sorted(sub[k].items(), key=lambda x: -x[1]):
        print(f"    {v/sim*100:5.1f}%  {v/sim*TOTAL_MS:6.1f} ms   {c}")
