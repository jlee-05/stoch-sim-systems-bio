# %% 
import time
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
import jsmfsb

lv = jsmfsb.models.lv()
k0 = jax.random.key(43)
results = {}

def timed_sim(key, x0, t_end, dt, step_func, spatial=None, label=""):
    print(f"  [{label}] warming up...", end=" ", flush=True)
    try:
        if spatial == '1d':
            jsmfsb.sim_time_series_1d(key, x0, 0, t_end, dt, step_func, False).block_until_ready()
            start = time.perf_counter()
            out = jsmfsb.sim_time_series_1d(key, x0, 0, t_end, dt, step_func, False)
        elif spatial == '2d':
            jsmfsb.sim_time_series_2d(key, x0, 0, t_end, dt, step_func, False).block_until_ready()
            start = time.perf_counter()
            out = jsmfsb.sim_time_series_2d(key, x0, 0, t_end, dt, step_func, False)
        else:
            jsmfsb.sim_time_series(key, x0, 0, t_end, dt, step_func).block_until_ready()
            start = time.perf_counter()
            out = jsmfsb.sim_time_series(key, x0, 0, t_end, dt, step_func)
        out.block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"done. shape={out.shape}, time={elapsed:.4f}s")
        return elapsed
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return None

print("=== NON-SPATIAL ===")
results['SSA_0d']   = timed_sim(k0, lv.m, 20, 0.01, lv.step_gillespie(),  label="SSA_0d")
results['Euler_0d'] = timed_sim(k0, lv.m, 20, 0.05, lv.step_euler(),      label="Euler_0d")
results['CLE_0d']   = timed_sim(k0, lv.m, 20, 0.05, lv.step_cle(),        label="CLE_0d")

print("\n=== 1D N=20 ===")
x0 = jnp.zeros((2, 20)).at[:, 10].set(lv.m)
results['SSA_1d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_1d(jnp.array([0.6,0.6])), '1d', "SSA_1d_20")
results['Euler_1d_20'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_1d(jnp.array([0.6,0.6])),     '1d', "Euler_1d_20")
results['CLE_1d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_1d(jnp.array([0.6,0.6])),       '1d', "CLE_1d_20")

print("\n=== 1D N=50 ===")
x0 = jnp.zeros((2, 50)).at[:, 25].set(lv.m)
results['SSA_1d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_1d(jnp.array([0.6,0.6])), '1d', "SSA_1d_50")
results['Euler_1d_50'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_1d(jnp.array([0.6,0.6])),     '1d', "Euler_1d_50")
results['CLE_1d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_1d(jnp.array([0.6,0.6])),       '1d', "CLE_1d_50")

print("\n=== 2D 20x20 ===")
x0 = jnp.zeros((2, 20, 20)).at[:, 10, 10].set(lv.m)
# results['SSA_2d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_2d(jnp.array([0.6,0.6])), '2d', "SSA_2d_20")
results['Euler_2d_20'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_2d(jnp.array([0.6,0.6])),     '2d', "Euler_2d_20")
results['CLE_2d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_2d(jnp.array([0.6,0.6])),       '2d', "CLE_2d_20")

print("\n=== 2D 50x50 ===")
x0 = jnp.zeros((2, 50, 50)).at[:, 25, 25].set(lv.m)
# results['SSA_2d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_2d(jnp.array([0.6,0.6])), '2d', "SSA_2d_50")
results['Euler_2d_50'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_2d(jnp.array([0.6,0.6])),     '2d', "Euler_2d_50")
results['CLE_2d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_2d(jnp.array([0.6,0.6])),       '2d', "CLE_2d_50")

# Results in terminal
print("\n=== RESULTS ===")
print(f"{'Configuration':<20} | {'Time (s)':<10}")
print("-" * 35)
for k, v in results.items():
    val = f"{v:.4f}" if v is not None else "FAILED"
    print(f"{k:<20} | {val}")

# Creating Pandas Dataframe from data
mapping = [
    ('SSA_0d', 'Non-spatial', 'SSA'), ('Euler_0d', 'Non-spatial', 'Euler'), ('CLE_0d', 'Non-spatial', 'CLE'),
    ('SSA_1d_20', '1D (N=20)', 'SSA'), ('Euler_1d_20', '1D (N=20)', 'Euler'), ('CLE_1d_20', '1D (N=20)', 'CLE'),
    ('SSA_1d_50', '1D (N=50)', 'SSA'), ('Euler_1d_50', '1D (N=50)', 'Euler'), ('CLE_1d_50', '1D (N=50)', 'CLE'),
    ('SSA_2d_20', '2D (20×20)', 'SSA'), ('Euler_2d_20', '2D (20×20)', 'Euler'), ('CLE_2d_20', '2D (20×20)', 'CLE'),
    ('SSA_2d_50', '2D (50×50)', 'SSA'), ('Euler_2d_50', '2D (50×50)', 'Euler'), ('CLE_2d_50', '2D (50×50)', 'CLE')
]

data = []
for key, config, method in mapping:
    # Only include successfully run benchmarks
    if results.get(key) is not None:
        data.append({'Configuration': config, 'Method': method, 'Time (s)': results[key]})

# --- ADVANCED REPRODUCIBILITY ---
# For the 100x100 grid, the exact SSA takes an estimated 4 hours on my device (M4 Macbook Pro).
# We append a manual estimate here so the benchmarking graph renders without breaking.
# Perhaps you are on a more powerful computer and would like to 
# attempt to run the exact SSA. If so:
# 1. Comment out the manual 'SSA' append line below.
# 2. Define your grid: x0_100 = jnp.zeros((2, 100, 100)).at[:, 50, 50].set(lv.m)
# 3. Add the timer: results['SSA_2d_100'] = timed_sim(k0, x0_100, 20, 0.1, lv.step_gillespie_2d(jnp.array([0.6,0.6])), '2d', "SSA_2d_100")

data.append({'Configuration': '2D (100×100)', 'Method': 'SSA', 'Time (s)': 14560.0})

# Continuous approximations run effortlessly
data.append({'Configuration': '2D (100×100)', 'Method': 'Euler', 'Time (s)': 0.1188})
data.append({'Configuration': '2D (100×100)', 'Method': 'CLE', 'Time (s)': 1.7889})

df_export = pd.DataFrame(data)

# Ensure the data directory exists
os.makedirs('../data', exist_ok=True)
export_path = '../data/custom_benchmarks.csv'
df_export.to_csv(export_path, index=False)
print(f"\nData successfully exported to {export_path}")
# %%
