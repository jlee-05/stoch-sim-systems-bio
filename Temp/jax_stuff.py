# %%
import jax
import matplotlib.pyplot as plt
import jsmfsb
import jax.random as jrandom
import jax.numpy as jnp
import numpy as np
import jsmfsb.models
import time
import seaborn as sns
import scienceplots
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
plt.style.use(['science', 'notebook'])
# %%
k0 = jax.random.key(34)

# --- THE FIX: INCREASE LOCAL DENSITY ---
N = 20      # Coarser grid means more molecules per voxel
M = 20
T = 50
S = 10000 # 100,000 / 400 voxels = 250 S per voxel (Safe! > 20)
I_seed = 5 # 100 I in the center voxel (Safe! > 20)
dt = 0.1

# --- Initial conditions ---
x0 = jnp.zeros((3, N, M))
sir = jsmfsb.models.sir(th=[0.03, 0.3])

num_boxes = N * M
flat_indices = jax.random.choice(k0, num_boxes, shape=(S,))
S_counts_flat = jnp.bincount(flat_indices, length=num_boxes)
S_counts_grid = S_counts_flat.reshape((N, M))

x0 = x0.at[0, :, :].set(S_counts_grid)
x0 = x0.at[1, N//2, M//2].set(I_seed)
x0 = x0.at[2, :, :].set(0)

true_total = float(x0.sum())
print(f"Initial S+I+R: {true_total:.0f}")
print(f"Molecules per voxel = {S/(N*M)}")
# --- Step function ---
d = jnp.array([0.1, 0.1, 0.1])
step_cle = sir.step_cle_2d(d=d)

# --- Run CLE ---
print("Running CLE...", flush=True)
start = time.perf_counter()
out_cle = jsmfsb.sim_time_series_2d(k0, x0, 0, T, dt, step_cle, False)
out_cle.block_until_ready()
time_cle = time.perf_counter() - start
print(f"CLE done in {time_cle:.2f}s")

# --- Compute totals over time ---
t_axis = jnp.linspace(0, T, out_cle.shape[-1])

cle_S = out_cle[0].sum(axis=(0, 1))
cle_I = out_cle[1].sum(axis=(0, 1))
cle_R = out_cle[2].sum(axis=(0, 1))
cle_total = cle_S + cle_I + cle_R

print(f"\nCLE final total: {float(cle_total[-1]):.0f} (started {true_total:.0f})")
print(f"CLE lost: {true_total - float(cle_total[-1]):.0f} "
      f"({(true_total - float(cle_total[-1]))/true_total*100:.3f}%)")

# --- Plot ---
sns.set_theme(style='ticks')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: mass conservation
axes[0].plot(t_axis, cle_total, color='#E07B39', linewidth=1.5, label='CLE Total')
axes[0].axhline(y=true_total, color='black', linestyle='--', linewidth=1, label='True total')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Total population (S+I+R)')
axes[0].set_title('CLE Mass Conservation', fontweight='bold')
# Set y-axis limits to clearly show it stays perfectly flat
axes[0].set_ylim(true_total * 0.95, true_total * 1.05) 
axes[0].legend(frameon=False)

# Right panel: CLE compartments
axes[1].plot(t_axis, cle_S, color='#3EAF67', linewidth=2, label='S (CLE)')
axes[1].plot(t_axis, cle_I, color='#E07B39', linewidth=2, label='I (CLE)')
axes[1].plot(t_axis, cle_R, color='#4878CF', linewidth=2, label='R (CLE)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Population count')
axes[1].set_title('Compartment Dynamics', fontweight='bold')
axes[1].legend(frameon=False)

sns.despine()
plt.tight_layout()
plt.show()
# %%
plot_timepoints = [0, 5, 15, 30, 50]
compute_timepoints = list(range(0, 51, 5))
plot_indices = [compute_timepoints.index(t) for t in plot_timepoints]

# Re-run storing frames (CLE ONLY)
cle_frames = []
x_cle = x0
keys_cle = jax.random.split(k0, len(compute_timepoints))

cle_frames.append(x0)

t_current = 0
for i in range(1, len(compute_timepoints)):
    dt_step = compute_timepoints[i] - compute_timepoints[i-1]
    print(f"Stepping t={t_current} -> t={compute_timepoints[i]}...", flush=True)
    
    # Run only the CLE step
    x_cle = step_cle(keys_cle[i], x_cle, t_current, dt_step)
    cle_frames.append(x_cle)
    
    t_current = compute_timepoints[i]

# --- Plot infected ---
species_idx = 1  # I compartment

# Changed to 1 row, adjusted figsize height from 7 to 3.5
fig, axes = plt.subplots(1, len(plot_timepoints), figsize=(15, 3.5))
plt.subplots_adjust(right=0.88)

# Calculate global max for color scaling across all plotted CLE frames
row_vmax = max(float(cle_frames[idx][species_idx].max()) for idx in plot_indices)

for col, (idx, t) in enumerate(zip(plot_indices, plot_timepoints)):
    ax = axes[col] # axes is now a 1D array
    im = ax.imshow(cle_frames[idx][species_idx], cmap='YlOrRd',
                   vmin=0, vmax=row_vmax, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"$t = {t}$", fontsize=12)
    
    # Add the CLE label to the first image only
    if col == 0:
        ax.set_ylabel('CLE', fontsize=12, fontweight='bold')

# Adjust colorbar positioning for a single row
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7]) 
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Infected count', fontsize=10)

plt.show()
# %% SIR 2D CLE vs Gillespie Mass Conservation
k0 = jax.random.key(34)

N = 30  
M = 30
T = 50
S = 10000
dt = 0.1

# --- Initial conditions ---
x0 = jnp.zeros((3, N, M))
sir = jsmfsb.models.sir(th=[0.3, 0.3])

num_boxes = N * M
flat_indices = jax.random.choice(k0, num_boxes, shape=(S,))
S_counts_flat = jnp.bincount(flat_indices, length=num_boxes)
S_counts_grid = S_counts_flat.reshape((N, M))

x0 = x0.at[0, :, :].set(S_counts_grid)
x0 = x0.at[1, N//2, M//2].set(5)
x0 = x0.at[2, :, :].set(0)

print(f"Initial S+I+R: {float(x0.sum()):.0f}")

# --- Step functions ---
d = jnp.array([0.1, 0.1, 0.1])
step_ssa = sir.step_gillespie_2d(d=d)
step_cle = sir.step_cle_2d(d=d)

# --- Run SSA ---
print("\nRunning SSA...", flush=True)
start = time.perf_counter()
out_ssa = jsmfsb.sim_time_series_2d(k0, x0, 0, T, dt, step_ssa, False)
out_ssa.block_until_ready()
time_ssa = time.perf_counter() - start
print(f"SSA done in {time_ssa:.2f}s")

# --- Run CLE ---
print("Running CLE...", flush=True)
start = time.perf_counter()
out_cle = jsmfsb.sim_time_series_2d(k0, x0, 0, T, dt, step_cle, False)
out_cle.block_until_ready()
time_cle = time.perf_counter() - start
print(f"CLE done in {time_cle:.2f}s")

# --- Compute totals over time ---
# out shape: (3, N, M, n_steps)
t_axis = jnp.linspace(0, T, out_ssa.shape[-1])

ssa_S = out_ssa[0].sum(axis=(0, 1))
ssa_I = out_ssa[1].sum(axis=(0, 1))
ssa_R = out_ssa[2].sum(axis=(0, 1))
ssa_total = ssa_S + ssa_I + ssa_R

cle_S = out_cle[0].sum(axis=(0, 1))
cle_I = out_cle[1].sum(axis=(0, 1))
cle_R = out_cle[2].sum(axis=(0, 1))
cle_total = cle_S + cle_I + cle_R

true_total = float(x0.sum())

print(f"\nSSA final total: {float(ssa_total[-1]):.0f} (started {true_total:.0f})")
print(f"CLE final total: {float(cle_total[-1]):.0f} (started {true_total:.0f})")
print(f"CLE lost: {true_total - float(cle_total[-1]):.0f} "
      f"({(true_total - float(cle_total[-1]))/true_total*100:.1f}%)")
print(f"CLE was {time_ssa/time_cle:.2f}x quicker but...")

# --- Plot ---
sns.set_theme(style='ticks')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: mass conservation
axes[0].plot(t_axis, ssa_total, color='#4878CF', linewidth=1.5, label='SSA')
axes[0].plot(t_axis, cle_total, color='#E07B39', linewidth=1.5, label='CLE')
axes[0].axhline(y=true_total, color='black', linestyle='--',
                linewidth=1, label='True total')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Total population (S+I+R)')
axes[0].set_title('Mass Conservation Violation', fontweight='bold')
axes[0].legend(frameon=False)

# Right panel: CLE compartments vs SSA compartments
axes[1].plot(t_axis, ssa_S, color='#3EAF67', linewidth=1.5,
             linestyle='-',  label='S (SSA)')
axes[1].plot(t_axis, ssa_I, color='#E07B39', linewidth=1.5,
             linestyle='-',  label='I (SSA)')
axes[1].plot(t_axis, ssa_R, color='#4878CF', linewidth=1.5,
             linestyle='-',  label='R (SSA)')
axes[1].plot(t_axis, cle_S, color='#3EAF67', linewidth=1.5,
             linestyle='--', label='S (CLE)')
axes[1].plot(t_axis, cle_I, color='#E07B39', linewidth=1.5,
             linestyle='--', label='I (CLE)')
axes[1].plot(t_axis, cle_R, color='#4878CF', linewidth=1.5,
             linestyle='--', label='R (CLE)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Population count')
axes[1].set_title('Compartment Dynamics: SSA vs CLE', fontweight='bold')
axes[1].legend(frameon=False, fontsize=9, ncol=2)

sns.despine()
plt.tight_layout()
plt.savefig('Report Images/SIR_mass_conservation.png', dpi=300,
            bbox_inches='tight')
plt.show()
# %% ^^^Visualisation of the above^^^
# Timepoints to visualise

plot_timepoints = [0, 5, 15, 30, 50]
compute_timepoints = list(range(0, 51, 5))
plot_indices = [compute_timepoints.index(t) for t in plot_timepoints]

# Re-run storing frames
ssa_frames, cle_frames = [], []
x_ssa, x_cle = x0, x0
keys_ssa = jax.random.split(k0, len(compute_timepoints))
keys_cle = jax.random.split(jax.random.fold_in(k0, 1), len(compute_timepoints))

ssa_frames.append(x0)
cle_frames.append(x0)

t_current = 0
for i in range(1, len(compute_timepoints)):
    dt_step = compute_timepoints[i] - compute_timepoints[i-1]
    print(f"Stepping t={t_current} -> t={compute_timepoints[i]}...", flush=True)
    x_ssa = step_ssa(keys_ssa[i], x_ssa, t_current, dt_step)
    x_cle = step_cle(keys_cle[i], x_cle, t_current, dt_step)
    ssa_frames.append(x_ssa)
    cle_frames.append(x_cle)
    t_current = compute_timepoints[i]

# --- Plot infected ---
species_idx = 1  # I compartment
species_name = 'Infected (I)'

fig, axes = plt.subplots(2, len(plot_timepoints), figsize=(15, 7))
plt.subplots_adjust(right=0.88)

for row, (label, frames) in enumerate(zip(['SSA', 'CLE'],
                                           [ssa_frames, cle_frames])):
    row_vmax = max(float(frames[idx][species_idx].max())
                   for idx in plot_indices)

    for col, (idx, t) in enumerate(zip(plot_indices, plot_timepoints)):
        ax = axes[row, col]
        im = ax.imshow(frames[idx][species_idx], cmap='YlOrRd',
                       vmin=0, vmax=row_vmax, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f"$t = {t}$", fontsize=12)
        if col == 0:
            ax.set_ylabel(label, fontsize=12, fontweight='bold')

    cbar_ax = fig.add_axes([0.90, 0.55 - row * 0.5, 0.015, 0.38])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Infected count', fontsize=10)

# plt.suptitle(f'Spatial SIR: Infected Population on $({N}\\times{M})$ Grid',
#              fontsize=13, fontweight='bold')
plt.savefig('Report Images/SIR_spatial_snapshots.png', dpi=300,
            bbox_inches='tight')
plt.show()
# %%
M = 20
N = 30
T = 10
x0 = jnp.zeros((2, M, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)
step_lv_2d = lv.step_gillespie_2d(jnp.array([0.6, 0.6]))
k0 = jax.random.key(42)
x1 = step_lv_2d(k0, x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i, :, :])
    axis.set_title(lv.n[i])

# %%
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

M = 20
N = 20
T = 10
lv = jsmfsb.models.lv()
k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)
diff = jnp.array([0.6, 0.6])

x0 = jnp.zeros((2, M, N))
x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)

step_ssa = lv.step_gillespie_2d(diff)
step_cle = lv.step_cle_2d(diff)

compute_timepoints = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12]
plot_timepoints = [0, 6, 8, 10, 12] # plot all first

plot_indices = [compute_timepoints.index(t) for t in plot_timepoints]

# --- Run simulations ---
ssa_frames, cle_frames = [x0], [x0]
x_ssa, x_cle = x0, x0
keys_ssa = jax.random.split(k1, len(compute_timepoints))
keys_cle = jax.random.split(k2, len(compute_timepoints))

for i in range(1, len(compute_timepoints)):
    dt = compute_timepoints[i] - compute_timepoints[i-1]
    print(f"Stepping t={compute_timepoints[i-1]} -> t={compute_timepoints[i]}...", flush=True)
    x_ssa = step_ssa(keys_ssa[i], x_ssa, compute_timepoints[i-1], dt)
    x_cle = step_cle(keys_cle[i], x_cle, compute_timepoints[i-1], dt)
    ssa_frames.append(x_ssa)
    cle_frames.append(x_cle)
    print(f"  SSA prey total: {float(x_ssa[0].sum()):.0f}  CLE prey total: {float(x_cle[0].sum()):.0f}", flush=True)

for species_idx, species_label in enumerate(['Prey', 'Predator']):
    fig, axes = plt.subplots(2, len(plot_timepoints), figsize=(14, 6))
    plt.subplots_adjust(right=0.9)

    for row, (method, frames) in enumerate(zip(
        ['SSA', 'CLE'],
        [ssa_frames, cle_frames]
    )):
        row_vmax = max(float(frames[idx][species_idx].max()) for idx in plot_indices)

        for col, (idx, t) in enumerate(zip(plot_indices, plot_timepoints)):
            ax = axes[row, col]
            im = ax.imshow(frames[idx][species_idx], cmap='YlOrRd',
                           vmin=0, vmax=row_vmax, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$t = {t}$", fontsize=12)
            if col == 0:
                ax.set_ylabel(method, fontsize=12, fontweight='bold')

        cbar_ax = fig.add_axes([0.92, 0.55 - row * 0.5, 0.015, 0.38])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'{species_label} count', fontsize=10)

    # plt.suptitle(f'Spatial Lotka-Volterra: {species_label}', 
    #             fontsize=13, fontweight='bold')
    plt.savefig(f'Report Images/LV_spatial_{species_label.lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
# %%
from jax import jit
import jax.numpy as jnp
import jax.lax as jl
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def step_rk4(model, dt=0.01):
    sto = (model.post - model.pre).T
    
    def f(x, t):
        return sto.dot(model.h(x, t))
    
    @jit
    def step(key, x0, t0, deltat):
        x = x0
        t = t0
        termt = t0 + deltat
        
        def advance(state):
            x, t = state
            k1 = f(x, t)
            k2 = f(x + 0.5*dt*k1, t + 0.5*dt)
            k3 = f(x + 0.5*dt*k2, t + 0.5*dt)
            k4 = f(x + dt*k3, t + dt)
            x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            x = jnp.where(x < 0, -x, x) 
            t = t + dt
            return (x, t)
        
        x, t = jl.while_loop(
            lambda state: state[1] < termt, advance, (x, t)
        )
        return x
    return step
    
def simulate_lv(omega, key, n_ssa=5, dt=0.05):
    lv_sh = f"""
    @model:3.1.1=LotkaVolterra
    @compartments
     Cell
    @species
     Cell:Prey={int(50 * omega)}
     Cell:Predator={int(100 * omega)}
    @reactions
     @r=PreyReproduction
      Prey -> 2Prey
      1.0*Prey
     @r=PredatorPreyInteraction
      Prey+Predator -> 2Predator
      {0.005 / omega}*Prey*Predator
     @r=PredatorDeath
      Predator ->
      0.6*Predator
    """
    mod = jsmfsb.shorthand_to_spn(lv_sh)

    # Multiple SSA
    keys = jax.random.split(key, n_ssa + 2)
    ssa_runs = [jsmfsb.sim_time_series(keys[k], mod.m, 0, 30, dt,
                mod.step_gillespie()) for k in range(n_ssa)]

    # Single CLE 
    cle_run = jsmfsb.sim_time_series(keys[n_ssa], mod.m, 0, 30, dt,
                mod.step_cle())

    # Deterministic Euler
    ode_run = jsmfsb.sim_time_series(keys[n_ssa+1], mod.m, 0, 30, dt,
                step_rk4(mod))

    return ssa_runs, cle_run, ode_run

k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)

volumes  = [10, 100]
species  = ['Prey', 'Predator']
colours  = ['#4878CF', '#E07B39']

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for col, (omega, key) in enumerate(zip(volumes, [k1, k2])):
    ssa_runs, cle_run, ode_run = simulate_lv(omega, key, dt=0.01)
    t = jnp.linspace(0, 30, ssa_runs[0].shape[0])

    for row, (sp, col_hex) in enumerate(zip(species, colours)):
        ax = axes[row, col]
        sp_idx = row

        # Faint SSA
        for run in ssa_runs:
            ax.step(t, run[:, sp_idx] / omega, color=col_hex,
                    alpha=0.2, linewidth=0.8, where='post')

        # CLE
        ax.plot(t, cle_run[:, sp_idx] / omega, color=col_hex,
                linewidth=2, label='CLE', zorder=3)

        # Deterministic Euler
        ax.plot(t, ode_run[:, sp_idx] / omega, color='black',
                linewidth=1.5, linestyle='--', label='Euler (det.)', zorder=4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"{sp} concentration")
        if row == 0:
            ax.set_title(f"$\\Omega = {omega}$", fontweight='bold', fontsize=13)
        if row == 1:
            ax.set_xlabel("Time")

legend_elements = [
    Line2D([0], [0], color='grey', alpha=0.4, linewidth=1.5, label='SSA realisations'),
    Line2D([0], [0], color='grey', linewidth=2, label='CLE'),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='RK4 (det.)'),
]
axes[0, 0].legend(handles=legend_elements, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('Report Images/LV_validation.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))

omega = 100
ssa_runs, cle_run, ode_run = simulate_lv(omega, k1)

# Faint SSA realisations
for run in ssa_runs:
    ax.plot(run[:, 0] / omega, run[:, 1] / omega,
            color='green', alpha=0.2, linewidth=0.8)

# Bold CLE
ax.plot(cle_run[:, 0] / omega, cle_run[:, 1] / omega,
        color='green', linewidth=2, label='CLE', zorder=3)

# Deterministic Euler
ax.plot(ode_run[:, 0] / omega, ode_run[:, 1] / omega,
        color='black', linewidth=1.5, linestyle='--',
        label='RK4 (det.)', zorder=4)

# Mark initial condition
ax.scatter([50], [100], color='red', zorder=5, s=60, label='Initial condition')
legend_elements_phase= [
Line2D([0], [0], color='green', alpha=0.3, linewidth=1.5, label='SSA realisations'),
Line2D([0], [0], color='green', linewidth=2, label='CLE'),
Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='RK4 (det.)'),
]
ax.set_xlabel('Prey concentration ($X_1 / \\Omega$)')
ax.set_ylabel('Predator concentration ($X_2 / \\Omega$)')
ax.legend(handles=legend_elements_phase, frameon=True, fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig('Report Images/LV_phase_portrait.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def simulate_lv(omega, key):
    lv_sh = f"""
    @model:3.1.1=LotkaVolterra
    @compartments
     Cell
    @species
     Cell:Prey={int(50 * omega)}
     Cell:Predator={int(100 * omega)}
    @reactions
     @r=PreyReproduction
      Prey -> 2Prey
      1.0*Prey
     @r=PredatorPreyInteraction
      Prey+Predator -> 2Predator
      {0.005 / omega}*Prey*Predator
     @r=PredatorDeath
      Predator ->
      0.6*Predator
    """
    mod = jsmfsb.shorthand_to_spn(lv_sh)
    step_ssa = mod.step_gillespie()
    out_ssa = jsmfsb.sim_time_series(key, mod.m, 0, 20, 0.05, step_ssa)
    step_ode = mod.step_euler()
    out_ode = jsmfsb.sim_time_series(key, mod.m, 0, 20, 0.05, step_ode)
    return out_ssa, out_ode, mod.n

k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)

volumes = [0.1, 10]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

for i, omega in enumerate(volumes):
    key = k1 if i == 0 else k2
    data_ssa, data_ode, names = simulate_lv(omega, key)
    t = jnp.linspace(0, 20, data_ssa.shape[0])
    
    # SSA
    axes[i].step(t, data_ssa[:, 0], color='C0', label='Prey (SSA)', where='post')
    axes[i].step(t, data_ssa[:, 1], color='C1', label='Predator (SSA)', where='post')

    # ODE 
    axes[i].plot(t, data_ode[:, 0], color='C0', alpha=0.5,linewidth=2, label='Prey (ODE)')
    axes[i].plot(t, data_ode[:, 1], color='C1', alpha=0.5,linewidth=2, label='Predator (ODE)')

    if i == 0:
        axes[i].set_title(f"Stochastic Dynamics ($\Omega = 1.0$)", fontweight='bold')
    else:
        axes[i].set_title(f"Stochastic Dynamics ($\Omega = 100$)", fontweight='bold')
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Molecule Count")
    
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.show()
# %%
sns.set_theme(style="ticks")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif'
})

lvmod_sh = """
@model:3.1.1=LotkaVolterra
@compartments
 Cell
@species
 Cell:Prey=50
 Cell:Predator=100
@reactions
 @r=PreyReproduction
  Prey -> 2Prey
  1.0*Prey
 @r=PredatorPreyInteraction
  Prey+Predator -> 2Predator
  0.005*Prey*Predator
 @r=PredatorDeath
  Predator ->
  0.6*Predator
"""
lvmod = jsmfsb.shorthand_to_spn(lvmod_sh)
step_ssa = lvmod.step_gillespie()

def run_ensemble(num_sims, t_max, dt, key):
    def sim_one(k):
        return jsmfsb.sim_time_series(k, lvmod.m, 0, t_max, dt, step_ssa)
    
    keys = jax.random.split(key, num_sims)
    return jax.vmap(sim_one)(keys)

k0 = jax.random.key(42)
num_trajectories = 1000
t_max = 20
dt = 0.05
# ensemble shape: [1000, 401, 2] -> [Sims, Timepoints, Species]
ensemble = run_ensemble(num_trajectories, t_max, dt, k0)
times = jnp.linspace(0, t_max, ensemble.shape[1])

fig0, ax = plt.subplots(figsize=(14, 6))

# Probability Cloud
for i in range(50):
    ax.plot(times, ensemble[i, :, 0], color='C0', alpha=0.1, linewidth=0.5)
    ax.plot(times, ensemble[i, :, 1], color='C1', alpha=0.1, linewidth=0.5)

ax.step(times, ensemble[0, :, 0], color='C0', label='Prey Realisation', alpha=0.8, where='post')
ax.step(times, ensemble[0, :, 1], color='C1', label='Predator Realisation', alpha=0.8, where='post')

# ax.set_title("Stochastic Trajectories (The 'Flow' of Probability)", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Molecule Count")
ax.legend(loc='upper right', frameon=False)

# CME
fig1, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
snapshot_idxs = [1, 100, 400]

for i in range(3):
    prey_counts = ensemble[:, snapshot_idxs[i], 0]
    pred_counts = ensemble[:, snapshot_idxs[i], 1]
    t = int(times[snapshot_idxs[i]])
    
    sns.kdeplot(prey_counts, ax=axes[i], fill=True, color='C0', label='Prey $P(x, t)$', alpha=0.5)
    sns.kdeplot(pred_counts, ax=axes[i], fill=True, color='C1', label='Predator $P(x, t)$', alpha=0.5)

    axes[i].set_title(f"$t = {t}$s", fontsize=12)
    axes[i].set_xlabel("Molecule Count")
    
    if i > 0:
        axes[i].set_ylabel("")
    else:
        axes[i].set_ylabel("Density")

# fig1.suptitle("Temporal Evolution of the Probability Landscape", fontweight='bold', fontsize=14, y=1.02)

handles, labels = axes[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False)

sns.despine()
plt.tight_layout()
plt.show()

# %%
k_single = jax.random.key(123)
dt_fine = 0.0001  
t_max = 0.5   

step_ssa = lvmod.step_gillespie()
raw_path = jsmfsb.sim_time_series(k_single, lvmod.m, 0, t_max, dt_fine, step_ssa)

times_fine = jnp.linspace(0, t_max, raw_path.shape[0])
prey_raw = raw_path[:, 0]

fig, ax = plt.subplots(figsize=(12, 6))

ax.step(times_fine, prey_raw, where='post', color='C0', linewidth=2.5, label='Prey Count $X(t)$')

ax.set_xlim(0.1, 0.2) 
ax.set_ylim(jnp.min(prey_raw[1000:2000])-5, jnp.max(prey_raw[1000:2000])+5)

# ax.set_title("Wait-and-Jump Dynamics: Exact SSA Realisation", fontweight='bold', fontsize=14)

ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Molecule Count", fontsize=12)

sns.despine()
plt.tight_layout()
jump_indices = np.where(np.diff(prey_raw) != 0)[0]

visible_jumps = [i for i in jump_indices if 0.12 < times_fine[i] < 0.18]

if len(visible_jumps) >= 2:
    idx_prev_jump = visible_jumps[0]
    idx_curr_jump = visible_jumps[1]
    
    t_start = times_fine[idx_prev_jump]
    t_end = times_fine[idx_curr_jump]
    y_level = prey_raw[idx_curr_jump]      
    y_next = prey_raw[idx_curr_jump + 1] 
    
    ax.annotate('', xy=(t_end, y_level), xytext=(t_start, y_level),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((t_start + t_end)/2, y_level + 0.3, r'$\tau$', 
            ha='center', va='bottom', fontsize=14)

    ax.annotate('', xy=(t_end, y_next), xytext=(t_end, y_level),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5, mutation_scale=15))
    ax.text(t_end + 0.001, (y_level + y_next)/2, r'$\nu_j$', 
            color='red', va='center', fontweight='bold', fontsize=14)

plt.show()
# %%
def make_step_tau(mod):
    V = (jnp.array(mod.post) - jnp.array(mod.pre)).T 

    def step_tau(key, x, t, dt):
        h_vector = mod.h(x, t)
        rates = h_vector * dt
        k_reactions = jrandom.poisson(key, rates)
        
        x_new = x + jnp.dot(V, k_reactions)
        
        return jnp.maximum(x_new, 0)

    return step_tau

k_master = jax.random.key(88)
k1, k2 = jax.random.split(k_master)
t_end = 5

# 1. Gillespie 
step_ssa = lvmod.step_gillespie()
path_ssa = jsmfsb.sim_time_series(k1, lvmod.m, 0, t_end, 0.01, step_ssa)

# 2. Tau-Leaping 
step_tau = make_step_tau(lvmod)
path_tau = jsmfsb.sim_time_series(k2, lvmod.m, 0, t_end, 0.1, step_tau)

# Plot
plt.figure(figsize=(12, 6))
t_ssa = jnp.linspace(0, t_end, path_ssa.shape[0])
t_tau = jnp.linspace(0, t_end, path_tau.shape[0])

plt.plot(t_ssa, path_ssa[:, 0], label="Exact SSA (Gillespie)", alpha=0.5, color='C0')
plt.step(t_tau, path_tau[:, 0], where='post', label=r"Tau-Leaping ($\Delta t=0.1$)", 
         color='C1', linestyle='--', linewidth=2)

# plt.title("Statistical Comparison: Exact SSA vs. Tau-Leaping Approximation", fontweight='bold')
plt.xlabel("Time (s)")
plt.ylabel("Prey Population")
plt.legend()
sns.despine()
plt.show()
# %%
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

def plot_wiener_paths(num_paths=3, t_max=1.0, dt=0.001):
    key = jax.random.key(667)
    steps = int(t_max / dt)
    times = jnp.linspace(0, t_max, steps)
    keys = jax.random.split(key, num_paths)
    
    plt.figure(figsize=(10, 5))
    for k in keys:
        increments = jax.random.normal(k, (steps,)) * jnp.sqrt(dt)
        path = jnp.cumsum(increments)
        plt.plot(times, path, linewidth=1.5)

    plt.xlabel("Time $t$")
    plt.ylabel("$W(t)$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.show()

plot_wiener_paths()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import seaborn as sns

# Set the means to demonstrate convergence (Low vs High activity)
means = [1, 10, 20]
colors = ['C0', 'C1', 'C2']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for i, lam in enumerate(means):
    # Poisson (Discrete)
    x_pois = np.arange(0, lam * 3)
    axes[i].stem(x_pois, poisson.pmf(x_pois, lam), linefmt=colors[i], 
                 markerfmt=colors[i]+'o', basefmt=" ", label=f'Poisson($\lambda={lam}$)')
    
    # Gaussian (Continuous Approximation)
    x_norm = np.linspace(0, lam * 3, 100)
    axes[i].plot(x_norm, norm.pdf(x_norm, lam, np.sqrt(lam)), 
                 color='black', lw=2, label=f'Gaussian($\mu={lam}, \sigma=\sqrt{{{lam}}}$)')
    
    axes[i].set_title(f"$\lambda = {lam}$", fontweight='bold')
    axes[i].legend(fontsize=9)
    sns.despine(ax=axes[i])

plt.tight_layout()
plt.show()
# %%
import time
lv_sh = """
    @model:3.1.1=LotkaVolterra
    s=item, t=second, v=litre, e=item
    @compartments
    Cell
    @species
    Cell:Prey=500 s
    Cell:Predator=1000 s
    @reactions
    @r=PreyReproduction
    Prey -> 2Prey
    c1*Prey : c1=1
    @r=PredatorPreyInteraction
    Prey+Predator -> 2Predator
    c2*Prey*Predator : c2=0.0005
    @r=PredatorDeath
    Predator ->
    c3*Predator : c3=0.6
    """
lvmod = jsmfsb.shorthand_to_spn(lv_sh)
k_master = jax.random.key(42)
k_ssa, k_tau, k_cle = jax.random.split(k_master, 3)
t_end = 50
dt = 0.05 

# Define the steps
step_ssa = lvmod.step_gillespie()
step_tau = make_step_tau(lvmod) 
step_cle = lvmod.step_cle()

def timed_simulation(key, step_func, dt_val):
    # Warm-up (Compilation)
    _ = jsmfsb.sim_time_series(key, lvmod.m, 0, t_end, dt_val, step_func)
    
    # Timed run
    start = time.perf_counter()
    path = jsmfsb.sim_time_series(key, lvmod.m, 0, t_end, dt_val, step_func)
    path.block_until_ready() # Important: waits for JAX to actually finish
    end = time.perf_counter()
    
    return path, end - start

# 2. Run and Time
path_ssa, dur_ssa = timed_simulation(k_ssa, step_ssa, 0.01) # SSA needs smaller steps
path_tau, dur_tau = timed_simulation(k_tau, step_tau, dt)
path_cle, dur_cle = timed_simulation(k_cle, step_cle, dt)

# 3. Print Results
print(f"{'Method':<15} | {'Execution Time (s)':<20}")
print("-" * 40)
print(f"{'Exact SSA':<15} | {dur_ssa:.5f}")
print(f"{'Tau-Leaping':<15} | {dur_tau:.5f}")
print(f"{'CLE':<15} | {dur_cle:.5f}")

# 4. Plotting
fig, ax = plt.subplots(figsize=(12, 6))
t_ssa = jnp.linspace(0, t_end, path_ssa.shape[0])
t_tau = jnp.linspace(0, t_end, path_tau.shape[0])
t_cle = jnp.linspace(0, t_end, path_cle.shape[0])

ax.plot(t_ssa, path_ssa[:, 0], label=f"SSA ({dur_ssa:.3f}s)", alpha=0.3, color='black')
ax.plot(t_tau, path_tau[:, 0], label=f"Tau-Leaping ({dur_tau:.3f}s)", ls='--', color='C1')
ax.plot(t_cle, path_cle[:, 0], label=f"CLE ({dur_cle:.3f}s)", color='C2', lw=2)

# ax.set_title("Performance & Accuracy: Stochastic Simulation Hierarchy", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Prey Population")
ax.legend()
sns.despine()
plt.show()
# %%
def plot_1d_diffusion():
    # Simulation parameters
    nx = 50  # number of voxels
    x = np.arange(nx)
    
    # Generate 'Noisy' Stochastic Data (RDME-style)
    # A central pulse with Poisson noise
    base = 100 * np.exp(-0.02 * (x - 25)**2)
    stochastic_counts = np.random.poisson(base)
    
    # Generate 'Smooth' Deterministic Data (PDE-style)
    deterministic_curve = base

    plt.figure(figsize=(10, 5))
    
    # Plotting the discrete voxels (RDME)
    plt.step(x, stochastic_counts, where='mid', label='RDME (Stochastic Voxels)', color='C0', alpha=0.8)
    plt.fill_between(x, stochastic_counts, step="mid", alpha=0.2, color='C0')
    
    # Plotting the smooth limit (Spatial CLE/PDE)
    plt.plot(x, deterministic_curve, label='Deterministic Limit (PDE)', color='black', lw=2, linestyle='--')

    # plt.title("1D Spatial Diffusion: Discrete Voxels vs. Continuous Limit", fontweight='bold')
    plt.xlabel("Spatial Coordinate (Voxel Index)")
    plt.ylabel("Molecule Count")
    plt.legend()
    sns.despine()
    plt.show()

plot_1d_diffusion()
#  %%
coords = ""
for t, x in zip(time_axis[::4], out[::4, 0]):
    coords += f"({t:.2f}, {x:.2f}) "
print(coords)
# %%

ID_shorthand = """
@model:3.1.1=ImmigrationDeath
 s=item, t=second, v=litre, e=item
@compartments
 Cell
@species
 Cell:X=0 s
@reactions
@r=Immigration
 -> X
 lam : lam=10
@r=Death
 X ->
 mu*X : mu=0.1
"""

idmod = jsmfsb.shorthand_to_spn(ID_shorthand)

step = idmod.step_gillespie()
k0 = jax.random.key(42)

t_start = 0
t_end = 50 
dt = 0.1

out = jsmfsb.sim_time_series(k0, idmod.m, t_start, t_end, dt, step)
time_axis = np.linspace(t_start, t_end, out.shape[0])

fig, ax = plt.subplots(figsize=(6, 4)) 

ax.plot(time_axis, out[:, 0], color='red', linewidth=1.5, alpha=0.8, label='Stochastic (SSA)')

analytical_sol = 100 * (1 - np.exp(-0.1 * time_axis))
ax.plot(time_axis, analytical_sol, color='blue', linestyle='--', linewidth=2, label='Deterministic (ODE)')

ax.set_xlabel("Time ($t$)")
ax.set_ylabel("Molecule Count ($X$)")
ax.set_title("Immigration-Death Process")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig("immigration_death_simulation.pdf")
plt.show()
# %%
dimermod = jsmfsb.models.dimer()
step = dimermod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, dimermod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, dimermod.m, 0, 30, 0.1, step)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(dimermod.n)
# %%
idmod = jsmfsb.models.id()
step = idmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, idmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, idmod.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(1):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(idmod.n)
# %%
mmmod = jsmfsb.models.mm()
step = mmmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, mmmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, mmmod.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(4):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(mmmod.n)
# %% LV Pred-Prey Reaction-Diffusion System 1D (20 subvolumes, T=30, Step_Gillespie) -----------------------------------
start = time.time()
N = 20
T = 30
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(N / 2)].set(lv.m)
k0 = jax.random.key(42)
print(k0)
k_sim1, k_sim2 = jax.random.split(k0)
print(k_sim1)
step_lv_1d = lv.step_gillespie_1d(jnp.array([0.6, 0.6]))
x1 = step_lv_1d(k_sim1, x0, 0, 1)
out = jsmfsb.sim_time_series_1d(k_sim1, x0, 0, T, 0.1, step_lv_1d, True)

fig, axis = plt.subplots(2,2, figsize=(12,8))
fig.suptitle("Step_Gillespie", fontsize=16)
for i in range(2):
    axis[0, i].imshow(out[i, :, :], aspect='auto')
    axis[0, i].set_title(f"{lv.n[i]} (V=20)")
    axis[0, i].set_ylabel("Space")

# LV Pred-Prey Reaction-Diffusion System 1D (40 Subvolumes, T=30, Step_Gillespie) -----------------------------------
N = 40
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv( th=[1, 0.01, 0.6]) # 2x 2nd-order Reaction Rate Constant: Default 0.005 -> 0.01
x0 = x0.at[:, int(N / 2)].set(lv.m)
print(k_sim2)
step_lv_1d = lv.step_gillespie_1d(jnp.array([2.4, 2.4])) # 20 -> 40 subvols means 4x Diffusion Rate Constant
x1 = step_lv_1d(k_sim2, x0, 0, 1)
out = jsmfsb.sim_time_series_1d(k_sim2, x0, 0, T,0.1, step_lv_1d, True) 

for i in range(2):
    axis[1, i].imshow(out[i, :, :], aspect='auto')
    axis[1, i].set_title(f"{lv.n[i]} (V=40)")
    axis[1, i].set_xlabel("Time")
    axis[1, i].set_ylabel("Space")
end = time.time()
print(f"Time to run simulation: {end - start} seconds!")
plt.tight_layout()
# %%  LV Pred-Prey Reaction-Diffusion System 1D (20 Subvolumes, T=30, approximate using Step_CLE)
start = time.time()
N = 20
T = 30
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(N / 2)].set(lv.m)
k0 = jax.random.key(42)
print(k0)
k_sim1, k_sim2 = jax.random.split(k0)
print(k_sim1)
step_lv_1d = lv.step_cle_1d(jnp.array([0.6, 0.6]))
x1 = step_lv_1d(k_sim1, x0, 0, 1)
out = jsmfsb.sim_time_series_1d(k_sim1, x0, 0, T, 0.1, step_lv_1d, True) # Smaller dt

fig, axis = plt.subplots(2,2, figsize=(12,8))
fig.suptitle("Approx using Step_CLE", fontsize=16)
for i in range(2):
    axis[0, i].imshow(out[i, :, :], aspect='auto')
    axis[0, i].set_title(f"{lv.n[i]} (V=20)")
    axis[0, i].set_ylabel("Space")

# LV Pred-Prey Reaction-Diffusion System 1D (40 Subvolumes, T=30, approx using Step_CLE) -----------------------------------
N = 80
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv(th=[1, 0.02, 0.6]) # 4x 2nd-order Reaction Rate Constant: Default 0.005 -> 0.04
x0 = x0.at[:, int(N / 2)].set(lv.m)
print(k_sim2)
step_lv_1d = lv.step_cle_1d(jnp.array([9.6, 9.6])) # 20 -> 80 subvols means 16x Diffusion Rate Constant (40/20)^2 = 4
x1 = step_lv_1d(k_sim2, x0, 0, 1)
out = jsmfsb.sim_time_series_1d(k_sim2, x0, 0, T, 0.1, step_lv_1d, True) 

for i in range(2):
    axis[1, i].imshow(out[i, :, :], aspect='auto')
    axis[1, i].set_title(f"{lv.n[i]} (V=40)")
    axis[1, i].set_xlabel("Time")
    axis[1, i].set_ylabel("Space")

end = time.time()
print(f"Time to run simulation: {end - start} seconds!")
plt.tight_layout()
# %% LV Pred-Prey Reaction-Diffusion System 1D (80 Subvolumes, T=30, approx using Step_CLE) -----------------------------------
start = time.time()
N = 80
T = 30
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv() # default: th=[1, 0.005, 0.6] Inverse scale?
x0 = x0.at[:, int(N / 2)].set(lv.m)
step_lv_1d = lv.step_cle_1d(jnp.array([9.6, 9.6]), dt = 0.01) # (80/20)^2 = 16 then 16 x 0.6 = 9.6, change dt here??
k0 = jax.random.key(42)

out = jsmfsb.sim_time_series_1d(k0, x0, 0, T, 0.1, step_lv_1d, True)
# print(out)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(out[i, :, :], aspect='auto')
    axis.set_title(lv.n[i])

end = time.time()
print(f"Time to run simulation: {end - start} seconds!")
plt.tight_layout()
# %% LV Pred-Prey Reaction-Diffusion System 2D (20 x 30 grid, T=30, approx using Step_CLE_2D) -----------------------------------
M = 80
N = 120

x0 = jnp.zeros((2, M, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)
step_lv_2d = lv.step_cle_2d(jnp.array([0.6, 0.6]), dt = 0.01)
k0 = jax.random.key(42)
for T in range(10):
    x1 = step_lv_2d(k0, x0, 0, T)

    fig, axis = plt.subplots(1,2)
    for i in range(2):
        axis[i].imshow(x1[i, :, :])
        axis[i].set_title(lv.n[i])
# %% LV Pred-Prey Reaction-Diffusion System 2D (200 x 250 grid, T=30, approx using Step_CLE_2D) -----------------------------------
import imageio as iio

M = 200
N = 250
frames = []
x0 = jnp.zeros((2, M, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)
step_lv_2d = lv.step_cle_2d(jnp.array([0.6, 0.6]), dt = 0.1)
k0 = jax.random.key(42)
for T in range(30):
    x1 = step_lv_2d(k0, x0, 0, T)
    print(x1)
    fig, axis = plt.subplots(1,2)
    for i in range(2):
        axis[i].imshow(x1[i, :, :])
        axis[i].set_title(lv.n[i])
    fig.savefig(f"LV_Test{T:05d}.png")
    im = iio.imread(f"LV_Test{T:05d}.png")
    frames.append(im)

iio.mimsave("LV.gif", frames)

# %% SIR Epidemic Model

sir = jsmfsb.models.sir()
step_sir = sir.step_gillespie()
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series(k0, sir.m, 0, 50, 0.05, step_sir)

fig, axis = plt.subplots()
for i in range(3):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(sir.n)

# %% SIR Epidemic Model 1D (Gillespie)

"""
@model:3.1.1=SIR "SIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=100 s
 Pop:I=5 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> 2I
 beta*S*I : beta=0.1
@r=Removal
 I -> R
 gamma*I : gamma=0.25
 """

N = 40
T = 60
S = 100
x0 = jnp.zeros((3, N))

sir = jsmfsb.models.sir(th=[0.1,0.25])

avg_S_per_box = round(S / N)

x0 = x0.at[0, :].set(avg_S_per_box) # Uniformly distributed in space
x0 = x0.at[1, int(N/2)].set(5) # 5 infected in the middle
print(x0)

k0 = jax.random.key(42)
step_sir_1d = sir.step_gillespie_1d(jnp.array([0.1, 0.1, 0.1]))

print(step_sir_1d(k0,x0,0,2))
out = jsmfsb.sim_time_series_1d(k0, x0, 0, T, 1, step_sir_1d, True)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
for i in range(3):
    axes[i].imshow(out[i,:,:])
    axes[i].set_title(sir.n[i])

axes[2].set_xlabel("Time")
plt.show()
# %% SIR Epidemic Model 1D (CLE)

N = 15
T = 10
S = 100
x0 = jnp.zeros((3, N))

sir = jsmfsb.models.sir(th=[0.1,0.5])

avg_S_per_box = round(S / N)

x0 = x0.at[0, :].set(avg_S_per_box) # Uniformly distributed in space
x0 = x0.at[1, int(N/2)].set(5) # 5 infected in the middle
print(x0)

k0 = jax.random.key(42)
step_sir_1d = sir.step_cle_1d(jnp.array([0.1, 0.1, 0.1]))

print(step_sir_1d(k0,x0,0,2))
out = jsmfsb.sim_time_series_1d(k0, x0, 0, T, 1, step_sir_1d, True)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
for i in range(3):
    axes[i].imshow(out[i,:,:])
    axes[i].set_title(sir.n[i])

axes[2].set_xlabel("Time")
plt.show()

# %% SIR Epidemic Model 2D (CLE) ----------------------------------------------------------------------------------------
import imageio as iio
from pathlib import Path 
import shutil

base_path = Path("/Users/josh/Documents/SMFSB Python")
save_folder = base_path / "SIR_2D_CLE_Frames"
if save_folder.is_dir():
    print(f"Deleting existing folder: {save_folder}")
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Frames will be saved in: {save_folder}")

k0 = jax.random.key(42)
S_count_key, sim_key= jax.random.split(k0, 2)

N = 100
M = 100
T = 10
S = 10000000
dt = 0.1

n_steps = int(T/dt)
x0 = jnp.zeros((3, N, M))
sir = jsmfsb.models.sir(th=[0.01,0.5])

num_boxes = N * M 

flat_indices = jax.random.choice(S_count_key, num_boxes, shape=(S,))

S_counts_flat = jnp.bincount(flat_indices, length=num_boxes)

S_counts_grid = S_counts_flat.reshape((N, M))

x0 = x0.at[0, :, :].set(S_counts_grid) 
x0 = x0.at[1, int(N/2), int(M/2)].set(sir.m[1]) # Set no. of Infected to the centre

step_sir_2d_cle= sir.step_cle_2d(
    d=jnp.array([0.1, 0.1, 0.1]),
    dt=0.01
)
print(f"Running JAX simulation for {n_steps} steps...")
start = time.time()
out = jsmfsb.sim_time_series_2d(sim_key, x0, 0, T, dt, step_sir_2d_cle, False)
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

final_state_array = out[:, :, :, -1] # Extract the last state array

final_S_count = jnp.sum(final_state_array[0])
final_I_count = jnp.sum(final_state_array[1])
final_R_count = jnp.sum(final_state_array[2]) 

print("\n--- INITIAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {jnp.sum(x0[0]):.0f}")
print(f"  Total Infected (I):    {jnp.sum(x0[1]):.0f}")
print(f"  Total Recovered (R):   {jnp.sum(x0[2]):.0f}")
print(f"  Total (S+I+R):   {jnp.sum(x0[0])+jnp.sum(x0[1])+jnp.sum(x0[2]):.0f}")
print("--------------------------------\n")

for i in range(100):
    print(jnp.sum(out[0,:,:,i])) # S Count starts to increase due to noise


print("\n--- FINAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {final_S_count:.0f}")
print(f"  Total Infected (I):    {final_I_count:.0f}")
print(f"  Total Recovered (R):   {final_R_count:.0f}")
print(f"  Total (S+I+R):   {final_S_count+final_I_count+final_R_count:.0f}")
print("--------------------------------\n")

print("Saving frames...")
start_save = time.time()
for i in range(out.shape[3]):  

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True)

    for j in range(3):
        
        axes[j].imshow(out[j, :, :, i]) # out = (species, x, y, time)
        axes[j].set_title(sir.n[j])

    filename = f"SIR_CLE_Frames_{i:05d}.png"
    
    save_path = save_folder / filename
    
    plt.savefig(save_path)
    plt.close(fig)

end_save = time.time()
print(f"Frames saved in {end_save - start_save:.2f} seconds.")

print("Compiling GIF...")
file_list = sorted(save_folder.glob("*.png"))
frames = [iio.imread(file) for file in file_list]

gif_path = base_path / "SIR_2D_Outbreak_CLE.gif"
iio.mimsave(
    gif_path, frames
    )

print(f"GIF saved to {gif_path}")
# %% SIR Epidemic Model 2D (Gillespie) ----------------------------------------------------------------------------------------
import imageio as iio
from pathlib import Path 
import shutil

base_path = Path("/Users/josh/Documents/SMFSB Python")
save_folder = base_path / "SIR_2D_Gillespie_Frames"
if save_folder.is_dir():
    print(f"Deleting existing folder: {save_folder}")
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Frames will be saved in: {save_folder}")

k0 = jax.random.key(42)

N = 50
M = 50
T = 100
S = 50000
dt = 0.1

n_steps = int(T/dt)
x0 = jnp.zeros((3, N, M))
sir = jsmfsb.models.sir(th=[0.3,0.3])

num_boxes = N * M 

flat_indices = jax.random.choice(k0, num_boxes, shape=(S,))

S_counts_flat = jnp.bincount(flat_indices, length=num_boxes)

S_counts_grid = S_counts_flat.reshape((N, M))

x0 = x0.at[0, :, :].set(S_counts_grid) 
x0 = x0.at[1, int(N/2), int(M/2)].set(5)
x0 = x0.at[2, :, :].set(0)

step_sir_2d_gillespie = sir.step_gillespie_2d(
    d=jnp.array([0.1, 0.1, 0.1])
)
print(f"Running JAX simulation for {n_steps} steps...")
start = time.time()
out = jsmfsb.sim_time_series_2d(k0, x0, 0, T, dt, step_sir_2d_gillespie, False)
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

final_state_array = out[:, :, :, -1] # Extract the last state array

initial_S_count = jnp.sum(x0[0])
initial_I_count = jnp.sum(x0[1])
initial_R_count = jnp.sum(x0[2])
final_S_count = jnp.sum(final_state_array[0])
final_I_count = jnp.sum(final_state_array[1])
final_R_count = jnp.sum(final_state_array[2]) 

print("\n--- INITIAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {initial_S_count:.0f}")
print(f"  Total Infected (I):    {initial_I_count:.0f}")
print(f"  Total Recovered (R):   {initial_R_count:.0f}")
print(f"  Total (S+I+R):   {initial_S_count+initial_I_count+initial_R_count:.0f}")
print("--------------------------------\n")

print("\n--- FINAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {final_S_count:.0f}")
print(f"  Total Infected (I):    {final_I_count:.0f}")
print(f"  Total Recovered (R):   {final_R_count:.0f}")
print(f"  Total (S+I+R):   {final_S_count+final_I_count+final_R_count:.0f}")
print("--------------------------------\n")

# print("Saving frames...")
# start_save = time.time()
# for i in range(out.shape[3]):  

#     fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True)

#     for j in range(3):
        
#         axes[j].imshow(out[j, :, :, i]) # out = (species, x, y, time)
#         axes[j].set_title(sir.n[j])

#     filename = f"SIR_Gillespie_Frames_{i:05d}.png"
    
#     save_path = save_folder / filename
    
#     plt.savefig(save_path)
#     plt.close(fig)

# end_save = time.time()
# print(f"Frames saved in {end_save - start_save:.2f} seconds.")

# print("Compiling GIF...")
# file_list = sorted(save_folder.glob("*.png"))
# frames = [iio.imread(file) for file in file_list]

# gif_path = base_path / "SIR_2D_Outbreak_Gillespie.gif"
# iio.mimsave(
#     gif_path, frames
#     )

# print(f"GIF saved to {gif_path}")
# %% Poisson Timestep Implementation 

lv = jsmfsb.models.lv()
print(lv)
step_lv = lv.step_poisson()
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series(k0, lv.m, 0, 100, 0.1, step_lv)


fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)

# %% SIR Epidemic Model 2D (Poisson Timestep) ----------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

def step_poisson_2d(model, d, dt=0.01):
    """
    Create a function for advancing the state of an SPN on a 2D regular grid
    using a Poisson time-stepping method (Tau-leaping).

    Parameters
    ----------
    d : array
        A vector of diffusion coefficients - one coefficient for each
        reacting species, in order. The coefficient is the reaction
        rate for a molecule moving into an adjacent
        compartment. The hazard for a given molecule leaving the
        compartment is therefore four times this value (as it can leave
        in one of 4 directions).
    dt : float
        The time step for the time-stepping integration method. Defaults to 0.01.

    Returns
    -------
    A function which can be used to advance the state of the SPN
    model by using a Poisson time stepping method with step size
    ‘dt’. The function closure has interface
    ‘function(key, x0, t0, deltat)’, where ‘x0’ is a 3d array with 
    indices species, then rows and columns corresponding to voxels, 
    representing the initial condition, ‘t0’ represents the
    initial time, and ‘deltat’ represents the amount of time
    by which the process should be advanced. The function closure
    returns a 3d array representing the simulated state of the system at
    the new time.
        
    """
    sto = (model.post - model.pre).T
    D = d.reshape(-1, 1, 1)

    def left(a):
        return jnp.roll(a, -1, axis=1)

    def right(a):
        return jnp.roll(a, +1, axis=1)

    def up(a):
        return jnp.roll(a, -1, axis=2)

    def down(a):
        return jnp.roll(a, +1, axis=2)

    def laplacian(a):
        return left(a) + right(a) + up(a) + down(a) - 4 * a

    def rectify(a):
        return jnp.where(a < 0, 0, a)

    def diffuse(x):
        return (D * laplacian(x)) * dt
    
    h_grid = jax.vmap(jax.vmap(model.h))

    def react(key, x, t):
        """
        Discrete Reaction step (Poisson).
        """
        # A. Transpose x to (Width, Height, Species) for our h_grid function
        x_transposed = jnp.transpose(x, (1, 2, 0))
        
        # B. Calculate Rates for the whole grid at once
        rates = h_grid(x_transposed, t) # Shape: (Width, Height, Reactions)
        
        # C. Sample Poisson Events (The core difference from CLE)
        # "How many times did this reaction happen in dt?"
        events = jax.random.poisson(key, rates * dt)
        
        # D. Update Species (The einsum replacement)
        # events: (Width, Height, Reactions)
        # sto:    (Species, Reactions)
        # We want: (Width, Height, Species)
        dx_transposed = jnp.einsum('ijr,sr->ijs', events, sto)
        
        # E. Transpose back to (Species, Width, Height)
        return jnp.transpose(dx_transposed, (2, 0, 1))
    
    @jit
    def step(key, x0, t0, deltat):
        steps = jnp.ceil(deltat / dt).astype(int)
        
        def scan_body(carry, _):
            key, x, t = carry
            key, k_react = jax.random.split(key)
            
            dx_diff = diffuse(x)
            dx_react = react(k_react, x, t)
            
            x = x + dx_diff + dx_react
            x = rectify(x)
            t = t + dt
            return (key, x, t), None

        final_state, _ = jax.lax.scan(scan_body, (key, x0, t0), length=steps)
        return final_state[1]

    step = jax.jit(step, static_argnums=(3,))
    return step

import imageio as iio
from pathlib import Path 
import shutil

base_path = Path("/Users/josh/Documents/SMFSB Python")
save_folder = base_path / "SIR_2D_PTS_Frames"
if save_folder.is_dir():
    print(f"Deleting existing folder: {save_folder}")
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Frames will be saved in: {save_folder}")

k0 = jax.random.key(42)
S_count_key, sim_key= jax.random.split(k0, 2)

N = 10
M = 10
T = 10
S = 100000
dt = 0.1

n_steps = int(T/dt)
x0 = jnp.zeros((3, N, M))
sir = jsmfsb.models.sir(th=[0.01,0.5])

num_boxes = N * M 

flat_indices = jax.random.choice(S_count_key, num_boxes, shape=(S,))

S_counts_flat = jnp.bincount(flat_indices, length=num_boxes)

S_counts_grid = S_counts_flat.reshape((N, M))

x0 = x0.at[0, :, :].set(S_counts_grid) 
x0 = x0.at[1, int(N/2), int(M/2)].set(sir.m[1]) # Set no. of Infected to the centre

step_sir_2d_poisson= step_poisson_2d(sir,
    d=jnp.array([0.1, 0.1, 0.1]),
    dt=0.01
)
print(f"Running JAX simulation for {n_steps} steps...")
start = time.time()
out = jsmfsb.sim_time_series_2d(sim_key, x0, 0, T, dt, step_sir_2d_poisson, False)
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

final_state_array = out[:, :, :, -1] # Extract the last state array

final_S_count = jnp.sum(final_state_array[0])
final_I_count = jnp.sum(final_state_array[1])
final_R_count = jnp.sum(final_state_array[2]) 

print("\n--- INITIAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {jnp.sum(x0[0]):.0f}")
print(f"  Total Infected (I):    {jnp.sum(x0[1]):.0f}")
print(f"  Total Recovered (R):   {jnp.sum(x0[2]):.0f}")
print(f"  Total (S+I+R):   {jnp.sum(x0[0])+jnp.sum(x0[1])+jnp.sum(x0[2]):.0f}")
print("--------------------------------\n")

for i in range(100):
    print(jnp.sum(out[0,:,:,i])) # S Count starts to increase due to noise


print("\n--- FINAL SIMULATION STATE ---")
print(f"  Total Susceptible (S): {final_S_count:.0f}")
print(f"  Total Infected (I):    {final_I_count:.0f}")
print(f"  Total Recovered (R):   {final_R_count:.0f}")
print(f"  Total (S+I+R):   {final_S_count+final_I_count+final_R_count:.0f}")
print("--------------------------------\n")

print("Saving frames...")
start_save = time.time()
for i in range(out.shape[3]):  

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10), sharex=True)

    for j in range(3):
        
        axes[j].imshow(out[j, :, :, i]) # out = (species, x, y, time)
        axes[j].set_title(sir.n[j])

    filename = f"SIR_PTS_Frames_{i:05d}.png"
    
    save_path = save_folder / filename
    
    plt.savefig(save_path)
    plt.close(fig)

end_save = time.time()
print(f"Frames saved in {end_save - start_save:.2f} seconds.")

print("Compiling GIF...")
file_list = sorted(save_folder.glob("*.png"))
frames = [iio.imread(file) for file in file_list]

gif_path = base_path / "SIR_2D_Outbreak_PTS.gif"
iio.mimsave(
    gif_path, frames
    )

print(f"GIF saved to {gif_path}")
# %%
# %% BENCHMARKING
import time

lv = jsmfsb.models.lv()
k0 = jax.random.key(42)
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
results['SSA_2d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_2d(jnp.array([0.6,0.6])), '2d', "SSA_2d_20")
results['Euler_2d_20'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_2d(jnp.array([0.6,0.6])),     '2d', "Euler_2d_20")
results['CLE_2d_20']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_2d(jnp.array([0.6,0.6])),       '2d', "CLE_2d_20")

print("\n=== 2D 50x50 ===")
x0 = jnp.zeros((2, 50, 50)).at[:, 25, 25].set(lv.m)
results['SSA_2d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_gillespie_2d(jnp.array([0.6,0.6])), '2d', "SSA_2d_50")
results['Euler_2d_50'] = timed_sim(k0, x0, 20, 0.1, lv.step_euler_2d(jnp.array([0.6,0.6])),     '2d', "Euler_2d_50")
results['CLE_2d_50']   = timed_sim(k0, x0, 20, 0.1, lv.step_cle_2d(jnp.array([0.6,0.6])),       '2d', "CLE_2d_50")

print("\n=== RESULTS ===")
print(f"{'Configuration':<20} | {'Time (s)':<10}")
print("-" * 35)
for k, v in results.items():
    val = f"{v:.4f}" if v is not None else "FAILED"
    print(f"{k:<20} | {val}")

# %% BENCHMARKS
config_order = [
    'Non-spatial', '1D (N=20)', '1D (N=50)',
    '2D (20×20)', '2D (50×50)', '2D (100×100)'
]

df = pd.DataFrame({
    'Configuration': [
        'Non-spatial', 'Non-spatial', 'Non-spatial',
        '1D (N=20)', '1D (N=20)', '1D (N=20)',
        '1D (N=50)', '1D (N=50)', '1D (N=50)',
        '2D (20×20)', '2D (20×20)', '2D (20×20)',
        '2D (50×50)', '2D (50×50)', '2D (50×50)',
        '2D (100×100)', '2D (100×100)', '2D (100×100)',
    ],
    'Method': [
        'SSA', 'Euler', 'CLE',
        'SSA', 'Euler', 'CLE',
        'SSA', 'Euler', 'CLE',
        'SSA', 'Euler', 'CLE',
        'SSA', 'Euler', 'CLE',
        'SSA', 'Euler', 'CLE',
    ],
    'Time (s)': [
        0.1165, 0.0322, 0.0743,
        0.8626, 0.0476, 0.1697,
        1.5789, 0.0485, 0.1792,
        73.2206, 0.0472, 0.3699,
        1040.0956, 0.0625, 1.0662,
        14560.0, 0.1188, 1.7889,
    ]
})

palette = {'SSA': '#4878CF', 'Euler': '#E07B39', 'CLE': '#3EAF67'}

sns.set_theme(style='ticks')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=df,
    x='Configuration', y='Time (s)',
    hue='Method',
    order=config_order,
    hue_order=['SSA', 'Euler', 'CLE'],
    palette=palette,
    ax=ax
)

ax.set_yscale('log')
ax.set_ylim(1e-2, 1e5)
ax.set_ylabel('Time (s) [log scale]')
ax.set_xlabel('Spatial Configuration')
ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

ax.patches[5].set_hatch('//')
ax.patches[5].set_edgecolor('white')

for idx, label in [(4, '17 min'), (5, '~4 hrs\n(est.)')]:
    bar = ax.patches[idx]
    x = bar.get_x() + bar.get_width() / 2
    if idx == 4:
        ax.text(x, 1.1e3, label, ha='center', va='bottom',
            fontsize=10, color='#4878CF', style='italic')
    else:
        ax.text(x, 1.6e4, label, ha='center', va='bottom',
            fontsize=10, color='#4878CF', style='italic')

solid_patch = Patch(facecolor='#4878CF', label='SSA (exact)')
hatch_patch = Patch(facecolor='#4878CF', hatch='//', edgecolor='white',
                    label='SSA (estimated)')
euler_patch = Patch(facecolor='#E07B39', label='Euler')
cle_patch   = Patch(facecolor='#3EAF67', label='CLE')

ax.legend(handles=[solid_patch, hatch_patch, euler_patch, cle_patch],
          frameon=True, title='Method')

sns.despine()
plt.tight_layout()
plt.savefig('/Users/josh/Documents/Year 3/SMFSB Python/Report Images/benchmark_timings.png', 
            bbox_inches='tight', dpi=300)
plt.show()
# %%
