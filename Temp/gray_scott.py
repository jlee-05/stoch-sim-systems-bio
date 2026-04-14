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
import qrcode

# REPLACE THIS with your actual link
website_url = "https://github.com/jlee-05/stock-sim-systems-bio"

# Generate QR
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H, # High error correction
    box_size=10,
    border=2,
)
qr.add_data(website_url)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Save it
img.save("poster_qr_code.png")
print("QR Code saved as poster_qr_code.png")

# %% GRAY-SCOTT POSTER GENERATOR
import jax
import matplotlib.pyplot as plt
import jsmfsb
import jax.numpy as jnp
import time
import shutil
from pathlib import Path

# --- 1. CONFIGURATION ---
Omega = 100
F = 0.055      # Zebra/Labyrinth parameter
k = 0.062
N = 100        # Grid Size
M = 100
T = 5000       
dt = 1.0       

# Folder to save images
save_folder = Path("Poster_Images_Final")
if save_folder.is_dir():
    shutil.rmtree(save_folder)
save_folder.mkdir()
print(f"Saving images to: {save_folder.resolve()}")

# --- 2. MODEL DEFINITION ---
c_feed = F * Omega
c_kill = F + k
c_auto = 2 / (Omega*Omega)

gs_shorthand = f"""
@model:3.1.1=GrayScott "Gray-Scott Model"
 s=item, t=second, v=litre, e=item
@species
 Cell:U=0 s
 Cell:V=0 s
@reactions
@r=Feed
 -> U
 c_feed : c_feed={c_feed}
@r=U_Removal
 U -> 
 c_kill*U : c_kill={c_kill}
@r=V_Decay
 V ->
 c_kill*V : c_kill={c_kill}
@r=Autocatalysis
 U + 2V -> 3V
 c_auto*U*0.5*V*(V-1) : c_auto={c_auto}
"""
gs_model = jsmfsb.shorthand_to_spn(gs_shorthand)

# --- 3. INITIALIZATION (Robust Block Seed) ---
x0 = jnp.zeros((2, N, M))
x0 = x0.at[0, :, :].set(Omega) # Fill U

# <--- FIX: Using a 20x20 BLOCK seed, not a single pixel
mid_n, mid_m = int(N/2), int(M/2)
seed_radius = 10 
x0 = x0.at[1, mid_n-seed_radius:mid_n+seed_radius, mid_m-seed_radius:mid_m+seed_radius].set(0.25 * Omega)
x0 = x0.at[0, mid_n-seed_radius:mid_n+seed_radius, mid_m-seed_radius:mid_m+seed_radius].set(0.5 * Omega)

# --- 4. SIMULATION ---
diff_rates = jnp.array([0.2, 0.1])
step_gs_cle_2d = gs_model.step_cle_2d(diff_rates, dt=dt)
k_jax = jax.random.key(42)

print(f"Running simulation (T={T})... This may take 10-20 seconds.")
start = time.time()
out = jsmfsb.sim_time_series_2d(k_jax, x0, 0, T, dt, step_gs_cle_2d, True)
print(f"Simulation done in {time.time() - start:.2f}s")

# --- 5. SAVE FRAMES ---
total_frames = out.shape[3]
mid = total_frames // 2

# Pick robust timepoints
target_indices = [
    0,                # Start
    1000,             # Middle (Instability visible)
    total_frames - 1  # End (Full Maze)
]

labels = ["Start", "Middle", "End"]

print("Saving selected frames...")
for i, label in zip(target_indices, labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Species V (Index 1) shows the pattern
    grid = out[1, :, :, i] / Omega
    
    plt.imshow(grid, cmap='viridis', interpolation='bicubic')
    plt.axis('off')
    
    filename = f"GS_{label}.png"
    plt.savefig(save_folder / filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved {filename}")

print("Done! Check the 'Poster_Images_Final' folder.")

# %% STEP GILLESPIE 1D
N = 50
T = 100
dt = 1.0

mid = int(N/2)
x0 = jnp.zeros((2, N))
x0 = x0.at[0, :].set(100) # Fill with U
x0 = x0.at[1, mid-5:mid+5].set(int(0.25 * Omega)) # Add block of V (0.25 concentration)
x0 = x0.at[0, mid-5:mid+5].set(int(0.25 * Omega) // 2)
diff_rates = jnp.array([0.2, 0.1]) 

step_gs_gillespie_1d = gs_model.step_gillespie_1d(diff_rates)
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series_1d(k0, x0, 0, T, dt, step_gs_gillespie_1d, True)

midway = out[:, :, int(T/2)] / Omega # Convert back to concentrations

plt.figure(figsize=(10, 5))
plt.plot(midway[0], label="U (conc)")
plt.plot(midway[1], label="V (conc)")
plt.title(f"Gray-Scott 1D Gillespie (Omega={Omega})")
plt.legend()
plt.show()

# %% STEP GILLESPIE 2D 
gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=1000
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U ->
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs_model = jsmfsb.shorthand_to_spn(gs_sh)
diff_rates = jnp.array([0.1, 0.2])
Omega = 1000
T = 50  
dt = 1.0
M, N = 20, 20
k = jax.random.key(42)
k1, k2 = jax.random.split(k)
x0 = jnp.zeros((2, N, M))
x0 = x0.at[0, :, :].set(Omega)  # U uniform (feed species)
x0 = x0.at[1, N//2-2:N//2+2, M//2-2:M//2+2].set(Omega)  # V central block
x0 = x0.at[0, :, :].add(0.01 * Omega * jax.random.uniform(k1, (N, M)))



# step_gs_cle_2d = gs_model.step_cle_2d(diff_rates, dt = dt)
step_gs_gillespie_2d = gs_model.step_gillespie_2d(diff_rates)

# out_cle_2d = jsmfsb.sim_time_series_2d(k1, x0, 0, T, dt, step_gs_cle_2d, True)
out_gillespie_2d = jsmfsb.sim_time_series_2d(k2, x0, 0, T, dt, step_gs_gillespie_2d, True)

# final_cle_2d_conc = out_cle_2d[:, :, -1] / Omega
final_gillespie_2d_conc = out_gillespie_2d[:, :, -1] / Omega

# fig, axes = plt.subplots(1, 2)

# plt.plot(final_cle_2d_conc[0], label="U (conc)")
# plt.plot(final_cle_2d_conc[1], label="V (conc)")
# plt.title(f"Gray-Scott 2D CLE (Omega={Omega})")

fig, axes = plt.subplots(2, 3)
timepoints = [0, len(out_gillespie_2d[0,0,0,:])//2, -1]
titles = ['t=0', f't={T//2}', f't={T}']

for idx, (t_idx, title) in enumerate(zip(timepoints, titles)):
    axes[0, idx].imshow(out_gillespie_2d[0, :, :, t_idx], vmin=0, vmax=Omega)
    axes[0, idx].set_title(title)
    axes[0, idx].axis('off')
    axes[1, idx].imshow(out_gillespie_2d[1, :, :, t_idx], vmin=0, vmax=Omega)
    axes[1, idx].axis('off')

axes[0,0].set_ylabel('U')
axes[1,0].set_ylabel('V')
plt.suptitle(f'Gray-Scott SSA (Omega={Omega})')
plt.tight_layout()
plt.show()
# %% STEP CLE 2D
import imageio as iio
from pathlib import Path 
import shutil

base_path = Path("/Users/josh/Documents/SMFSB Python")
save_folder = base_path / "GS_2D_CLE_Frames"
if save_folder.is_dir():
    print(f"Deleting existing folder: {save_folder}")
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Frames will be saved in: {save_folder}")

M = 100
N = 100
T = 100
dt = 1.0
n_steps = int(T/dt)
x0 = jnp.zeros((2, N, M))
x0 = x0.at[0, :, :].set(Omega)
x0 = x0.at[1, int(M/2), int(N/2)].set(0.025 * Omega)

k = jax.random.key(42)

step_gs_cle_2d = gs_model.step_cle_2d(diff_rates, dt = dt)

print(f"Running JAX simulation for {n_steps} steps...")
start = time.time()
out = jsmfsb.sim_time_series_2d(k, x0, 0, T, dt, step_gs_cle_2d, False)
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

print("Saving frames...")
start_save = time.time()
for i in range(out.shape[3]):  

    fig, axes = plt.subplots(figsize=(6, 10))
    
    for j in range(2):

        plt.imshow(out[i, :, :, i]) # out = (species, x, y, time)

    filename = f"GS_CLE_Frames_{i:05d}.png"
    
    save_path = save_folder / filename
    
    plt.savefig(save_path)
    plt.close(fig)

end_save = time.time()
print(f"Frames saved in {end_save - start_save:.2f} seconds.")

print("Compiling GIF...")
file_list = sorted(save_folder.glob("*.png"))
frames = [iio.imread(file) for file in file_list]

gif_path = base_path / "GS_CLE_2D.gif"
iio.mimsave(
    gif_path, frames
    )

print(f"GIF saved to {gif_path}")
# %%
base_path = Path("/Users/josh/Documents/Year 3/SMFSB Python")

save_folder = base_path / "GS_2D_CLE_Frames"

if not base_path.exists():
    print(f"ERROR: The folder '{base_path}' does not exist!")
    print("Check your spelling or create the folder manually first.")
else:
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Success! Saving to: {save_folder}")

gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=0 s
 Pop:V=0 s
@parameters
 N=10000
 F=0.029
 k=0.057
@reactions
@r=DegradationU
 U -> 
 (F+k)*U
@r=Production
 -> V
 F*N
@r=DegredationV
 V ->
 F*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

M = 1000
N = 1000
T = 500
D = 2
diff_base_rate = 0.1

x0 = jnp.zeros((2, M, N)) # init U to 0
x0 = x0.at[1,:,:].set(10000) # init V to 1000
x0 = x0.at[:, int(M / 2), int(N / 2)].set(gs.m)
step_gs_2d = gs.step_cle_2d(jnp.array([diff_base_rate, D*diff_base_rate]), 0.2)
k0 = jax.random.key(int(time.time() * 1e9))
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, T, 50, step_gs_2d, True)
print(ts.shape)
u_stack = []
v_stack = []
fig_size_inches = 8
output_dpi = 100 

for i in range(ts.shape[3]):
    if i % 10 == 0: # Optional: Print progress every 10 frames
        print(f"Processing frame {i} of {ts.shape[3]}")
    
    filename_u = save_folder / f"gs-U-{i:05d}.png"

    fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=output_dpi)
    
    ax.imshow(ts[0,:,:,i], interpolation='nearest')
    ax.set_axis_off()
    plt.savefig(filename_u, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
    plt.close(fig) 
    
    filename_v = save_folder / f"gss-V-{i:05d}.png"
    
    fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=output_dpi)
    ax.imshow(ts[1,:,:,i], interpolation='nearest')
    ax.set_axis_off()
    plt.savefig(filename_v, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
    plt.close(fig)

    u_stack.append(iio.imread(filename_u))
    v_stack.append(iio.imread(filename_v))

print("\nCreating animated gifs...")
iio.mimsave(base_path / "GS-U.gif", u_stack)
iio.mimsave(base_path / "GS-V.gif", v_stack)
print("Done!")

# %%
import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio as io
import shutil
from pathlib import Path
import time

GRID_W = 200  
GRID_H = 200
TOTAL_TIME = 8000.0
DT_SIM = 0.2     
DT_SAVE = 200.0      

POPULATION = 1000
# Varying 'a' (Feed) and 'b' (Kill) changes the pattern type.
# Zebra/Maze: a=0.037, b=0.06
# Spots:      a=0.030, b=0.062
# Bigger spots: a=0.0140, b=0.0510
FEED_A = 0.030
KILL_B = 0.062
# ==========================================
base_path = Path.cwd()
save_folder = base_path / f"Spots_Omega{POPULATION}.gif"
if save_folder.exists():
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Saving to: {save_folder}")

gs_sh = f"""
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U={POPULATION} s
 Pop:V={POPULATION} s
@parameters
 N={POPULATION}
 a={FEED_A}
 b={KILL_B}
@reactions
@r=DegradationU
 U -> 
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

D = 2
diff_base_rate = 0.1

x0 = jnp.zeros((2, GRID_W, GRID_H))
x0 = x0.at[1,:,:].set(POPULATION)  # Fill V
x0 = x0.at[:, int(GRID_W / 2), int(GRID_H / 2)].set(gs.m) # Seed in center

print("Compiling JAX functions...")
step_gs_2d = gs.step_cle_2d(jnp.array([diff_base_rate, D*diff_base_rate]), DT_SIM)
k0 = jax.random.key(42)

print(f"Running Simulation (T={TOTAL_TIME})...")
start_time = time.time()
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, TOTAL_TIME, DT_SAVE, step_gs_2d, True)
end_time = time.time()
print(f"Done! Took {end_time - start_time:.2f} seconds.")
print(f"Output shape: {ts.shape}")

u_stack = []
for i in range(ts.shape[3]):
    if i % 5 == 0: print(f"Saving frame {i}/{ts.shape[3]}")
    
    img_data = ts[1,:,:,i] / POPULATION 
    
    filename = save_folder / f"gs_{i:04d}.png"
    
    plt.imsave(filename, img_data, cmap='viridis', vmin=0, vmax=0.6)
    u_stack.append(io.imread(filename))

gif_path = base_path / f"Spots_Omega{POPULATION}.gif"
io.mimsave(gif_path, u_stack)
print(f"GIF saved: {gif_path}")
# %% GS High Omega CLE
sns.set_theme(style='ticks')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

M = 100
N = 100

gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=1000
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U ->
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

x0 = jnp.zeros((2, M, N))
x0 = x0.at[1, :, :].set(1000)
r = 5
x0 = x0.at[0, M//2-r:M//2+r, N//2-r:N//2+r].set(1000)

print(f"U mean: {float(jnp.mean(x0[0])):.1f}")
print(f"V mean: {float(jnp.mean(x0[1])):.1f}")

step_gs_cle_2d = gs.step_cle_2d(jnp.array([0.1, 0.2]))
k0 = jax.random.key(42)

# --- JIT warmup ---
print("Warming up JIT...")
_ = step_gs_cle_2d(k0, x0, 0, 100.0)
_.block_until_ready()
print("Done.")

checkpoints = [0, 1000, 2000]
frames = [x0]
x_current = x0
t_current = 0

print("Running simulation...")
start = time.time()
for t_next in checkpoints[1:]:
    dt_chunk = float(t_next - t_current)
    print(f"  Stepping t={t_current} -> t={t_next}...", flush=True)
    x_current = step_gs_cle_2d(k0, x_current, t_current, dt_chunk)
    x_current.block_until_ready()
    frames.append(x_current)
    t_current = t_next
    print(f"    V mean: {float(jnp.mean(x_current[1])):.1f}", flush=True)
print(f"Done in {time.time() - start:.2f}s")

import pickle
with open('gs_frames_high_cle.pkl', 'wb') as f:
    pickle.dump([jnp.array(frame) for frame in frames], f)

# %% GS High Omega CLE Plot
import pickle
with open('gs_frames_high_cle.pkl', 'rb') as f:
    frames = pickle.load(f)

plot_labels = [f'$t={t}$' for t in checkpoints]

# --- Plot V species ---
fig, axes = plt.subplots(1, len(checkpoints), figsize=(16, 4))
plt.subplots_adjust(right=0.88)

vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

for col, (frame, tlabel) in enumerate(zip(frames, plot_labels)):
    ax = axes[col]
    im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                   vmin=0, vmax=vmax, origin='lower',
                   interpolation='bicubic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tlabel, fontsize=12)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$V$ count', fontsize=10)

# plt.suptitle('Gray-Scott Pattern Formation (CLE)',
#              fontsize=12, fontweight='bold')
plt.savefig('Report Images/GS_patterns_high_cle.png', dpi=300, bbox_inches='tight')
plt.show()
# %% GS Low Omega CLE
gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=10
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U ->
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

x0 = jnp.zeros((2, M, N))
x0 = x0.at[1, :, :].set(10) 
r = 5
x0 = x0.at[0, M//2-r:M//2+r, N//2-r:N//2+r].set(10)

print(f"U mean: {float(jnp.mean(x0[0])):.1f}")
print(f"V mean: {float(jnp.mean(x0[1])):.1f}")

step_gs_cle_2d = gs.step_cle_2d(jnp.array([0.1, 0.2]))
k0 = jax.random.key(42)

# --- JIT warmup ---
print("Warming up JIT...")
_ = step_gs_cle_2d(k0, x0, 0, 100.0)
_.block_until_ready()
print("Done.")

checkpoints = [0, 1000, 2000]
frames = [x0]
x_current = x0
t_current = 0

print("Running simulation...")
start = time.time()
for t_next in checkpoints[1:]:
    dt_chunk = float(t_next - t_current)
    print(f"  Stepping t={t_current} -> t={t_next}...", flush=True)
    x_current = step_gs_cle_2d(k0, x_current, t_current, dt_chunk)
    x_current.block_until_ready()
    frames.append(x_current)
    t_current = t_next
    print(f"    V mean: {float(jnp.mean(x_current[1])):.1f}", flush=True)
print(f"Done in {time.time() - start:.2f}s")

import pickle
with open('gs_frames_low_cle.pkl', 'wb') as f:
    pickle.dump([jnp.array(frame) for frame in frames], f)

# %% GS Low Omega CLE Plot
import pickle
with open('gs_frames_low_cle.pkl', 'rb') as f:
    frames = pickle.load(f)

plot_labels = [f'$t={t}$' for t in checkpoints]

# --- Plot V species ---
fig, axes = plt.subplots(1, len(checkpoints), figsize=(16, 4))
plt.subplots_adjust(right=0.88)

vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

for col, (frame, tlabel) in enumerate(zip(frames, plot_labels)):
    ax = axes[col]
    im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                   vmin=0, vmax=vmax, origin='lower',
                   interpolation='bicubic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tlabel, fontsize=12)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$V$ count', fontsize=10)

# plt.suptitle('Gray-Scott Pattern Formation (CLE)',
#              fontsize=12, fontweight='bold')
plt.savefig('Report Images/GS_patterns_low_cle.png', dpi=300, bbox_inches='tight')
plt.show()
# %% GS Low v High Omega CLE 
import pickle

with open('gs_frames_high_cle.pkl', 'rb') as f:
    frames_high = pickle.load(f)

with open('gs_frames_low_cle.pkl', 'rb') as f:
    frames_low = pickle.load(f)

plot_labels_low = ['$t=0$', '$t=1000$', '$t=2000$']
plot_labels_high = ['$t=0$', '$t=1000$', '$t=2000$']
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
plt.subplots_adjust(right=0.88)

for row, (frames, labels, omega) in enumerate(zip(
        [frames_low, frames_high],
        [plot_labels_low, plot_labels_high],
        [10, 1000])):

    row_vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

    for col, (frame, tlabel) in enumerate(zip(frames, labels)):
        ax = axes[row, col]
        im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                       vmin=0, vmax=row_vmax, origin='lower',
                       interpolation='bicubic')
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(tlabel, fontsize=12)
        if col == 0:
            ax.set_ylabel(f'$\Omega={omega}$', fontsize=12, fontweight='bold')

    cbar_ax = fig.add_axes([0.90, 0.55 - row * 0.5, 0.015, 0.38])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('$V$ count', fontsize=10)

# plt.suptitle('Gray-Scott Turing Patterns: Effect of System Size $\Omega$',
#              fontsize=12, fontweight='bold')
plt.savefig('Report Images/GS_omega_comparison_cle.png', dpi=300, bbox_inches='tight')
plt.show()
# %% GS High Omega Euler
sns.set_theme(style='ticks')
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

M = 100
N = 100

gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=1000
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U ->
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)
Omega = 1000
x0 = jnp.zeros((2, M, N))
x0 = x0.at[1, :, :].set(Omega) 
r = 5
x0 = x0.at[0, M//2-2*r:M//2+r, N//2-r:N//2+r].set(Omega)
key = jax.random.key(43)
x0 = x0.at[0, :, :].add(0.01 * Omega * jax.random.uniform(key, (M, N)))

print(f"U mean: {float(jnp.mean(x0[0])):.1f}")
print(f"V mean: {float(jnp.mean(x0[1])):.1f}")

step_gs_euler_2d = gs.step_euler_2d(jnp.array([0.1, 0.2]))
k0 = jax.random.key(425)

# --- JIT warmup ---
print("Warming up JIT...")
_ = step_gs_euler_2d(k0, x0, 0, 100.0)
_.block_until_ready()
print("Done.")

checkpoints = [0, 1000, 2000]
frames = [x0]
x_current = x0
t_current = 0

print("Running simulation...")
start = time.time()
for t_next in checkpoints[1:]:
    dt_chunk = float(t_next - t_current)
    print(f"  Stepping t={t_current} -> t={t_next}...", flush=True)
    x_current = step_gs_euler_2d(k0, x_current, t_current, dt_chunk)
    x_current.block_until_ready()
    frames.append(x_current)
    t_current = t_next
    print(f"    V mean: {float(jnp.mean(x_current[1])):.1f}", flush=True)
print(f"Done in {time.time() - start:.2f}s")

import pickle
with open('gs_frames_high_euler.pkl', 'wb') as f:
    pickle.dump([jnp.array(frame) for frame in frames], f)

# %% GS High Omega Euler Plot
import pickle
with open('gs_frames_high_euler.pkl', 'rb') as f:
    frames = pickle.load(f)

plot_labels = [f'$t={t}$' for t in checkpoints]

# --- Plot V species ---
fig, axes = plt.subplots(1, len(checkpoints), figsize=(16, 4))
plt.subplots_adjust(right=0.88)

vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

for col, (frame, tlabel) in enumerate(zip(frames, plot_labels)):
    ax = axes[col]
    im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                   vmin=0, vmax=vmax, origin='lower',
                   interpolation='bicubic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tlabel, fontsize=12)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$V$ count', fontsize=10)

# plt.suptitle('Gray-Scott Pattern Formation (EUler)',
#              fontsize=12, fontweight='bold')
plt.savefig('Report Images/GS_patterns_high_euler.png', dpi=300, bbox_inches='tight')
plt.show()
# %% GS Low Omega Euler
gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=10
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U ->
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

x0 = jnp.zeros((2, M, N))
x0 = x0.at[1, :, :].set(10) 
r = 5
x0 = x0.at[0, M//2-r:M//2+r, N//2-r:N//2+r].set(10)

print(f"U mean: {float(jnp.mean(x0[0])):.1f}")
print(f"V mean: {float(jnp.mean(x0[1])):.1f}")

step_gs_euler_2d = gs.step_euler_2d(jnp.array([0.1, 0.2]))
k0 = jax.random.key(435)

# --- JIT warmup ---
print("Warming up JIT...")
_ = step_gs_euler_2d(k0, x0, 0, 100.0)
_.block_until_ready()
print("Done.")

checkpoints = [0, 1000, 2000]
frames = [x0]
x_current = x0
t_current = 0

print("Running simulation...")
start = time.time()
for t_next in checkpoints[1:]:
    dt_chunk = float(t_next - t_current)
    print(f"  Stepping t={t_current} -> t={t_next}...", flush=True)
    x_current = step_gs_euler_2d(k0, x_current, t_current, dt_chunk)
    x_current.block_until_ready()
    frames.append(x_current)
    t_current = t_next
    print(f"    V mean: {float(jnp.mean(x_current[1])):.1f}", flush=True)
print(f"Done in {time.time() - start:.2f}s")

import pickle
with open('gs_frames_low_euler.pkl', 'wb') as f:
    pickle.dump([jnp.array(frame) for frame in frames], f)

# %% GS Low Omega Euler Plot
import pickle
with open('gs_frames_low_euler.pkl', 'rb') as f:
    frames = pickle.load(f)

plot_labels = [f'$t={t}$' for t in checkpoints]

# --- Plot V species ---
fig, axes = plt.subplots(1, len(checkpoints), figsize=(16, 4))
plt.subplots_adjust(right=0.88)

vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

for col, (frame, tlabel) in enumerate(zip(frames, plot_labels)):
    ax = axes[col]
    im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                   vmin=0, vmax=vmax, origin='lower',
                   interpolation='bicubic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tlabel, fontsize=12)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$V$ count', fontsize=10)

# plt.suptitle('Gray-Scott Pattern Formation (Euler)',
#              fontsize=12, fontweight='bold')
plt.savefig('Report Images/GS_patterns_low_euler.png', dpi=300, bbox_inches='tight')
plt.show()
# %% GS Low v High Omega Euler 
import pickle

with open('gs_frames_low_euler.pkl', 'rb') as f:
    frames = pickle.load(f)

plot_labels = ['$t=0$', '$t=1000$', '$t=2000$']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(right=0.88)

row_vmax = max(float(jnp.max(jnp.clip(f[1], 0, None))) for f in frames)

for col, (frame, tlabel) in enumerate(zip(frames, plot_labels)):
    ax = axes[col]
    im = ax.imshow(jnp.clip(frame[1], 0, None), cmap='viridis',
                   vmin=0, vmax=row_vmax, origin='lower',
                   interpolation='bicubic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tlabel, fontsize=12)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$V$ count', fontsize=10)

plt.savefig('Report Images/GS_euler.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
