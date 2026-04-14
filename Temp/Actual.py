# %% Import Functions
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random
import time 
import smfsb
import smfsb.models
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
# %% LV

lv = smfsb.models.lv()
print(lv)
step_lv = lv.step_gillespie()
out = smfsb.sim_time_series(lv.m, 0, 25, 0.1, step_lv)


fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lv.n)
# %% LV Gillespie 

lv = smfsb.models.lv()
print(lv)
times, states = lv.gillespie(10000)

plt.figure()
for i in range(2):
    if i == 1:
        style = '--'
    else:
        style = '-'
    plt.step(times, states[1:, i], where="post", label=f'Y{i+1}', linestyle=style) 
plt.xlabel("Time")
plt.ylabel("Y")
plt.legend()

plt.figure() 
plt.plot(states[:, 0], states[:, 1]) 
plt.xlabel("Y1")
plt.ylabel("Y2")
plt.legend()
plt.show()

# %% LV Gillespied

lv = smfsb.models.lv()
print(lv)
states = lv.gillespied(30, 0.1)

fig, axis = plt.subplots()
for i in range(2):
    axis.step(np.arange(0, 30, 0.1), states[:, i], where="post")

axis.legend(lv.n)

# %% LV Discretise
dt = 0.01

lv = smfsb.models.lv()
print(lv)
times, states = lv.gillespie(2000)
out = smfsb.discretise(times, states, dt)

fig, axis = plt.subplots()
for i in range(2):
    axis.step(np.arange(0, times[-1], dt), out[:, i], where="post")

axis.legend(lv.n)
# %% Immigration-Death Process

times, states = smfsb.imdeath(150)

fig, axis = plt.subplots()
axis.step(times, states[1:], where="post")

id = smfsb.models.id()
print(id)
step = id.step_gillespie()
out = smfsb.sim_time_series(id.m, 0, 100, 0.1, step)


fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:, 0])

axis.legend(id.n)

# %% Dimerisation: Fig 7.2

dimer_determ_sh = """
@model:3.1.1=DimerKineticsDet "Dimerisation Kinetics (deterministic)"
 s=mole, t=second, v=litre, e=mole
@compartments
 Cell=1e-15
@species
 Cell:[P]=5e-7 s
 Cell:[P2]=0 s
@reactions
@r=Dimerisation
 2P->P2
 Cell*k1*P*P : k1=5e5
@r=Dissociation
 P2->2P
 Cell*k2*P2 : k2=0.2
"""

dimer_determ = smfsb.shorthand_to_spn(dimer_determ_sh)

fig, axis = plt.subplots(figsize=(10, 8))

# V = 1e-15
# NA = 6.022e23
# conversion_factor = V * NA

step_e = dimer_determ.step_euler(0.001)
out_e = smfsb.sim_time_series(dimer_determ.m, 0, 10, 0.1, step_e)
time_vector = np.linspace(0, 10, out_e.shape[0])


for i in range(2):
    style = '--' if i == 1 else '-'
    axis.plot(time_vector, out_e[:, i], linestyle=style)

axis.set_xlabel("Time (s)")
axis.set_ylabel("Concentration (M)")
axis.legend(dimer_determ.n)

# dimer_stoch_sh = """
# @model:3.1.1=DimerKineticsStoch "Dimerisation Kinetics (stochastic)"
#  s=item, t=second, v=litre, e=item
# @compartments
#  Cell=1e-15
# @species
#  Cell:P=301 s
#  Cell:P2=0 s
# @reactions
# @r=Dimerisation
#  2P->P2
#  c1*P*(P-1)/2 : c1=1.66e-3
# @r=Dissociation
#  P2->2P
#  c2*P2 : c2=0.2
# """

# dimer_stoch = smfsb.shorthand_to_spn(dimer_stoch_sh)

# step_g = dimer_stoch.step_gillespie()
# out_g = smfsb.sim_time_series(dimer_stoch.m, 0, 10, 0.1, step_g)

# for i in range(2):
#     style = '--' if i == 1 else '-'
#     axes[1].plot(time_vector, out_g[:, i], linestyle=style)

# axes[1].set_xlabel("Time")
# axes[1].set_ylabel("# of molecules")
# axes[1].legend(dimer_stoch.n)
# axes[1].set_title("Stochastic (Gillespie)")

plt.tight_layout()
plt.show()

# %% Figure 7.5 Left

dimer_stoch_sh = """
@model:3.1.1=DimerKineticsStoch "Dimerisation Kinetics (stochastic)"
 s=item, t=second, v=litre, e=item
@compartments
 Cell=1e-15
@species
 Cell:P=301 s
 Cell:P2=0 s
@reactions
@r=Dimerisation
 2P->P2
 c1*P*(P-1)/2 : c1=1.66e-3
@r=Dissociation
 P2->2P
 c2*P2 : c2=0.2
"""

fig, axis = plt.subplots()

dimer_stoch = smfsb.shorthand_to_spn(dimer_stoch_sh)

step_g = dimer_stoch.step_gillespie()
out_g = smfsb.sim_time_series(dimer_stoch.m, 0, 20, 0.1, step_g)

V = 1e-15
NA = 6.022e23
conversion_factor = V * NA
time_vector = np.linspace(0, 10, out_g.shape[0])

out_g_conc = out_g / conversion_factor

for i in range(2):
    style = '--' if i == 1 else '-'
    axis.plot(time_vector, out_g_conc[:, i], linestyle=style)

axis.set_xlabel("Time")
axis.set_ylabel("Concentration (M)")
axis.legend(dimer_stoch.n)
axis.set_title("Stochastic (Gillespie)")

plt.tight_layout()
plt.show()
# %% Figure 7.5 Right

n = 100 # Number of simulations

dimer = smfsb.models.dimer()
print(dimer)

fig, axis = plt.subplots()

for _ in range(n):
    step = dimer.step_gillespie()
    out = smfsb.sim_time_series(dimer.m, 0, 10, 0.1, step)
    axis.plot(range(out.shape[0]), out[:, 0])

plt.ylim(bottom=0)
plt.xlabel("Time")
plt.ylabel("# of molecules")
axis.legend(dimer.n)
plt.show()

# %% Figure 7.6

num_sh = """
@model:3.1.1=DimerKineticsStoch "Dimerisation Kinetics"
 s=item, t=second, v=litre, e=item
@compartments
 Cell=1e-15
@species
 Cell:P=301 s
 Cell:P2=0 s
@reactions
@r=Dimerisation
 2P->P2
 c1*P*(P-1)/2 : c1=1.66e-3
@r=Dissociation
 P2->2P
 c2*P2 : c2=0.2
"""
dimer = smfsb.shorthand_to_spn(num_sh)

n = 10000

def run_one_simulation():
    step = dimer.step_gillespie()
    out = smfsb.sim_time_series(dimer.m, 0, 10, 0.1, step)
    return out[:, 0]

start = time.time()
# 21 seconds (1000 sims)
# simulation_data = []

# for _ in range(n):
    
#     step = dimer.step_gillespie()
#     out = smfsb.sim_time_series(dimer.m, 0, 10, 0.1, step)

#     simulation_data.append(out[:,0])

simulation_data = Parallel(n_jobs=-1)(delayed(run_one_simulation)() for _ in range(n))
# 16 seconds (1000 sims)
# 67 seconds (10000 sims)
# 7 seconds now! (10000 sims)
end = time.time()


all_states = np.array(simulation_data)
mean_states = np.mean(all_states,axis=0)
var_states = np.var(all_states,axis=0)
sd_states = np.sqrt(var_states)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
print(f"Time taken to run {n} simulations = {end - start} seconds!")
axes[0].plot(np.linspace(0, 10, out.shape[0]), mean_states, linestyle='-', label='Sample mean')

for i in range(2):
    axes[0].plot(np.linspace(0, 10, out.shape[0]), mean_states+(-1)**i*3*sd_states, linestyle = '--', label='Sample mean plus/minus 3 SDs' if i == 0 else None)

axes[0].set_ylim(bottom=0)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("# Molecules")
axes[1].set_title("Mean Trajectory of P")
axes[0].legend()

final_time_data = all_states[:, -1]

min_val = int(np.min(final_time_data))
max_val = int(np.max(final_time_data))
bin_edges = np.arange(min_val - 0.5, max_val + 1.5)

axes[1].hist(final_time_data, bins=bin_edges,  density=True, edgecolor='black')
axes[1].set_xlabel("P(10)")
axes[1].set_ylabel("Density")
axes[1].set_title("PMF Estimate for P(10)")
plt.tight_layout()
plt.show()

# %% Michaelis-Menten Kinetics (Deterministic) Figure 7.8
mm_cont_sh = """
@model:3.1.1=MMKineticsDet "Michaelis-Menten Kinetics (deterministic)"
 s=mole, t=second, v=litre, e=mole
@compartments
 Cell=1e-15
@species
 Cell:S=301 s
 Cell:E=120 s
 Cell:SE=0 s
 Cell:P=0 s
@reactions
@r=Binding
 S+E->SE
 c1*S*E : c1=1.66e-3
@r=Dissociation
 SE->S+E
 c2*SE : c2=1e-4
@r=Conversion
 SE->P+E
 c3*SE : c3=0.1
"""
mm = smfsb.shorthand_to_spn(mm_cont_sh)

fig, axis = plt.subplots(figsize=(10,6))

step_e = mm.step_euler()
out_e = smfsb.sim_time_series(mm.m, 0, 50, 0.1, step_e)

time_vector = np.linspace(0, 50, out_e.shape[0])
V = 1e-15
NA = 6.022e23
conversion_factor = V * NA
out_e_conc = out_e / conversion_factor

for i in range(4):
    if i == 0:
        style = '-'
    elif i == 1:
        style = '--' 
    elif i == 2:
        style = ':'
    else:
        style = '-.'
    axis.plot(time_vector, out_e_conc[:, i], linestyle=style)

axis.set_xlabel("Time")
axis.set_ylabel("Concentration (M)")
axis.legend(mm.n)
axis.set_title("Deterministic (Euler)") 
plt.show()
# %% Michaelis-Menten Kinetics (Determinstic) Figure 7.9

mm_stoch_sh = """
@model:3.1.1=MMKineticsStoch "Michaelis-Menten Kinetics (determinstic)" 
 s=mole, t=second, v=litre, e=mole
@compartments
 Cell=1e-15
@species
 Cell:S=301 s
 Cell:E=120 s
 Cell:SE=0 s
 Cell:P=0 s
@reactions
@r=Binding
 S+E->SE
 c1*S*E : c1=1.66e-3
@r=Dissociation
 SE->S+E
 c2*SE : c2=1e-4
@r=Conversion
 SE->P+E
 c3*SE : c3=0.1
"""

mm = smfsb.shorthand_to_spn(mm_stoch_sh)

fig, axis = plt.subplots(figsize=(10,6))

step_g = mm.step_gillespie()
out_g= smfsb.sim_time_series(mm.m, 0, 100, 0.1, step_g)

for i in range(4):
    if i == 0:
        style = '-'
    elif i == 1:
        style = '--' 
    elif i == 2:
        style = ':'
    else:
        style = '-.'
    axis.plot(np.linspace(0,100,out_g.shape[0]), out_g[:, i], linestyle=style)

axis.set_ylabel("# Molecules")
axis.legend(mm.n)
axis.set_title("Stochastic (Gillespie)") 
plt.show()
# %% Auto-Regulatory Network

ar_sh = """
@model:3.1.1=AutoRegulatoryNetwork "Auto-regulatory network (altered version)"
 s=item, t=second, v=litre, e=item
@compartments
 Cell
@species
 Cell:Gene=10 s
 Cell:P2Gene=0 s "P2.Gene"
 Cell:Rna=0 s
 Cell:P=0 s
 Cell:P2=0 s
@reactions
@r=RepressionBinding "Repression binding"
 Gene+P2 -> P2Gene
 k1*Gene*P2 : k1=1
@r=ReverseRepressionBinding "Reverse repression binding"
 P2Gene -> Gene+P2
 k1r*P2Gene : k1r=10
@r=Transcription
 Gene -> Gene+Rna
 k2*Gene : k2=0.02
@r=Translation
 Rna -> Rna+P
 k3*Rna : k3=10
@r=Dimerisation
 2P -> P2
 k4*0.5*P*(P-1) : k4=1
@r=Dissociation
 P2 -> 2P
 k4r*P2 : k4r=1
@r=RnaDegradation "RNA Degradation"
 Rna ->
 k5*Rna : k5=0.1
@r=ProteinDegradation "Protein degradation"
 P ->
 k6*P : k6=0.01
"""

t_start = 0
t_max = 500
dt = 0.1
ar = smfsb.shorthand_to_spn(ar_sh)

fig1, axes1 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,8))
species_indices = [2, 3, 4] # Indices for Rna, P, P2 in the state vector
species_labels = ["RNA", "P", "P2"]

start = time.time()

step_g = ar.step_gillespie()
out_g = smfsb.sim_time_series(ar.m, t_start, t_max, dt, step_g)
out_times = np.linspace(t_start, t_max, out_g.shape[0])

for i in range(3):
    axes1[i].plot(out_times, out_g[:, species_indices[i]]) 
    axes1[i].set_ylabel(species_labels[i])

end = time.time()

axes1[2].set_xlabel("Time")
print(f"Time taken to run = {end - start} seconds!")

# First 25 seconds

t_max_zoom = 25

zoom_indices = np.where(out_times <= t_max_zoom)[0]

zoom_times = out_times[zoom_indices]
zoom_states = out_g[zoom_indices, :]

fig2, axes2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))

for i in range(3):
    axes2[i].plot(zoom_times, zoom_states[:, species_indices[i]])
    axes2[i].set_ylabel(species_labels[i])

axes2[2].set_xlabel("Time")

plt.show()
# %%
