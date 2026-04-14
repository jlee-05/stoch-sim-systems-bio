# %% Functions ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random
import time 
import smfsb
import smfsb.models

def lv_odes(t, y, k1, k2, k3):

    Y1, Y2 = y[0], y[1]
    dY1_dt = k1*Y1 - k2 * Y1 * Y2
    dY2_dt = k2*Y1*Y2 - k3*Y2

    return np.array([dY1_dt, dY2_dt])

# Gillespie Algorithm ------------------------------------------------------------------------------------

def gillespie(initial_state, t_max, hazard_func, stoich_matrix, **rate_constants):

    t = 0
    x = np.array(initial_state)

    t_history = [t]
    x_history = [x.copy()]

    while t < t_max:

        hazards = hazard_func(x, **rate_constants)
        total_haz = np.sum(hazards)

        if total_haz == 0:
            break

        dt = np.random.exponential(1/total_haz)
        t += dt

        num_reactions = len(hazards)
        reaction_probs = hazards / total_haz
        j = np.random.choice(num_reactions, p=reaction_probs)

        x += stoich_matrix[:, j]
        t_history.append(t)
        x_history.append(x.copy())

    return {'t': np.array(t_history), 'x': np.array(x_history)}

# Discretise Function

def discretise(gill_out, dt = 1, t_start = 0, t_end=10):
    
    times_events = gill_out['t']
    states_events = gill_out['x']
    n_regular_steps = int((t_end - t_start) // dt) + 1

    t_regular = t_start + np.arange(n_regular_steps) * dt
    num_species =  states_events.shape[1] 
    x = np.zeros((n_regular_steps, num_species))
    j = 0

    for i in range(n_regular_steps):
        target_time = t_regular[i]
        while (j + 1 < len(times_events) and 
               times_events[j + 1] <= target_time):
            j += 1
        x[i, :] = states_events[j]

    return {'t': t_regular, 'x': x}

            

# %% Plot LV ------------------------------------------------------------------------------------
initial_state = [4, 10]
rate_constants = {'k1': 1.0, 'k2': 0.1, 'k3': 0.1}
t_span = [0,100]
t_eval = np.linspace(t_span[0], t_span[1], 500)

soln = integrate.solve_ivp(lv_odes, t_span, y0=initial_state, t_eval=t_eval, args=(rate_constants['k1'], rate_constants['k2'], rate_constants['k3']))
plt.figure(figsize=(10, 6))
plt.plot(soln.t, soln.y[0], label='[Y1]')
plt.plot(soln.t, soln.y[1], label='[Y2]', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Lotka-Volterra Dynamics (Deterministic)")
plt.legend()

plt.figure(figsize=(10, 6)) # Equilibrium
eq_prey = rate_constants['k3'] / rate_constants['k2']
eq_predator = rate_constants['k1'] / rate_constants['k2']
plt.plot(eq_prey, eq_predator, 'ro')

soln = integrate.solve_ivp(lv_odes, t_span, y0=initial_state, t_eval=t_eval, args=(rate_constants['k1'], rate_constants['k2'], rate_constants['k3']))
plt.plot(soln.y[0], soln.y[1], linewidth=2, color = 'r')
for i in range(1,5):
    for j in range(5,11):
        soln = integrate.solve_ivp(lv_odes, t_span, y0=[i,j], t_eval=t_eval, args=(rate_constants['k1'], rate_constants['k2'], rate_constants['k3']))
        plt.plot(soln.y[0], soln.y[1], linestyle='--', linewidth = 1, color = 'black')
plt.xlabel("[Y1]")
plt.ylabel("[Y2]")
plt.legend()
plt.grid(True)
plt.show()
# %% Basic Transcription Algorithm ------------------------------------------------------------------------------------

X = [0] # Initial mRNA count
t = [0] # Initial time

tend = 100 

# Initial Rate Constants
k = 2
gamma = 0.1

while t[-1] < tend:

    currentX = X[-1]
    rates = [k, gamma * currentX]

    rateSum = sum(rates)

    tau = np.random.exponential(1/rateSum)
    t.append(t[-1] + tau)

    rand = random.uniform(0,1)

    # Production event
    if rand * rateSum > 0 and rand * rateSum <= rates[0]:
        X.append(X[-1] + 1)

    # Decay event, could do 'else' but convenient when we want to add more events
    elif rand * rateSum > rates[0] and rand * rateSum <= rates[0] + rates[1]:
        X.append(X[-1] - 1)

plt.plot(t,X)
plt.xlabel("time")
plt.ylabel("mRNA quantity")
plt.show()

# %% Simulations ------------------------------------------------------------------------------------

def prod_degr(x, c_prod, c_degr):
    mrna_count = x[0]
    return np.array([c_prod, c_degr * mrna_count])

intitial_mrna = 0
prod_degr_matrix = np.array([[1,-1]]) # Stoichiometry Matrix
rates = {'c_prod' : 2.0, 'c_degr' : 0.1}
t_max = 100

output = gillespie([intitial_mrna], t_max, prod_degr, prod_degr_matrix, **rates)

# Plot mRNA Count
plt.step(output['t'], output['x'], where='post', label='mRNA count')
plt.xlabel("Time")
plt.ylabel("mRNA Count")
plt.grid(True)
plt.legend()
plt.show()

# %% Stochastic Lotka Volterra ------------------------------------------------------------------------------------
def stoch_lv(x, c1, c2, c3):

    x1, x2 = x
    h1 = c1 * x1
    h2 = c2 * x1 * x2
    h3 = c3 * x2

    return np.array([max(0, h1), max(0, h2), max(0, h3)])

stoch_rates = {'c1': 1.0, 'c2': 0.005, 'c3': 0.6} # Stochastic Rate Constants
x = [50, 100] # Initial Prey, Predator Counts (Marking)
Pre = np.array([[1,0],[1,1],[0,1]])
Post = np.array([[2,0],[0,2],[0,0]])

output = gillespie(x, 25, stoch_lv, (Post - Pre).T, **stoch_rates)

plt.figure(figsize=(10,6))
plt.step(output['t'], output['x'][:, 0], label='[Y1]', where='post')
plt.step(output['t'], output['x'][:, 1], label='[Y2]', linestyle='--', where='post')
plt.xlabel("Time")
plt.ylabel("Y")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(output['x'][:, 0], output['x'][:, 1])
plt.xlabel('[Y1]')
plt.ylabel('[Y2]')
plt.show()
# %% Dimerisation Kinetics

def dimeri_kin_stoch(P, c1, c2): # Stochastic

    P, P2 = P[0], P[1]
    h1 = c1*P*(P-1)/2
    h2 = c2*P2

    return np.array([h1, h2])

def dimeri_kin_cont(t, P, k1, k2): # Continuous
    
    P, P2 = P[0], P[1]
    dP_dt = 2.0 * k2 * P2 - 2.0 * k1 * P**2
    dP2_dt = k1 * P**2 - k2 * P2

    return np.array([dP_dt, dP2_dt])

# %% Figure 6.3
initial_concentrations = [1, 0.0]  # [P], [P2] in Moles/litre
det_rates = {'k1': 1, 'k2': 0.5}
t_span = [0, 5] 
t_eval = np.linspace(t_span[0], t_span[1], 500) 
soln = integrate.solve_ivp(dimeri_kin_cont, t_span, y0=initial_concentrations, t_eval=t_eval, args=(det_rates['k1'], det_rates['k2']))

plt.figure(figsize=(10, 6))
plt.plot(soln.t, soln.y[0], label='[P]')
plt.plot(soln.t, soln.y[1], label='[P2]', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Concentration (M)")
plt.title("Dimerisation Kinetics (Continuous), P(0)=1, P2(0)=0, k1=1, k2=0.5")
plt.legend()

# %% Figure 7.2 (Left)
initial_concentrations = [5e-7, 0.0]  # [P], [P2] in Moles/litre
det_rates = {'k1': 5e5, 'k2': 0.2}
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 500)

soln = integrate.solve_ivp(
    dimeri_kin_cont,
    t_span,
    y0=initial_concentrations,
    t_eval=t_eval,
    args=(det_rates['k1'], det_rates['k2']),
    method='DOP853', # RK45/23' gives oscillations
)

plt.figure(figsize=(10,6))
plt.plot(soln.t, soln.y[0], label='[P]')
plt.plot(soln.t, soln.y[1], label='[P2]', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Concentration (M)")
plt.title("Dimerisation Kinetics (Continuous), P(0)=5e-7, P2(0)=0, k1=5e5, k2=0.2")
plt.legend()
plt.show()

# %% Figure 7.2 (Right)
Pre = np.array([[2,0],[0,1]])
Post = np.array([[0,2],[1,0]])
P = [301, 0] 
rate_constants = {'c1' : 1.66e-3, 'c2' : 0.2}

output = gillespie(P, 10, dimeri_kin_stoch, (Post-Pre).T, **rate_constants)

plt.figure(figsize=(10,6))
plt.plot(output['t'], output['x'][:, 0], label='[P]')
plt.plot(output['t'], output['x'][:, 1], label='[P2]', linestyle='--')
plt.xlabel("Time")
plt.ylabel("# of molecules")
plt.title("Dimerisation Kinetics (Stochastic), P(0)=301, P2(0)=0, k1=1.66e-3, k2=0.2")
plt.legend()
plt.show()

# %% Figure 7.5 (Right)
plt.figure(figsize=(10,6))
for _ in range(21):
    output = gillespie(P, 10, dimeri_kin_stoch, (Post-Pre).T, **rate_constants)
    plt.plot(output['t'], output['x'][:, 0], label='[P]')
plt.xlabel("Time")
plt.ylabel("# of molecules")
plt.ylim(bottom=0)
plt.title("Dimerisation Kinetics (Stochastic), P(0)=301, P2(0)=0, k1=1.66e-3, k2=0.2")
plt.show()
# %% Figure 7.6 (Left)

Pre = np.array([[2,0],[0,1]])
Post = np.array([[0,2],[1,0]])
P = [301, 0] 
rate_constants = {'c1' : 1.66e-3, 'c2' : 0.2}
n = 100 # Number of simulations
start_time = 0
end_time = 10
dt = 0.1
t_reg= np.arange(start_time, end_time + dt/2, dt) 
n_regular_steps = len(t_reg)

simulation_data = []
start = time.time()
for i in range(n):

    output = gillespie(P, end_time + dt, dimeri_kin_stoch, (Post-Pre).T, **rate_constants)
    disc_output = discretise(output, dt=dt, t_start=start_time, t_end=end_time)
    
    simulation_data.append(disc_output['x'])
    
    if i == 0: # Correct check for first iteration
        t_reg = disc_output['t']

end = time.time()

all_states = np.array(simulation_data)
mean_states = np.mean(all_states,axis=0)
var_states = np.var(all_states,axis=0)
print(var_states)
sd_states = np.sqrt(var_states)

t_target = 4.9

idx = int(round((t_target - start_time) / dt))
print(all_states[:,idx])

print(f"Time taken to run {n} simulations = {end - start} seconds!")
plt.plot(t_reg, mean_states[:, 0], label='Sample mean of P')
plt.plot(t_reg, mean_states[:, 0] + 3*sd_states[:, 0], linestyle='--', c='r', label='Sample mean +/- 3 sample SDs')
plt.plot(t_reg, mean_states[:, 0] - 3*sd_states[:, 0], linestyle='--', c='r')
# plt.plot(t_reg, mean_states[:, 1], label='Sample mean of P2')
plt.xlabel('Time')
plt.ylabel('# of molecules')
plt.ylim(bottom=0)
plt.title(f"Mean trajectory of P based on {n} simulations")
plt.legend()
plt.show()

# %%
lv = smfsb.models.lv()

times, states = lv.gillespie(10000)


fig, axis = plt.subplots()
for i in range(2):
    axis.step(times, states[1:, i], where="post")

axis.legend(lv.n)
fig.savefig("s-m-lv-gillespie.pdf")

# %%
def lv(th=[1, 0.1, 0.1]):
    def rhs(x, t):
        return np.array(
            [th[0] * x[0] - th[1] * x[0] * x[1], th[1] * x[0] * x[1] - th[2] * x[1]]
        )

    return rhs

out = smfsb.simple_euler(lv(), np.array([4, 10]), 100)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(np.arange(0, 100, 0.001), out[:, i])
axis.legend(["Prey", "Predator"])
fig.savefig("simple_euler.pdf")
# %%
lamb = 2
alpha = 1
mu = 0.1
sig = 0.2


def my_drift(x, t):
    return np.array([lamb - x[0] * x[1], alpha * (mu - x[1])])


def my_diff(x, t):
    return np.array([[np.sqrt(lamb + x[0] * x[1]), 0], [0, sig * np.sqrt(x[1])]])


step_proc = smfsb.step_sde(my_drift, my_diff, dt=0.001)
out = smfsb.sim_time_series(np.array([1, 0.1]), 0, 30, 0.01, step_proc)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(np.arange(0, 30, 0.01), out[:, i])
# %%
M = 20
N = 30
T = 10
x0 = np.zeros((2, M, N))
lv = smfsb.models.lv()
x0[:, int(M / 2), int(N / 2)] = lv.m
step_lv2d = lv.step_gillespie_2d(np.array([0.6, 0.6]))
x1 = step_lv2d(x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i, :, :])
    axis.set_title(lv.n[i])
# %%
import imageio as iio
from pathlib import Path 
import shutil

base_path = Path("/Users/josh/Documents/SMFSB Python")
save_folder = base_path / "GS_2D_CLE_Frames_NUMPY"
if save_folder.is_dir():
    print(f"Deleting existing folder: {save_folder}")
    shutil.rmtree(save_folder)
save_folder.mkdir(parents=True, exist_ok=True)
print(f"Frames will be saved in: {save_folder}")

Omega = 10000
F = 0.055
k = 0.062
diff_rates = np.array([0.2, 0.1])

gs_sh = f"""
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=0 s
 Pop:V=0 s
@parameters
 N={Omega}
 F={F}
 k={k}
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

gs = smfsb.shorthand_to_spn(gs_sh)
M = 100
N = 100
T = 100
dt = 1.0
n_steps = int(T/dt)
x0 = np.zeros((2, N, M))
x0[0, :, :] = Omega
x0[1, int(M/2)-5:int(M/2)+5, int(N/2)-5:int(N/2)+5] = 0.25 * Omega
x0[0, int(M/2)-5:int(M/2)+5, int(N/2)-5:int(N/2)+5] = 0.25 * Omega

step_gs_cle_2d = gs.step_cle_2d(diff_rates, dt = dt)

print(f"Running JAX simulation for {n_steps} steps...")
start = time.time()
out = smfsb.sim_time_series_2d(x0, 0, T, dt, step_gs_cle_2d, True)
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds!")

print("Saving frames...")
start_save = time.time()
for i in range(out.shape[3]):  

    fig, axes = plt.subplots(1,2)
    
    for j in range(2):

        axes[j].imshow(out[j, :, :, i]) # out = (species, x, y, time)

    filename = f"GS_CLE_Frames_NUMPY_{i:05d}.png"
    
    save_path = save_folder / filename
    
    plt.savefig(save_path)
    plt.close(fig)

end_save = time.time()
print(f"Frames saved in {end_save - start_save:.2f} seconds.")

print("Compiling GIF...")
file_list = sorted(save_folder.glob("*.png"))
frames = [iio.imread(file) for file in file_list]

gif_path = base_path / "GS_CLE_2D_NUMPY.gif"
iio.mimsave(
    gif_path, frames
    )

print(f"GIF saved to {gif_path}")
# %%
