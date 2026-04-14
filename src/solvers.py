import jax
import jax.numpy as jnp
import jax.lax as jl
from jax import jit

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

def make_step_tau(mod):
    V = (jnp.array(mod.post) - jnp.array(mod.pre)).T 

    def step_tau(key, x, t, dt):
        h_vector = mod.h(x, t)
        rates = h_vector * dt
        k_reactions = jax.random.poisson(key, rates)
        
        x_new = x + jnp.dot(V, k_reactions)
        
        return jnp.maximum(x_new, 0)

    return step_tau