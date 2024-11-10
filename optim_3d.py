
from mpm_solver import *
import torch 
import numpy as np 
import optax 
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)



mpm_solver = add_cube( lower_corner=[0.6, 0.6, 0.6], cube_size=[0.2, 0.2, 0.2] , n_grid=32, grid_lim=1,sample_density=4)

print( "Minimum y" , jnp.min(mpm_solver['mpm_state']['particle_x'][:,1]),"Maximum y" , jnp.max(mpm_solver['mpm_state']['particle_x'][:,1]))

material_params = {
    'E': 2000,
    'nu': 0.2,
    "material": "sand",
    'friction_angle': 35,
    'g': [0.0, -9.8, 0.0],
    "density": 200
}

mpm_solver['mpm_model'],mpm_solver['mpm_state'] = set_parameters_dict(mpm_solver['mpm_model'],mpm_solver['mpm_state'], material_params)

mpm_solver['mpm_state'],mpm_solver['mpm_model'] = finalize_mu_lam(mpm_solver['mpm_state'],mpm_solver['mpm_model']) # set mu and lambda from the E and nu input


# mpm_solver = add_grid_boundaries(mpm_solver)
# mpm_solver = add_surface_collider( mpm_solver , (0.0, 0.3, 0), (0.0,1.0,0.0), 'slip', 0)


directory_to_save = './sim_results'

@jax.jit
def mpm_solve(mpm_solver,dt):
    def body(i, ms):
        return p2g2p(ms,i,dt)
    

    mpm_solver = jax.lax.fori_loop(0, 50, body, mpm_solver)
    # mpm_solver = p2g2p(mpm_solver, 0, dt)
    # mpm_solver = p2g2p(mpm_solver, 1, dt)

    return mpm_solver


@jax.jit
def compute_loss(g, mpm, target_vel):
 
    mpm['mpm_model']['gravitational_acceleration'] = g
    result = mpm_solve(mpm, 1e-4)
    vel = result['mpm_state']['particle_v'][0][1]


    loss = jnp.linalg.norm(vel - target_vel)
    return loss


def optax_adam(params, niter, mpm, target_vel):
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)

    param_list = []
    loss_list = []
    t = tqdm(range(niter), desc=f"E: {params}")
    for _ in t:
        lo, grads = jax.value_and_grad(compute_loss)(params, mpm, target_vel)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        t.set_description(f"E: {params}")
        param_list.append(params[1])
        loss_list.append(lo)
    return param_list, loss_list




if __name__ == "__main__":
    try:
        target_vel =  mpm_solve(mpm_solver,1e-4)['mpm_state']['particle_v'][0][1]
        params = jnp.array([0.,-6.,0.])
        param_list, loss_list = optax_adam(params, 70, mpm_solver, target_vel)  

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].plot(param_list, "ko", markersize=2, label="E")
        ax[0].grid()
        ax[0].legend()
        ax[1].plot(loss_list, "ko", markersize=2, label="Loss")
        ax[1].grid()
        ax[1].legend()
        plt.show()
    except:
        with open("exceptions.log", "a") as logfile:
            traceback.print_exc(file=logfile)
        raise


