
from mpm_solver import *
import torch 
import numpy as np 



n_particles = 1000
mpm_solver = initialize(n_particles) 

mpm_solver = load_from_sampling("sand_column.h5", n_grid = 150) 


volume_tensor = torch.ones(mpm_solver['n_particles']) * 2.5e-8

np_array = np.asarray(mpm_solver['mpm_state']['particle_x'])
position_tensor = torch.from_numpy(np_array)



mpm_solver = load_initial_data_from_torch(position_tensor, volume_tensor)


material_params = {
    'E': 2000,
    'nu': 0.2,
    "material": "sand",
    'friction_angle': 35,
    'g': [0.0, 0.0, -4.0],
    "density": 200.0
}
mpm_solver['mpm_model'],mpm_solver['mpm_state'] = set_parameters_dict(mpm_solver['mpm_model'],mpm_solver['mpm_state'], material_params)

mpm_solver['mpm_state'],mpm_solver['mpm_model'] = finalize_mu_lam(mpm_solver['mpm_state'],mpm_solver['mpm_model']) # set mu and lambda from the E and nu input

mpm_solver = add_surface_collider( mpm_solver , (0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)


directory_to_save = './sim_results'

# save_data_at_frame(mpm_solver, directory_to_save, 0)
print(jax.default_backend())
for k in range(1,100):
    mpm_solver = p2g2p(mpm_solver,k, 0.002)
    save_data_at_frame(mpm_solver, directory_to_save, k)

