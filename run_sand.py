
from mpm_solver import *
import torch 
import numpy as np 



mpm_solver = add_cube( lower_corner=[0.6, 0.6, 0.6], cube_size=[0.2, 0.2, 0.2] , n_grid=32, grid_lim=1,sample_density=4)

print( "Minimum y" , jnp.min(mpm_solver['mpm_state']['particle_x'][:,1]),"Maximum y" , jnp.max(mpm_solver['mpm_state']['particle_x'][:,1]))
# mpm_solver = load_from_sampling("sand_column.h5", n_grid = 150) 


# volume_tensor = torch.ones(mpm_solver['n_particles']) * 2.5e-8

# np_array = np.asarray(mpm_solver['mpm_state']['particle_x'])
# position_tensor = torch.from_numpy(np_array)



# mpm_solver = load_initial_data_from_torch(position_tensor, volume_tensor)

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

print(mpm_solver['mpm_model']['mu'], mpm_solver['mpm_model']['lam'])

mpm_solver = add_grid_boundaries(mpm_solver)
mpm_solver = add_surface_collider( mpm_solver , (0.0, 0.3, 0), (0.0,1.0,0.0), 'slip', 0)


directory_to_save = './sim_results'
pos = [] # list of positions

    
    
# save_data_at_frame(mpm_solver, directory_to_save, 0)
# print(mpm_solver['mpm_model']['mu'], mpm_solver['mpm_model']['lam'])
mpm_solver['mpm_state']['particle_x'] = mpm_solver['mpm_state']['particle_x'] 
for k in range(1,2000):
    mpm_solver = p2g2p(mpm_solver,k,3e-4)
    np_x = mpm_solver['mpm_state']['particle_x'] 
    pos.append(np_x)
    # print( "Minimum y : " , jnp.min(mpm_solver['mpm_state']['particle_x'][:,1]),"| Maximum y : " , jnp.max(mpm_solver['mpm_state']['particle_x'][:,1]))
    # print(np.array(np_x))
    # save_data_at_frame(mpm_solver, directory_to_save, k)


print("Done")

np.save('my_sand_collapse_10.npy', np.array(pos))
