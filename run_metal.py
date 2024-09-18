
from mpm_solver import *
import torch 
import numpy as np 



mpm_solver = add_cube(lower_corner=[1, 1, 1], cube_size=[0.2, 0.2, 0.2],n_grid=128, grid_lim=2,sample_density=3)

print( "Minimum y" , jnp.min(mpm_solver['mpm_state']['particle_x'][:,1]),"Maximum y" , jnp.max(mpm_solver['mpm_state']['particle_x'][:,1]))





material_params = {
    'E': 2000,
    'nu': 0.2,
    "material": "metal",
    'friction_angle': 35,
    'g': [0.0, -9.8, 0.0],
    "density": 1000.0 ,
    'yield_stress': 1000.0,
    
   
}

mpm_solver['mpm_model'],mpm_solver['mpm_state'] = set_parameters_dict(mpm_solver['mpm_model'],mpm_solver['mpm_state'], material_params)

mpm_solver['mpm_state'],mpm_solver['mpm_model'] = finalize_mu_lam(mpm_solver['mpm_state'],mpm_solver['mpm_model']) # set mu and lambda from the E and nu input




mpm_solver = add_grid_boundaries(mpm_solver,padding=8) #padding 8 

mpm_solver = add_surface_collider( mpm_solver , (0.0, 0.5, 0.0), (0.0,1.0,0.0), 'slip',friction=0.3)
# box_length = 0.4
# mpm_solver = add_surface_collider(mpm_solver , (0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
# mpm_solver = add_surface_collider(mpm_solver , (0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
# mpm_solver = add_surface_collider(mpm_solver , (0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
# mpm_solver = add_surface_collider(mpm_solver , (0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
# mpm_solver = add_surface_collider(mpm_solver , (0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

directory_to_save = './sim_results'
pos = [] 

    
for k in range(1,4000):
    mpm_solver = p2g2p(mpm_solver,k,5e-4)
    np_x = mpm_solver['mpm_state']['particle_x'] 
    pos.append(np_x)
    # print( "Minimum y : " , jnp.min(mpm_solver['mpm_state']['particle_x'][:,1]),"| Maximum y : " , jnp.max(mpm_solver['mpm_state']['particle_x'][:,1]))
    # print(np.array(np_x))
    # save_data_at_frame(mpm_solver, directory_to_save, k)


print("Done")

np.save('my_metal_collapse_1.npy', np.array(pos))
