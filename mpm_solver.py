import jax
import jax.numpy as jnp
import numpy as np
import h5py
import os 
from mpm_utils import * 

def initialize(n_particles, n_grid=100, grid_lim=1.0):
    dx = grid_lim / n_grid
    inv_dx = float(n_grid / grid_lim)
    sin_phi = jnp.sin(25.0 / 180.0 * jnp.pi)  # 25 is friction angle (hard coded for now)
    mpm_simulator = {
    'n_particles': n_particles,
    'mpm_model':{
        'grid_lim': grid_lim,
        'n_grid': n_grid,
        'grid_dim_x': n_grid,
        'grid_dim_y': n_grid,
        'grid_dim_z': n_grid,
        'dx': dx,
        'inv_dx': inv_dx,
        'E': jnp.zeros(n_particles),
        'nu': jnp.zeros(n_particles),
        'mu': jnp.zeros(n_particles),
        'lam': jnp.zeros(n_particles),
        'update_cov_with_F': False,
        'material': 0,
        'plastic_viscosity': 0.0,
        'softening': 0.1,
        'yield_stress': jnp.zeros(n_particles),
        'friction_angle': 25.0,
        
        'gravitational_acceleration': jnp.array([0.0, 0.0, 0.0]),
        'rpic_damping': 0.0,
        'grid_v_damping_scale': 1.1,
        'alpha': jnp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
    },

    'mpm_state' :{
        'particle_x': jnp.empty((n_particles, 3)),
        'particle_v': jnp.zeros((n_particles, 3)),
        'particle_F': jnp.zeros((n_particles, 3, 3)),
        'particle_R': jnp.zeros((n_particles, 3, 3)),
        'particle_init_cov': jnp.zeros(n_particles * 6),
        'particle_cov': jnp.zeros(n_particles * 6),
        'particle_F_trial': jnp.zeros((n_particles, 3, 3)),
        'particle_stress': jnp.zeros((n_particles, 3, 3)),
        'particle_vol': jnp.zeros(n_particles),
        'particle_mass': jnp.zeros(n_particles),
        'particle_density': jnp.zeros(n_particles),
        'particle_C': jnp.zeros((n_particles, 3, 3)),
        'particle_Jp': jnp.zeros(n_particles),
        'particle_selection': jnp.zeros(n_particles, dtype=int),
        'grid_m': jnp.zeros((n_grid, n_grid, n_grid)),
        'grid_v_in': jnp.zeros((n_grid, n_grid, n_grid, 3)),
        'grid_v_out': jnp.zeros((n_grid, n_grid, n_grid, 3))
    },
    'time': 0.0,
    'tailored_struct_for_bc':{
        "point": jnp.zeros(3),
        "normal": jnp.zeros(3),
        "start_time": 0.0,
        "end_time": 0.0,
        "friction": 0.0,
        "surface_type": 0,
        "velocity": jnp.zeros(3),
        "threshold": 0.0,
        "reset": 0,
        "point_rotate": jnp.zeros(3),
        "normal_rotate": jnp.zeros(3),
        "x_unit": jnp.zeros(3),
        "y_unit": jnp.zeros(3),
        "radius": 0.0,
        "v_scale": 0.0,
        "width": 0.0,
        "point_plane": jnp.zeros(3),
        "normal_plane": jnp.zeros(3),
        "velocity_plane": jnp.zeros(3),
        "threshold_plane": 0.0
    },
    "grid_postprocess": [],
    "collider_params": [],
    "modify_bc": [],
    "pre_p2g_operations": [],
    "impulse_params": [],
    "particle_velocity_modifiers": [],
    "particle_velocity_modifier_params": []


    }

    return mpm_simulator 

def load_from_sampling(sampling_h5, n_grid=100, grid_lim=1.0):
    if not os.path.exists(sampling_h5):
        raise FileNotFoundError(f"h5 file cannot be found at {os.getcwd() + sampling_h5}")

    with h5py.File(sampling_h5, 'r') as h5file:
        x = h5file["x"][()].T
        particle_volume = np.squeeze(h5file["particle_volume"], 0)

    dim, n_particles = x.shape[1], x.shape[0]
    mpm_sim = initialize(n_particles, n_grid, grid_lim)
    mpm_sim['mpm_state']['particle_x'] = jnp.array(x)
    mpm_sim['mpm_state']['particle_vol'] = jnp.array(particle_volume)
    mpm_sim['dim'] = dim
    mpm_sim['n_particles'] = n_particles
    mpm_sim['mpm_state']['particle_F_trial'] = jnp.array([jnp.eye(3) for _ in range(n_particles)])
    return mpm_sim


def load_initial_data_from_torch(tensor_x, tensor_volume, tensor_cov=None, n_grid=100, grid_lim=1.0):
    dim, n_particles = tensor_x.shape[1], tensor_x.shape[0]
    mpm_sim = initialize(n_particles, n_grid, grid_lim)
    mpm_sim['dim'] = dim 
    mpm_sim['n_particles'] = n_particles


    mpm_sim['mpm_state']['particle_x'] = jnp.array(tensor_x)
    mpm_sim['mpm_state']['particle_vol'] = jnp.array(tensor_volume)

    if tensor_cov is not None:
        mpm_sim['mpm_state']['particle_init_cov'] = jnp.array(tensor_cov.reshape(-1))
        if mpm_sim['mpm_model']['update_cov_with_F']:
            mpm_sim['mpm_state']['particle_cov'] = mpm_sim['mpm_state']['particle_init_cov']
    mpm_sim['mpm_state']['particle_F_trial'] = jnp.array([jnp.eye(3) for _ in range(n_particles)])

    return mpm_sim


def set_parameters_dict(mpm_model, mpm_state, kwargs={}):
    material_map = {
        "jelly": 0,
        "metal": 1,
        "sand": 2,
        "foam": 3,
        "snow": 4,
        "plasticine": 5
    }

    material = kwargs.get("material")
    if material:
        if material in material_map:
            mpm_model['material'] = material_map[material]
        else:
            raise TypeError("Undefined material type")

    grid_lim = kwargs.get("grid_lim", mpm_model['grid_lim'])
    n_grid = kwargs.get("n_grid", mpm_model['n_grid'])

    mpm_model['grid_lim'] = grid_lim
    mpm_model['n_grid'] = n_grid
    mpm_model['grid_dim_x'] = n_grid
    mpm_model['grid_dim_y'] = n_grid
    mpm_model['grid_dim_z'] = n_grid
    mpm_model['dx'], mpm_model['inv_dx'] = grid_lim / n_grid, float(n_grid / grid_lim)

    mpm_state['grid_m'] = jnp.zeros((n_grid, n_grid, n_grid), dtype=float)
    mpm_state['grid_v_in'] = jnp.zeros((n_grid, n_grid, n_grid, 3), dtype=float)
    mpm_state['grid_v_out'] = jnp.zeros((n_grid, n_grid, n_grid, 3), dtype=float)

    if "E" in kwargs:
        mpm_model['E'] = mpm_model['E'].at[:].set(kwargs["E"])
    if "nu" in kwargs:
        mpm_model['nu'] = mpm_model['nu'].at[:].set( kwargs["nu"])
    if "yield_stress" in kwargs:
        mpm_model['yield_stress'] = mpm_model['yield_stress'].at[:].set(kwargs["yield_stress"])
    if "hardening" in kwargs:
        mpm_model['hardening'] = kwargs["hardening"]
    if "xi" in kwargs:
        mpm_model['xi'] = kwargs["xi"]
    if "friction_angle" in kwargs:
        mpm_model['friction_angle'] = kwargs["friction_angle"]
        sin_phi = jnp.sin(mpm_model['friction_angle'] / 180.0 * 3.14159265)
        mpm_model['alpha'] = jnp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

    if "g" in kwargs:
        mpm_model['gravitational_acceleration'] = jnp.array(kwargs["g"])

    if "density" in kwargs:
        density_value = kwargs["density"]
        mpm_state['particle_density'] = mpm_state['particle_density'].at[:].set(density_value)
        mpm_state['particle_mass'] = mpm_state['particle_density'] * mpm_state['particle_vol']
        


    if "rpic_damping" in kwargs:
        mpm_model['rpic_damping'] = kwargs["rpic_damping"]
    if "plastic_viscosity" in kwargs:
        mpm_model['plastic_viscosity'] = kwargs["plastic_viscosity"]
    if "softening" in kwargs:
        mpm_model['softening'] = kwargs["softening"]
    if "grid_v_damping_scale" in kwargs:
        mpm_model['grid_v_damping_scale'] = kwargs["grid_v_damping_scale"]

    return mpm_model, mpm_state




def finalize_mu_lam(mpm_state, mpm_model):
    mpm_state, mpm_model = compute_mu_lam_from_E_nu(mpm_state, mpm_model)
    return mpm_state, mpm_model


def add_surface_collider(mpm_sim,point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0):

    point = jnp.array(point)
    normal = jnp.array(normal)
    normal = normal / jnp.linalg.norm(normal)

    if surface == "sticky" and friction != 0:
        raise ValueError("friction must be 0 on sticky surfaces.")
    
    surface_type = {"sticky": 0, "slip": 1, "cut": 11}.get(surface, 2)
    collider_param = {
        "point": jnp.array(point),
        "normal": jnp.array(normal),
        "surface_type": surface_type,
        "friction": friction,
        "start_time": start_time,
        "end_time": end_time
    }
    mpm_sim['collider_params'].append(collider_param)


    def collide(time, dt, state, model, collider):
        def update_velocity(grid_x, grid_y, grid_z, state, model, collider, time):
            offset = jnp.array([
                grid_x * model['dx'] - collider['point'][0],
                grid_y * model['dx'] - collider['point'][1],
                grid_z * model['dx'] - collider['point'][2]
            ])
            n = collider['normal']
            dotproduct = jnp.dot(offset, n)

            def set_zero_velocity(state):
                state['grid_v_out'] = state['grid_v_out'].at[grid_x, grid_y, grid_z].set(jnp.zeros(3))
                return state 
            
            def handle_surface_type(state, v):
                normal_component = jnp.dot(v, n)
                
                def project_out_normal(v, n):
                    return v - normal_component * n
                
                def project_out_inward_normal(v, n):
                    return v - jnp.minimum(normal_component, 0.0) * n
                
                def apply_friction(v):
                    return jax.lax.cond(
                        normal_component < 0.0,
                        lambda: jnp.maximum(0.0, jnp.linalg.norm(v) + normal_component * collider['friction']) * v / jnp.linalg.norm(v),
                        lambda: v
                    )

                v = jax.lax.cond(
                    collider['surface_type'] == 1,
                    lambda v: project_out_normal(v, n),
                    lambda v: project_out_inward_normal(v, n),
                    v
                )
                v = jax.lax.cond(
                    (normal_component < 0.0) & (jnp.linalg.norm(v) > 1e-20),
                    lambda v: apply_friction(v),
                    lambda v: v,
                    v
                )
                
                state['grid_v_out'] = state['grid_v_out'].at[grid_x, grid_y, grid_z].set(v) 
                return state
            

            def type_11_condition(state):
                v_in = state['grid_v_out'][grid_x, grid_y, grid_z]
                state['grid_v_out'] = state['grid_v_out'].at[grid_x, grid_y, grid_z].set(jnp.array([v_in[0], 0.0, v_in[2]]) * 0.3)
                return state
            
            def inner_condition():
                return jax.lax.cond(
                    collider['surface_type'] == 0,
                    lambda state: set_zero_velocity(state),
                    lambda state: jax.lax.cond(
                        collider['surface_type'] == 11,
                        lambda state: jax.lax.cond(
                            (grid_z * model['dx'] < 0.4) | (grid_z * model['dx'] > 0.53),
                            lambda state: set_zero_velocity(state),
                            lambda state: type_11_condition(state),
                            state
                        ),
                        lambda state: handle_surface_type(state, state['grid_v_out'][grid_x, grid_y, grid_z]),
                        state
                    ),
                    state

                )

            return jax.lax.cond(dotproduct < 0.0, inner_condition, lambda: state)

        grid_shape = state['grid_v_out'].shape[:3]
        grid_x, grid_y, grid_z = jnp.indices(grid_shape)

        def outer_condition(state):
            return jax.lax.cond(
                (time >= collider['start_time']) & (time < collider['end_time']),
                lambda state: jax.lax.fori_loop(0, grid_shape[0] * grid_shape[1] * grid_shape[2], 
                                                 lambda i, state: update_velocity(*jnp.unravel_index(i, grid_shape), state, model, collider, time),
                                                 state),
                lambda state: state,
                state
            )

        return outer_condition(state)
    

    mpm_sim['grid_postprocess'].append(collide)
    mpm_sim['modify_bc'].append(None)
    return mpm_sim













def p2g2p(mpm_sim,step, dt, device="cuda:0"):
    mpm_state = mpm_sim['mpm_state']
    mpm_model = mpm_sim['mpm_model']

    grid_size = (mpm_model['grid_dim_x'], mpm_model['grid_dim_y'], mpm_model['grid_dim_z'])

    
    mpm_state = zero_grid(mpm_state, grid_size)

    for operation,params in zip(mpm_sim['pre_p2g_operations'],mpm_sim['impulse_params']):
        mpm_state = operation(mpm_sim['time'], dt, mpm_state, params)

    for modifier, params in zip(mpm_sim['particle_velocity_modifiers'], mpm_sim['particle_velocity_modifier_params']):
        mpm_state = modifier(mpm_sim['time'], mpm_state, params)

    

    mpm_state = compute_stress_from_F_trial(mpm_state, mpm_model, dt)

    # print(mpm_state['grid_v_in'][51,48,29])

    mpm_state = p2g_apic_with_stress(mpm_state, mpm_model, dt)
    # print(mpm_state['grid_v_in'][51,48,29])

    
    mpm_state = grid_normalization_and_gravity(mpm_state, mpm_model, dt)
    # print(mpm_state['grid_v_in'][51,48,29])

    if mpm_model['grid_v_damping_scale'] < 1.0:
        mpm_state = add_damping_via_grid(mpm_state, mpm_model['grid_v_damping_scale'])

    # print(mpm_state['grid_v_in'][51,48,29])


    for k in range(len(mpm_sim['grid_postprocess'])):
        mpm_state = mpm_sim['grid_postprocess'][k](mpm_sim['time'], dt, mpm_state, mpm_model, mpm_sim['collider_params'][k])
        if mpm_sim['modify_bc'][k] is not None:
            mpm_sim['modify_bc'][k](mpm_sim['time'], mpm_state, mpm_model, mpm_sim['collider_params'][k])

    # print(mpm_state['grid_v_in'][51,48,29])


    mpm_state = g2p(mpm_state, mpm_model, dt)
    # print(mpm_state['grid_v_in'][51,48,29])

    mpm_sim['time'] += dt
    mpm_sim['mpm_state'] = mpm_state

    return mpm_sim

