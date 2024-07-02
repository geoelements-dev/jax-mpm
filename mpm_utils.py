import jax.numpy as jnp
import jax
from jax import jit
import os 
import numpy as np

def zero_grid(state, grid_size):
    state['grid_m'] = jnp.zeros(grid_size)
    state['grid_v_in'] = jnp.zeros(grid_size + (3,))
    state['grid_v_out'] = jnp.zeros(grid_size + (3,))
    return state

def add_damping_via_grid(state, scale):
    state['grid_v_out'] = state['grid_v_out'] * scale
    return state

def compute_mu_lam_from_E_nu(mpm_state, mpm_model):
    E = mpm_model['E']
    nu = mpm_model['nu']
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mpm_model['mu'] = mu
    mpm_model['lam'] = lam
    # jax.debug.print("{v_in_add}", v_in_add=E)

    return mpm_state, mpm_model


def svd3(F):
    U, S, Vh = jnp.linalg.svd(F)
    return U, S, Vh.T


def sand_return_mapping(F_trial, state, model, p):
    U, sig, V = svd3(F_trial)

    epsilon = jnp.log(jnp.clip(jnp.abs(sig), 1e-14))
    tr = jnp.sum(epsilon)
    epsilon_hat = epsilon - tr / 3.0
    epsilon_hat_norm = jnp.linalg.norm(epsilon_hat)
    delta_gamma = epsilon_hat_norm + ((3.0 * model['lam'][p] + 2.0 * model['mu'][p]) / (2.0 * model['mu'][p])) * tr * model['alpha']
    # jax.debug.print("{delta_gamma}", delta_gamma=model['mu'][p])
    F_elastic = jax.lax.select(
        delta_gamma <= 0,
        F_trial,
        jax.lax.select(
            tr > 0,
            jnp.dot(U, V.T),
            jnp.dot(U, jnp.diag(jnp.exp(epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)))) @ V.T
        )
    )

    return F_elastic


def kirchoff_stress_drucker_prager(F, U, V, sig, mu, lam):
    log_sig_sum = jnp.log(sig[0]) + jnp.log(sig[1]) + jnp.log(sig[2])
    center00 = 2.0 * mu * jnp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (1.0 / sig[0])
    center11 = 2.0 * mu * jnp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (1.0 / sig[1])
    center22 = 2.0 * mu * jnp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (1.0 / sig[2])
    center = jnp.array([[center00, 0.0, 0.0], [0.0, center11, 0.0], [0.0, 0.0, center22]])
    result = jnp.dot(jnp.dot(jnp.dot(U, center), V.T), F.T)
    return result



def compute_stress_from_F_trial(state, model, dt):
    def body(p, state):
        F_trial = state['particle_F_trial'][p]

        def compute_if_selected_0(state):
            F = sand_return_mapping(F_trial, state, model, p)
            # jax.debug.print("{v_in_add}", v_in_add=F)

            J = jnp.linalg.det(F)
            U, sig, V = svd3(F)
            stress = kirchoff_stress_drucker_prager(F, U, V, sig, model['mu'][p], model['lam'][p])

            stress = (stress + stress.T) / 2.0  
            state['particle_F'] = state['particle_F'].at[p].set(F)
            state['particle_stress'] = state['particle_stress'].at[p].set(stress)
            return state
        
        def compute_else(state):
            return state
        
        state = jax.lax.cond(state['particle_selection'][p] == 0, compute_if_selected_0, compute_else,state)
        return state

    state = jax.lax.fori_loop(0, state['particle_F'].shape[0], body, state)
    return state


def compute_dweight(model, w, dw, i, j, k):
    dweight = jnp.array([
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k]
    ])
    return dweight * model['inv_dx']



def p2g_apic_with_stress(state, model, dt):
    def body(p, state):
        def compute_if_selected_0(state):
            stress = state['particle_stress'][p]
            grid_pos = state['particle_x'][p] * model['inv_dx']
            base_pos = (grid_pos - 0.5).astype(jnp.int32)
            fx = grid_pos - base_pos
            wa = 1.5 - fx
            wb = fx - 1.0
            wc = fx - 0.5
            w = jnp.array([
                0.5 * wa ** 2,
                0.75 - wb ** 2,
                0.5 * wc ** 2
            ])
            dw = jnp.array([
                fx - 1.5,
                -2.0 * (fx - 1.0),
                fx - 0.5
            ])

            def inner_body(ijk, val):
                i, j, k = jnp.unravel_index(ijk, (3, 3, 3))
                dpos = (jnp.array([i, j, k]) - fx) * model['dx']
                ix, iy, iz = base_pos + jnp.array([i, j, k])
                weight = w[0, i] * w[1, j] * w[2, k]
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state['particle_C'][p]

                def compute_C():
                    return (1.0 - model['rpic_damping']) * C + model['rpic_damping'] / 2.0 * (C - C.T)
                
                def set_zero_C():
                    return jnp.zeros((3, 3))

                C = jax.lax.cond(model['rpic_damping'] < -0.001, set_zero_C, compute_C)

                elastic_force = -state['particle_vol'][p] * stress @ dweight
                v_in_add = (
                    weight * state['particle_mass'][p] * (state['particle_v'][p] + C @ dpos)
                    + dt * elastic_force
                )
                # jax.debug.print("{v_in_add}", v_in_add=C)
                
                val['grid_v_in'] = val['grid_v_in'].at[ix, iy, iz].add(v_in_add)
                val['grid_m'] = val['grid_m'].at[ix, iy, iz].add(weight * state['particle_mass'][p])
                return val

            state = jax.lax.fori_loop(
                0, 27 , inner_body, state
            )
            
            
            return state

        def compute_else(state):
            return state

        state = jax.lax.cond(state['particle_selection'][p] == 0, compute_if_selected_0, compute_else, state)
        return state

    state = jax.lax.fori_loop(0, state['particle_F'].shape[0], body, state)
    return state


def grid_normalization_and_gravity(state, model, dt):
    def body(index, state):
        grid_x, grid_y, grid_z = jnp.unravel_index(index, state['grid_m'].shape)
        def update_velocity(state):
            v_out = state['grid_v_in'][grid_x, grid_y, grid_z] * (
                1.0 / state['grid_m'][grid_x, grid_y, grid_z]
            )

            # jax.debug.print("{v_out}", v_out=state['grid_v_in'][grid_x, grid_y, grid_z])
            v_out = v_out + dt * model['gravitational_acceleration']
            state['grid_v_out'] = state['grid_v_out'].at[grid_x, grid_y, grid_z].set(v_out)
            return state

        def no_update(state):
            return state

        state = jax.lax.cond(
            state['grid_m'][grid_x, grid_y, grid_z] > 1e-15,
            update_velocity,
            no_update,
            state
        )

        return state

    num_elements = state['grid_m'].size
    state = jax.lax.fori_loop(0, num_elements, body, state)
    return state



def update_cov(state, p, grad_v, dt):
    cov_n = jnp.array([
        [state['particle_cov'][p * 6], state['particle_cov'][p * 6 + 1], state['particle_cov'][p * 6 + 2]],
        [state['particle_cov'][p * 6 + 1], state['particle_cov'][p * 6 + 3], state['particle_cov'][p * 6 + 4]],
        [state['particle_cov'][p * 6 + 2], state['particle_cov'][p * 6 + 4], state['particle_cov'][p * 6 + 5]]
    ])
    
    cov_np1 = cov_n + dt * (grad_v @ cov_n + cov_n @ grad_v.T)
    
    state['particle_cov'] = state['particle_cov'].at[p * 6].set(cov_np1[0, 0])
    state['particle_cov'] = state['particle_cov'].at[p * 6 + 1].set(cov_np1[0, 1])
    state['particle_cov'] = state['particle_cov'].at[p * 6 + 2].set(cov_np1[0, 2])
    state['particle_cov'] = state['particle_cov'].at[p * 6 + 3].set(cov_np1[1, 1])
    state['particle_cov'] = state['particle_cov'].at[p * 6 + 4].set(cov_np1[1, 2])
    state['particle_cov'] = state['particle_cov'].at[p * 6 + 5].set(cov_np1[2, 2])
    
    return state


def g2p(state, model, dt):
    def body_fun(p, state):

        def inside_fun(p,state):
            grid_pos = state['particle_x'][p] * model['inv_dx']
            base_pos = jnp.floor(grid_pos - 0.5).astype(int)
            fx = grid_pos - (base_pos)
            
            wa = 1.5 - fx
            wb = fx - 1.0
            wc = fx - 0.5
            w = jnp.array([
                0.5 * wa ** 2,
                0.75 - wb ** 2,
                0.5 * wc ** 2
            ])
            
            dw = jnp.array([
                fx - 1.5,
                -2.0 * (fx - 1.0),
                fx - 0.5
            ])
            
            new_v = jnp.zeros(3)
            new_C = jnp.zeros((3, 3))
            new_F = jnp.zeros((3, 3))
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        ix = base_pos[0] + i
                        iy = base_pos[1] + j
                        iz = base_pos[2] + k
                        dpos = jnp.array([i, j, k]) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]
                        grid_v = state['grid_v_out'][ix, iy, iz]
                        # jax.debug.print(" {ix} {iy} {iz} ", ix=ix, iy=iy, iz=iz)

                        new_v = new_v + grid_v * weight
                        new_C = new_C + jnp.outer(grid_v, dpos) * (weight * model['inv_dx'] * 4.0)
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        new_F = new_F + jnp.outer(grid_v, dweight)
            
            state['particle_v'] = state['particle_v'].at[p].set(new_v)
            new_x = state['particle_x'][p] + dt * new_v
            state['particle_x'] = state['particle_x'].at[p].set(new_x)
            state['particle_C'] = state['particle_C'].at[p].set(new_C)
            
            I33 = jnp.eye(3)
            F_tmp = (I33 + new_F * dt) @ state['particle_F'][p]
            state['particle_F_trial'] = state['particle_F_trial'].at[p].set(F_tmp)
            
            state = jax.lax.cond(
                model['update_cov_with_F'],
                lambda state: update_cov(state, p, new_F, dt),
                lambda state: state,
                state
            )
            return state
        state = jax.lax.cond(state['particle_selection'][p] == 0, lambda state: inside_fun(p,state), lambda state: state, state)
        return state 
    
    state = jax.lax.fori_loop(0, state['particle_x'].shape[0], body_fun, state)
    return state


def save_data_at_frame(mpm_solver, dir_name, frame):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)
    filename = dir_name + '/sim_' + str(frame).zfill(10) + '.ply'
    if os.path.exists(filename):
        os.remove(filename)
    position = np.asarray(mpm_solver['mpm_state']['particle_x'])
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, 'wb') as f: 
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

    