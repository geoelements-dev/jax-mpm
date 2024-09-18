import jax.numpy as jnp
import jax
from jax import jit
import os 
import numpy as np
jax.config.update("jax_enable_x64", True)

from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack


def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def t2j(x_torch):
    x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

def log_array(filename, array_name, array, step, comments=""):
    # Convert to NumPy array if the array is not already one
    if not isinstance(array, np.ndarray):
        array = np.array(array)
        
    with open(filename, 'a') as f:
        f.write(f"Step {step}: {array_name} | Comments: {comments}\n")
        if array.ndim == 0:
            # Handle 0-dimensional array
            f.write(f"{array.item()}\n")
        else:
            np.savetxt(f, array, delimiter=',')
        f.write("\n\n")

@jax.jit
def set_bulk(mpm_model):
    lam = mpm_model['lam']
    mu = mpm_model['mu']
    mpm_model['bulk'] = lam + 2./3. * mu
    return mpm_model


@jax.jit
def zero_grid(state):
    state['grid_m']=state['grid_m'].at[:,:,:].set(0)
    state['grid_v_in']=state['grid_v_in'].at[:,:,:,:].set(0)
    state['grid_v_out']=state['grid_v_out'].at[:,:,:,:].set(0)
    return state

@jax.jit
def kirchoff_stress_water(J, bulk):
    gamma = 1.1


    # clip J between 0.9 and 1.1 
    # J = jnp.clip(J, 0.9, 1.5)
    pressure = -bulk * (jnp.power(J, -gamma) - 1.0)
    # jax.debug.print("pressure = {pressure}", pressure=pressure)
    # jax.debug.print("J = {J}", J=J)

    # jax.debug.print("pressure = {pressure}", pressure=pressure)
    identity_matrix = jnp.eye(3)  # 3x3 identity matrix
    cauchy_stress = identity_matrix * pressure
    return J * cauchy_stress

@jax.jit
def kirchoff_stress_neoHookean(F, U, V, J, sig, mu, lam):
    b = jnp.array([sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2]])
    b_mean = jnp.mean(b)
    b_hat = b - b_mean

    tau = mu * J ** (-2.0 / 3.0) * b_hat + (lam / 2.0) * (J * J - 1.0) * jnp.ones(3)
    
    tau_diag = jnp.diag(tau)  # Diagonal matrix with tau values
    stress = U @ tau_diag @ V.T @ F.T
    
    return stress

@jax.jit
def von_mises_return_mapping(F_trial, model, p):
    # Perform SVD
    U, sig_old, V = svd3(F_trial)

    # Prevent NaN in extreme cases by clamping singular values
    sig = jnp.maximum(sig_old, 0.01)

    # Logarithmic strains (natural strains)
    epsilon = jnp.log(sig)
    temp = jnp.mean(epsilon)

    # Compute tau (stress)
    tau = 2.0 * model['mu'][p] * epsilon + model['lam'][p] * jnp.sum(epsilon) * jnp.ones(3)
    sum_tau = jnp.sum(tau)

    # Deviatoric part of stress
    cond = tau - sum_tau / 3.0

    # Yield condition check
    cond_norm = jnp.linalg.norm(cond)

    # Function to handle plastic correction
    def plastic_correction(_):
        epsilon_hat = epsilon - temp
        epsilon_hat_norm = jnp.linalg.norm(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model['yield_stress'][p] / (2.0 * model['mu'][p])
        
        # Update strain
        epsilon_updated = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat

        # Reconstruct elastic part of F
        sig_elastic = jnp.diag(jnp.exp(epsilon_updated))
        F_elastic = U @ sig_elastic @ V.T

        # Hardening condition
        new_yield_stress = jax.lax.cond(
            model['hardening'] == 1,
            lambda _: model['yield_stress'][p] + 2.0 * model['mu'][p] * model['xi'] * delta_gamma,
            lambda _: model['yield_stress'][p],
            operand=None
        )

        # Return F_elastic and updated yield stress
        return F_elastic, new_yield_stress

    # Function for elastic region, no modification
    def elastic_region(_):
        return F_trial, model['yield_stress'][p]

    # Use lax.cond to select between elastic and plastic behavior
    F_elastic, updated_yield_stress = jax.lax.cond(
        cond_norm > model['yield_stress'][p],
        plastic_correction,  # Plastic case
        elastic_region,      # Elastic case
        operand=None
    )

    # Update yield stress in model (if required)

    return F_elastic , updated_yield_stress


@jax.jit
def compute_R_from_F(mpm_state):
    def body_fun(p, state):
        F = state['particle_F_trial'][p]

        # polar SVD decomposition
        U, sig, V = svd3(F)

        def adjust_u(U):
            return U.at[:, 2].set(-U[:, 2])
        
        def adjust_v(V):
            return V.at[:, 2].set(-V[:, 2])

        U = jax.lax.cond(jnp.linalg.det(U) < 0.0, adjust_u, lambda x: x, U)
        V = jax.lax.cond(jnp.linalg.det(V) < 0.0, adjust_v, lambda x: x, V)

        # Compute rotation matrix
        R = U @ V.T
        # state['particle_R'] = state['particle_R'].at[p].set(R.T)
        return R.T

    vmap_body = jax.vmap(body_fun, in_axes=(0, None))
    mpm_state['particle_R'] = vmap_body(jnp.arange(mpm_state['particle_F_trial'].shape[0]), mpm_state)
    return mpm_state

@jax.jit
def add_damping_via_grid(state, scale):
    state['grid_v_out'] = state['grid_v_out'] * scale
    return state

@jax.jit
def compute_mu_lam_from_E_nu(mpm_state, mpm_model):
    E = mpm_model['E']
    nu = mpm_model['nu']
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mpm_model['mu'] = mu
    mpm_model['lam'] = lam
    # jax.debug.print("{v_in_add}", v_in_add=E)

    return mpm_state, mpm_model

@jax.jit
def svd3(F):
    U, S, Vh = jnp.linalg.svd(F)
    return U, S, Vh.T

@jax.jit
def sand_return_mapping(F_trial, state, model, p):
    U, sig, V = svd3(F_trial)

    epsilon = jnp.array([jnp.log(jnp.clip(jnp.abs(sig[0]),1e-14)), jnp.log(jnp.clip(jnp.abs(sig[1]),1e-14)), jnp.log(jnp.clip(jnp.abs(sig[2]),1e-14))]).T
    tr = jnp.sum(epsilon)
    epsilon_hat = epsilon - tr / 3.0
    epsilon_hat_norm = jnp.linalg.norm(epsilon_hat)
    delta_gamma = epsilon_hat_norm + (3.0 * model['lam'][p] + 2.0 * model['mu'][p]) / (2.0 * model['mu'][p]) * tr * model['alpha']
    # jax.debug.print("{delta_gamma}", delta_gamma=model['mu'][p])
    # jax.debug.callback(log_array,('jax_log.txt', 'delta_gamma', delta_gamma, 0))

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

@jax.jit
def kirchoff_stress_drucker_prager(F, U, V, sig, mu, lam):
    log_sig_sum = jnp.log(sig[0]) + jnp.log(sig[1]) + jnp.log(sig[2])
    center00 = 2.0 * mu * jnp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (1.0 / sig[0])
    center11 = 2.0 * mu * jnp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (1.0 / sig[1])
    center22 = 2.0 * mu * jnp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (1.0 / sig[2])
    center = jnp.array([[center00, 0.0, 0.0], [0.0, center11, 0.0], [0.0, 0.0, center22]])
    result = jnp.dot(jnp.dot(jnp.dot(U, center), V.T), F.T)
    return result

def kirchoff_stress_StVK(F, U, V, sig, mu, lam):
    # Prevent NaN in extreme cases by clamping singular values
    sig = jnp.maximum(sig, 0.01)

    # Compute logarithmic strain (natural strain)
    epsilon = jnp.log(sig)
    log_sig_sum = jnp.sum(jnp.log(sig))

    # Identity vector
    ONE = jnp.ones(3)

    # Compute stress (tau)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE

    # Create a diagonal matrix with tau values
    tau_diag = jnp.diag(tau)

    # Compute Kirchhoff stress using the formula
    kirchoff_stress = U @ tau_diag @ V.T @ F.T

    return kirchoff_stress

@jax.jit
def compute_stress_from_F_trial(state, model, dt):
    def body(p, info):
        state, model = info
        F_trial = state['particle_F_trial'][p]

        
        def compute_if_selected_0(state):
            F, yield_stress = jax.lax.cond(
                model['material']==2,
                lambda x: (sand_return_mapping(x, state, model, p),0.0),
                lambda y: jax.lax.cond(
                model['material']==1,
                lambda x: von_mises_return_mapping(x, model, p),
                lambda x: (x , 0.0),
                y
            ),
                F_trial
            )





            # jax.debug.print("{v_in_add}", v_in_add=F)

            J = jnp.linalg.det(F)
            U, sig, V = svd3(F)
            stress = jax.lax.cond(
                model['material']==2,
                lambda : kirchoff_stress_drucker_prager(F, U, V, sig, model['mu'][p], model['lam'][p]),
                lambda : jax.lax.cond(
                    model['material']==1,
                    lambda : kirchoff_stress_StVK(F, U, V, sig, model['mu'][p], model['lam'][p]),
                    lambda : kirchoff_stress_water(J, model['bulk'][p])
                )
                )


            stress = (stress + stress.T) / 2.0  
            
            return F, stress , yield_stress
        
        def compute_else(info):
            state , model = info
            return state['particle_F'][p], state['particle_stress'][p] , model['yield_stress'][p]
        
        pF,pS , pYS = jax.lax.cond(state['particle_selection'][p] == 0, compute_if_selected_0, compute_else,(state,model))
        return pF,pS , pYS

    vmap_body = jax.vmap(body, in_axes=(0, None))
    state['particle_F'], state['particle_stress'], model['yield_stress'] = vmap_body(jnp.arange(state['particle_F'].shape[0]), (state,model))
    # jax.debug.print("max_F : {max_F}", max_F=jnp.max(state['particle_F']))
    # jax.debug.print("max_stress : {max_stress}", max_stress=jnp.max(state['particle_stress']))
    return state , model 


@jax.jit
def compute_dweight(model, w, dw, i, j, k):
    dweight = jnp.array([
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k]
    ])
    return dweight * model['inv_dx']


# @jax.jit
# def p2g_apic_with_stress(state, model, dt):
#     def body(p, state):
#         def compute_if_selected_0(state):
#             stress = state['particle_stress'][p]
#             grid_pos = state['particle_x'][p] * model['inv_dx']
#             base_pos = (grid_pos - 0.5).astype(jnp.int32)
#             fx = grid_pos - base_pos.astype(jnp.float32)
#             wa = 1.5 - fx
#             wb = fx - 1.0
#             wc = fx - 0.5
#             w = jnp.array([
#                 0.5 * wa ** 2,
#                 0.75 - wb ** 2,
#                 0.5 * wc ** 2
#             ]).T
#             dw = jnp.array([
#                 fx - 1.5,
#                 -2.0 * (fx - 1.0),
#                 fx - 0.5
#             ]).T


#             def inner_body(i,j,k, state):
#                 dpos = (jnp.array([i, j, k]) - fx) * model['dx']
#                 ix, iy, iz = base_pos + jnp.array([i, j, k])
#                 weight = w[0, i] * w[1, j] * w[2, k]
#                 dweight = compute_dweight(model, w, dw, i, j, k)
#                 C = state['particle_C'][p]

#                 def compute_C():
#                     return (1.0 - model['rpic_damping']) * C + model['rpic_damping'] / 2.0 * (C - C.T)
                
#                 def set_zero_C():
#                     return jnp.zeros((3, 3))

#                 C = jax.lax.cond(model['rpic_damping'] < -0.001, set_zero_C, compute_C)

#                 elastic_force = -state['particle_vol'][p] * stress @ dweight
#                 v_in_add = (
#                     weight * state['particle_mass'][p] * (state['particle_v'][p] + C @ dpos)
#                     + dt * elastic_force
#                 )
#                 # jax.debug.print("{v_in_add}", v_in_add=stress)
                
#                 return v_in_add , weight * state['particle_mass'][p] , jnp.array([ix, iy, iz])


#             vmap_body = jax.vmap(
#                 jax.vmap(
#                     jax.vmap(inner_body, in_axes=(None, None, 0, None)),
#                     in_axes=(None, 0, None,None)
#                 ),
#                 in_axes=(0, None, None,None)
#             )

#             # grid_v_in ,grid_m = jnp.zeros_like(state['grid_v_in']), jnp.zeros_like(state['grid_m'])
#             v_in_add , m_in_add , idx = vmap_body(jnp.arange(3), jnp.arange(3), jnp.arange(3), state)

#             # construct 3x3x3 index without for loop

            
#             # jax.debug.print("{v_in_add}", v_in_add=v_in_add.shape)
#             # ds = jax.lax.dynamic_slice(grid_v_in, (base_pos[0],base_pos[1],base_pos[2],0), (3,3,3,3))
#             # grid_v_in = jax.lax.dynamic_update_slice(grid_v_in, ds+v_in_add, (base_pos[0],base_pos[1],base_pos[2],0))


#             # ds2 = jax.lax.dynamic_slice(grid_m, (base_pos[0],base_pos[1],base_pos[2]), (3,3,3))
#             # grid_m = jax.lax.dynamic_update_slice(grid_m, ds2+m_in_add, (base_pos[0],base_pos[1],base_pos[2]))




#             # state['grid_v_in'] = state['grid_v_in'].at[base_pos[0], base_pos[1], base_pos[2], :].add(v_in_add)
            
            
#             return v_in_add , m_in_add , idx 

#         def compute_else(state):
#             return jnp.zeros((3,3,3,3)), jnp.zeros((3,3,3)), jnp.zeros((3,3,3,3),dtype=jnp.int32)
        
#         return jax.lax.cond(state['particle_selection'][p] == 0, compute_if_selected_0, compute_else,state)

#     vmap_body = jax.vmap(body, in_axes=(0, None))
#     g_v_in_acc , g_m_in_acc , idx_acc = vmap_body(jnp.arange(state['particle_F'].shape[0]), state)
#     state['grid_v_in'] = state['grid_v_in'].at[idx_acc[...,0],idx_acc[...,1],idx_acc[...,2],:].add(g_v_in_acc)
#     state['grid_m'] = state['grid_m'].at[idx_acc[...,0],idx_acc[...,1],idx_acc[...,2]].add(g_m_in_acc)
#     return state
        

        



@jax.jit
def p2g_apic_with_stress(state, model, dt):
    def body(p, state):
        def compute_if_selected_0(state):
            stress = state['particle_stress'][p]
            grid_pos = jnp.clip(state['particle_x'][p] * model['inv_dx'], 0.0, model['n_grid'] - 2)

            # def print_debug(grid_pos):
            #     jax.debug.print("ERRROR!!! grid_pos: {grid_pos} , n_grid: {n_grid} , grid_bool: {grid_bool}", grid_pos=grid_pos, n_grid=model['n_grid'],grid_bool=grid_pos+2>=model['n_grid'])
            #     return None
            # def do_nothing(grid_pos):
            #     return None
            # jax.lax.cond(jnp.any(grid_pos+2>=model['n_grid']),print_debug,do_nothing,grid_pos)


            base_pos = (grid_pos - 0.5).astype(jnp.int64)

            
            fx = grid_pos - base_pos.astype(jnp.float64)
            wa = 1.5 - fx
            wb = fx - 1.0
            wc = fx - 0.5
            w = jnp.array([
                0.5 * wa ** 2,
                0.75 - wb ** 2,
                0.5 * wc ** 2
            ]).T
            dw = jnp.array([
                fx - 1.5,
                -2.0 * (fx - 1.0),
                fx - 0.5
            ]).T


            def inner_body(i,j,k, state):
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
                # jax.debug.print("{v_in_add}", v_in_add=stress)
                
                return v_in_add , weight * state['particle_mass'][p] , jnp.array([ix, iy, iz])


            vmap_body = jax.vmap(
                jax.vmap(
                    jax.vmap(inner_body, in_axes=(None, None, 0, None)),
                    in_axes=(None, 0, None,None)
                ),
                in_axes=(0, None, None,None)
            )

            # grid_v_in ,grid_m = jnp.zeros_like(state['grid_v_in']), jnp.zeros_like(state['grid_m'])
            v_in_add , m_in_add , idx = vmap_body(jnp.arange(3), jnp.arange(3), jnp.arange(3), state)

            # construct 3x3x3 index without for loop

            
            # jax.debug.print("{v_in_add}", v_in_add=v_in_add.shape)
            # ds = jax.lax.dynamic_slice(grid_v_in, (base_pos[0],base_pos[1],base_pos[2],0), (3,3,3,3))
            # grid_v_in = jax.lax.dynamic_update_slice(grid_v_in, ds+v_in_add, (base_pos[0],base_pos[1],base_pos[2],0))


            # ds2 = jax.lax.dynamic_slice(grid_m, (base_pos[0],base_pos[1],base_pos[2]), (3,3,3))
            # grid_m = jax.lax.dynamic_update_slice(grid_m, ds2+m_in_add, (base_pos[0],base_pos[1],base_pos[2]))




            # state['grid_v_in'] = state['grid_v_in'].at[base_pos[0], base_pos[1], base_pos[2], :].add(v_in_add)
            
            
            return v_in_add , m_in_add , idx 

        def compute_else(state):
            return jnp.zeros((3,3,3,3)), jnp.zeros((3,3,3)), jnp.zeros((3,3,3,3),dtype=jnp.int64)
        
        return jax.lax.cond(state['particle_selection'][p] == 0, compute_if_selected_0, compute_else,state)

    vmap_body = jax.vmap(body, in_axes=(0, None))
    g_v_in_acc , g_m_in_acc , idx_acc = vmap_body(jnp.arange(state['particle_F'].shape[0]), state)
    state['grid_v_in'] = state['grid_v_in'].at[idx_acc[...,0],idx_acc[...,1],idx_acc[...,2],:].add(g_v_in_acc)
    state['grid_m'] = state['grid_m'].at[idx_acc[...,0],idx_acc[...,1],idx_acc[...,2]].add(g_m_in_acc)
    return state
        
@jax.jit
def g2p(state, model, dt):
    def body_fun(p, state):

        def inside_fun(p,state):
            grid_pos = jnp.clip(state['particle_x'][p] * model['inv_dx'], 0.0, model['n_grid'] - 2)
            base_pos = (grid_pos - 0.5).astype(jnp.int64) # Check the truncating , floor ? 
            fx = grid_pos - (base_pos).astype(jnp.float64)
            wa = 1.5 - fx
            wb = fx - 1.0
            wc = fx - 0.5
            w = jnp.array([
                0.5 * wa ** 2,
                0.75 - wb ** 2,
                0.5 * wc ** 2
            ]).T
            
            dw = jnp.array([
                fx - 1.5,
                -2.0 * (fx - 1.0),
                fx - 0.5
            ]).T
            
            
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
            
            # jax.debug.print("new_F: {x}", x=new_F)
            # jax.debug.print("new_C: {x}", x=new_C)
            new_x = state['particle_x'][p] + dt * new_v
            I33 = jnp.eye(3)
            F_tmp = jnp.matmul((I33 + new_F * dt) , state['particle_F'][p],precision=jax.lax.Precision.HIGHEST)

            # jax.debug.print("F_tmp: {x}", x=F_tmp)

            
            # state = jax.lax.cond(
            #     model['update_cov_with_F'],
            #     lambda state: update_cov(state, p, new_F, dt),
            #     lambda state: state,
            #     state
            # )

            new_cov = jax.lax.cond(
                model['update_cov_with_F'],
                lambda state: update_cov(state, p, new_F, dt),
                lambda state: jax.lax.dynamic_slice(state['particle_cov'], (p*6,), (6,)),
                state
            )

            return new_v, new_x, new_C, F_tmp , new_cov 
        
        def identity_fun(p,state):
            return state['particle_v'][p], state['particle_x'][p], state['particle_C'][p], state['particle_F_trial'][p] ,  jax.lax.dynamic_slice(state['particle_cov'], (p*6,), (6,))
        
        pv,px,pc,pf , pcov = jax.lax.cond(state['particle_selection'][p] == 0, lambda state: inside_fun(p,state), lambda state : identity_fun(p,state), state)
        return pv,px,pc,pf , pcov
    
    vmap_body = jax.vmap(body_fun, in_axes=(0, None))
    state['particle_v'], state['particle_x'], state['particle_C'], state['particle_F_trial'] , pcov = vmap_body(jnp.arange(state['particle_x'].shape[0]), state)
    state['particle_cov'] = jnp.ravel(pcov)
    # state = jax.lax.fori_loop(0, state['particle_x'].shape[0], body_fun, state)
    return state


@jax.jit
def grid_normalization_and_gravity(state, model, dt):
    def update_velocity( grid_v_in, grid_m):
        v_out = grid_v_in * (1.0 / grid_m)
        v_out = v_out + dt * model['gravitational_acceleration']
        return v_out

    def no_update(grid_v_in,grid_m):
        return grid_v_in

    grid_shape = state['grid_m'].shape
    grid_indices = jnp.arange(state['grid_m'].size).reshape(grid_shape)
    
    def body(grid_x, grid_y, grid_z):
        cond = state['grid_m'][grid_x, grid_y, grid_z] > 1e-15
        v_out = jax.lax.cond(
            cond,
            update_velocity,
            no_update,
            state['grid_v_in'][grid_x, grid_y, grid_z],
            state['grid_m'][grid_x, grid_y, grid_z]
        )
        return v_out , jnp.array([grid_x, grid_y, grid_z])

    vmapped_body = jax.vmap(
        jax.vmap(
            jax.vmap(body, in_axes=(None, None, 0)),
            in_axes=(None, 0, None)
        ),
        in_axes=(0, None, None)
    )

    g_v , idx_acc =  vmapped_body(jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]), jnp.arange(grid_shape[2]))
    state['grid_v_out'] = state['grid_v_out'].at[idx_acc[...,0],idx_acc[...,1],idx_acc[...,2],:].set(g_v)

    return state


@jax.jit
def update_cov(state, p, grad_v, dt):
    cov_n = jnp.array([
        [state['particle_cov'][p * 6], state['particle_cov'][p * 6 + 1], state['particle_cov'][p * 6 + 2]],
        [state['particle_cov'][p * 6 + 1], state['particle_cov'][p * 6 + 3], state['particle_cov'][p * 6 + 4]],
        [state['particle_cov'][p * 6 + 2], state['particle_cov'][p * 6 + 4], state['particle_cov'][p * 6 + 5]]
    ])

    # jax.debug.print("{v_in_add}", v_in_add=grad_v)
    # jax.debug.print("{v_in_add}", v_in_add=cov_n)
    
    cov_np1 = cov_n + dt * (grad_v @ cov_n + cov_n @ grad_v.T)
    
    # state['particle_cov'] = state['particle_cov'].at[p * 6].set(cov_np1[0, 0])
    # state['particle_cov'] = state['particle_cov'].at[p * 6 + 1].set(cov_np1[0, 1])
    # state['particle_cov'] = state['particle_cov'].at[p * 6 + 2].set(cov_np1[0, 2])
    # state['particle_cov'] = state['particle_cov'].at[p * 6 + 3].set(cov_np1[1, 1])
    # state['particle_cov'] = state['particle_cov'].at[p * 6 + 4].set(cov_np1[1, 2])
    # state['particle_cov'] = state['particle_cov'].at[p * 6 + 5].set(cov_np1[2, 2])
    
    return jnp.array([cov_np1[0, 0], cov_np1[0, 1], cov_np1[0, 2], cov_np1[1, 1], cov_np1[1, 2], cov_np1[2, 2]])
    # return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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

    