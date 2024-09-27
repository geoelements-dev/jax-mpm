
import torch
import numpy as np
import mcubes
import argparse


def inverse_strip_lowerdiag(uncertainty):
    batch_size = uncertainty.size(0)
    sym = torch.zeros((batch_size, 3, 3), dtype=torch.float, device=uncertainty.device)
    
    # Fill the upper triangular part of the covariance matrix
    sym[:, 0, 1] = uncertainty[:, 0]
    sym[:, 0, 2] = uncertainty[:, 1]
    sym[:, 1, 2] = uncertainty[:, 2]
    sym[:, 0, 0] = uncertainty[:, 3]
    sym[:, 1, 1] = uncertainty[:, 4]
    sym[:, 2, 2] = uncertainty[:, 5]
    
    # Fill the lower triangular part by copying from the upper triangular part
    sym[:, 1, 0] = sym[:, 0, 1]
    sym[:, 2, 0] = sym[:, 0, 2]
    sym[:, 2, 1] = sym[:, 1, 2]
    
    return sym

def compute_density(index, pos, opacity, cov, grid_dx, device):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = ((index + torch.tensor([i, j, k], device=device)) * grid_dx)
                dist = pos - node_pos
                gaussian_weight += torch.exp(-0.5 * dist.dot(cov @ dist))

    return opacity * gaussian_weight / 8.0

def densify_grids(particles):
    grid = torch.zeros((particles.grid_n, particles.grid_n, particles.grid_n), dtype=torch.int32, device=particles.device)
    grid_density = torch.zeros((particles.grid_n, particles.grid_n, particles.grid_n), dtype=torch.float32, device=particles.device)
    unstripped_cov = inverse_strip_lowerdiag(particles.cov3D_precomp).to(particles.device)
    
    for pi in range(particles.pos.shape[0]):
        pos = particles.pos[pi]
        i, j, k = (pos / particles.grid_dx).long()
        grid[i, j, k] += 1
        cov = unstripped_cov[pi]
        sig, Q = torch.linalg.eigh(cov)
        sig = torch.clamp(sig, min=1e-8)
        sig_mat = torch.diag(1.0 / torch.sqrt(sig))
        cov = Q @ sig_mat @ Q.t()
        r = torch.ceil(torch.max(torch.sqrt(sig)) / particles.grid_dx).int()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (0 <= i + dx < particles.grid_n) and (0 <= j + dy < particles.grid_n) and (0 <= k + dz < particles.grid_n):
                        density = compute_density(torch.tensor([i + dx, j + dy, k + dz], device=particles.device), pos, particles.opacity[pi], cov, particles.grid_dx, particles.device)
                        grid_density[i + dx, j + dy, k + dz] += density.item()

    return grid, grid_density

def fill_dense_grids(particles, grid, grid_density, density_thres=2.0, max_particles_per_cell=1):
    new_particles = torch.zeros((particles.max_samples, 3), dtype=torch.float32, device=particles.device)
    new_start_idx = 0

    for i in range(particles.grid_n):
        for j in range(particles.grid_n):
            for k in range(particles.grid_n):
                if grid_density[i, j, k] > density_thres:
                    if grid[i, j, k] < max_particles_per_cell:
                        diff = max_particles_per_cell - grid[i, j, k]
                        grid[i, j, k] = max_particles_per_cell
                        tmp_start_idx = new_start_idx
                        new_start_idx += diff
                        if tmp_start_idx >= particles.max_samples:
                            break 
                        for index in range(tmp_start_idx, tmp_start_idx + diff):
                            new_particles[index] = (torch.tensor([i, j, k], dtype=torch.float32, device=particles.device) + torch.rand(3, device=particles.device)) * particles.grid_dx

    return new_particles, new_start_idx

def collision_search(grid, grid_density, index, dir_type, size, threshold, device):
    dir = torch.zeros(3, dtype=torch.int32, device=device)
    if dir_type == 0:
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while torch.max(torch.tensor([i, j, k], device=device)) < size and torch.min(torch.tensor([i, j, k], device=device)) >= 0:
        if grid_density[index[0], index[1], index[2]] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index

    return flag

def collision_times(grid, grid_density, index, dir_type, size, threshold, device):
    dir = torch.zeros(3, dtype=torch.int32, device=device)
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:
            dir[0] = 1
        elif dir_type == 1:
            dir[0] = -1
        elif dir_type == 2:
            dir[1] = 1
        elif dir_type == 3:
            dir[1] = -1
        elif dir_type == 4:
            dir[2] = 1
        elif dir_type == 5:
            dir[2] = -1

        state = grid[index[0], index[1], index[2]] > 0
        index += dir
        i, j, k = index
        while torch.max(torch.tensor([i, j, k], device=device)) < size and torch.min(torch.tensor([i, j, k], device=device)) >= 0:
            new_state = grid_density[index[0], index[1], index[2]] > threshold
            if new_state != state and not state:
                times += 1
            state = new_state
            index += dir
            i, j, k = index

    return times

def internal_filling(particles, grid, grid_density, new_particles, start_idx, max_particles_per_cell, exclude_dir, ray_cast_dir, threshold):
    new_start_idx = start_idx
    for i in range(particles.grid_n):
        for j in range(particles.grid_n):
            for k in range(particles.grid_n):
                if grid[i, j, k] == 0:
                    collision_hit = True
                    for dir_type in range(6):
                        if dir_type != exclude_dir:
                            hit_test = collision_search(grid, grid_density, torch.tensor([i, j, k], device=particles.device), dir_type, particles.grid_n, threshold, particles.device)
                            collision_hit = collision_hit and hit_test

                    if collision_hit:
                        hit_times = collision_times(grid, grid_density, torch.tensor([i, j, k], device=particles.device), ray_cast_dir, particles.grid_n, threshold, particles.device)
                        if hit_times % 2 == 1:
                            diff = max_particles_per_cell - grid[i, j, k]
                            grid[i, j, k] = max_particles_per_cell
                            tmp_start_idx = new_start_idx
                            new_start_idx += diff
                            if tmp_start_idx >= particles.max_samples:
                                break
                            for index in range(tmp_start_idx, tmp_start_idx + diff):
                                new_particles[index] = (torch.tensor([i, j, k], dtype=torch.float32, device=particles.device) + torch.rand(3, device=particles.device)) * particles.grid_dx

    return new_start_idx

def get_particle_volume(pos, grid_n: int, grid_dx: float, uniform:bool=False,):
    grid = torch.zeros((grid_n, grid_n, grid_n), dtype=torch.int32)
    particle_vol = torch.zeros(pos.shape[0], dtype=torch.float32)

    for pi in range(pos.shape[0]):
        p = pos[pi]
        i, j, k = (p / grid_dx).long()
        grid[i, j, k] += 1

    for pi in range(pos.shape[0]):
        p = pos[pi]
        i, j, k = (p / grid_dx).long()
        particle_vol[pi] = (grid_dx ** 3) / grid[i, j, k]

    if uniform:
        vol = torch.mean(particle_vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol

def fill_particles(particles, density_thres=2.0, search_thres=1.0, max_particles_per_cell=1, search_exclude_dir=5, ray_cast_dir=4, smooth=False):
    if particles.boundary is not None:
        assert len(particles.boundary) == 6
        mask = torch.ones(particles.pos.shape[0], dtype=torch.bool, device=particles.device)
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, particles.pos[:, i] > particles.boundary[2 * i])
            mask = torch.logical_and(mask, particles.pos[:, i] < particles.boundary[2 * i + 1])
            max_diff = max(max_diff, particles.boundary[2 * i + 1] - particles.boundary[2 * i])

        particles.pos = particles.pos[mask]
        particles.opacity = particles.opacity[mask]
        particles.cov3D_precomp = particles.cov3D_precomp[mask]

        particles.grid_dx = max_diff / particles.grid_n
        new_origin = torch.tensor([particles.boundary[0], particles.boundary[2], particles.boundary[4]], device=particles.device)
        particles.pos = particles.pos - new_origin

    grid, grid_density = densify_grids(particles)
    if smooth:
        grid_density = torch.tensor(mcubes.smooth(grid_density.cpu().numpy(), method="constrained", max_iters=500).astype(np.float32)).to(particles.device)

    new_start_idx = 0

    new_particles,new_start_idx = fill_dense_grids(particles, grid, grid_density, density_thres, max_particles_per_cell)
    new_start_idx = internal_filling(particles, grid, grid_density, new_particles, new_start_idx, max_particles_per_cell, search_exclude_dir, ray_cast_dir, search_thres)

    if particles.boundary is not None:
        new_origin = torch.tensor([particles.boundary[0], particles.boundary[2], particles.boundary[4]], device=particles.device)
        new_particles = new_particles[:new_start_idx] + new_origin
    else:
        new_particles = new_particles[:new_start_idx]

    print(new_particles.shape)
    return torch.cat([particles.pos, new_particles], dim=0) , len(particles.pos)


def get_attr_from_closest(ti_pos, ti_shs, ti_opacity, ti_unactivated_scaling,ti_unactivated_rotation,features_dc,features_rest, ti_new_pos):
    new_shs = torch.zeros(ti_new_pos.shape[0], ti_shs.shape[1], device=ti_new_pos.device)
    new_opacity = torch.zeros(ti_new_pos.shape[0],ti_opacity.shape[1], device=ti_new_pos.device)
    new_unactivated_scaling = torch.zeros(ti_new_pos.shape[0], 3, device=ti_new_pos.device)
    new_unactivated_rotation = torch.zeros(ti_new_pos.shape[0], 4, device=ti_new_pos.device)
    new_features_dc = torch.zeros(ti_new_pos.shape[0], features_dc.shape[1],features_dc.shape[2], device=ti_new_pos.device)
    new_features_rest = torch.zeros(ti_new_pos.shape[0], features_rest.shape[1], features_rest.shape[2],device=ti_new_pos.device)

    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = float('inf')
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = torch.norm(p - ti_pos[pj])
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        new_shs[pi] = ti_shs[min_idx]
        new_opacity[pi] = ti_opacity[min_idx]
        new_unactivated_rotation[pi] = ti_unactivated_rotation[min_idx]
        new_unactivated_scaling[pi] = ti_unactivated_scaling[min_idx]
        new_features_dc[pi] = features_dc[min_idx]
        new_features_rest[pi] = features_rest[min_idx]
        

    return new_shs, new_opacity, new_unactivated_scaling, new_unactivated_rotation, new_features_dc, new_features_rest


def init_filled_particles(pos, shs, scaling,rotation, opacity,features_dc,features_rest, new_pos):
    shs = shs.reshape(pos.shape[0], -1)

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).to(new_pos.device)

    new_shs, new_opacity, new_scaling,new_rotation , new_features_dc,new_features_rest = get_attr_from_closest(pos, shs, opacity, scaling,rotation, features_dc, features_rest,new_pos)

    shs_tensor = torch.cat([shs, new_shs], dim=0).view(-1, 3)
    opacity_tensor = torch.cat([opacity, new_opacity], dim=0).view(-1,1)
    rotation_tensor = torch.cat([rotation, new_rotation], dim=0)
    scaling_tensor = torch.cat([scaling, new_scaling], dim=0)
    features_dc = torch.cat([features_dc, new_features_dc], dim=0)
    features_rest = torch.cat([features_rest, new_features_rest], dim=0)


    return shs_tensor, opacity_tensor, scaling_tensor, rotation_tensor , features_dc, features_rest


if __name__ == "__main__":
    device =  'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    particles = GaussianParticles(grid_n=10,
    grid_dx=0.1,
    max_samples=1000,
    device=device
    )
    particles.load_checkpoint(args.model_path)

    uniform_particles = GaussianParticles()
    uniform_particles.uniformly_init_from_gaussian_particles(
            particles, grid_cell_length=0.01, particles_per_dim=1
    )  

    uniform_particles.save_params_to_file("./uniform_before_internal_filling.ply")
    new_particles_pos , cutoff = fill_particles(uniform_particles)
    # print(uniform_particles.unactivated_rotation.shape,"dhjd")
    
    print(uniform_particles.features_dc.shape)
    shs_tensor, opacity_tensor, scaling_tensor, rotation_tensor , features_dc ,  features_rest = init_filled_particles(new_particles_pos[:cutoff],uniform_particles.shs,uniform_particles.unactivated_scaling,uniform_particles.unactivated_rotation,uniform_particles.unactivated_opacity , uniform_particles.features_dc,uniform_particles.features_rest,new_particles_pos[cutoff:])

    filled_particles = GaussianParticles(
        pos=new_particles_pos,
        shs= shs_tensor,
        unactivated_opacity=opacity_tensor,
        unactivated_rotation=rotation_tensor,
        unactivated_scaling=scaling_tensor,
        features_dc=features_dc,
        features_rest=features_rest,
        grid_n=particles.grid_n,
        grid_dx=particles.grid_dx,
        max_samples=particles.max_samples,
        device=particles.device,
    )

    filled_particles.save_params_to_file("./uniform_after_internal_filling.ply")

