import jax
import jax.numpy as jnp
from jax import jit, device_put
import numpy as np
import mcubes
import torch 
import plotly.graph_objects as go
from mpm_utils import t2j, j2t
from scipy.spatial import cKDTree



# def find_closest_attributes(p, pos, shs, opacity, cov, dist_batch_size=1024):
#     min_dist = float('inf')
#     min_idx = 0

#     # Process `pos` in smaller batches to manage memory better
#     for i in range(0, pos.shape[0], dist_batch_size):
#         pos_batch = pos[i:i + dist_batch_size]
#         print("pbatch: "  , pos_batch)
#         print("P: ", p)
#         print("Diff ", pos_batch - p)
#         # Calculate distances for the current batch
#         dists = jnp.linalg.norm(pos_batch - p, axis=1)
        
#         # Find the minimum distance and corresponding index in this batch
#         batch_min_idx = jnp.argmin(dists)
#         batch_min_dist = dists[batch_min_idx]
        
#         # Update global minimum if the batch minimum is smaller
#         if batch_min_dist < min_dist:
#             min_dist = batch_min_dist
#             min_idx = i + batch_min_idx

#     # Retrieve the attributes for the closest point found
#     return shs[min_idx], opacity[min_idx], cov[min_idx]

# # Vectorized version for processing multiple points in `new_pos`
# get_attr_from_closest_batch = jax.vmap(
#     find_closest_attributes,
#     in_axes=(0, None, None, None, None)
# )

# def get_attr_from_closest(new_pos, pos, shs, opacity, cov, batch_size=1024):
#     # Initialize lists to accumulate results for each batch
#     new_shs_batches = []
#     new_opacity_batches = []
#     new_cov_batches = []

#     # Process `new_pos` in batches
#     print("Shape of new_pos: ", new_pos.shape)
#     for i in range(0, new_pos.shape[0], batch_size):
#         print("Batch: ", i)
#         print("Indices of new_pos: ", i, ":", min(i + batch_size, new_pos.shape[0]))
#         batch_new_pos = new_pos[i:max(i + batch_size, new_pos.shape[0])]
        
#         # Apply the vectorized function to the batch
#         batch_new_shs, batch_new_opacity, batch_new_cov = get_attr_from_closest_batch(
#             batch_new_pos, pos, shs, opacity, cov
#         )
        
#         # Append results to the list
#         new_shs_batches.append(batch_new_shs)
#         new_opacity_batches.append(batch_new_opacity)
#         new_cov_batches.append(batch_new_cov)
    
#     # Concatenate results from all batches
#     new_shs = jnp.concatenate(new_shs_batches, axis=0)
#     new_opacity = jnp.concatenate(new_opacity_batches, axis=0)
#     new_cov = jnp.concatenate(new_cov_batches, axis=0)

#     return new_shs, new_opacity, new_cov



def build_kd_tree(pos):
    # Convert `pos` to a NumPy array for compatibility with cKDTree
    pos_np = jnp.asarray(pos)
    return cKDTree(pos_np)

def get_attr_from_closest(new_pos, kdtree, pos, shs, opacity, cov):
    # Query the k-D tree to find the closest point in `pos` for each point in `new_pos`
    _, indices = kdtree.query(new_pos)

    # Gather attributes based on the nearest neighbor indices
    new_shs = shs[indices]
    new_opacity = opacity[indices]
    new_cov = cov[indices]

    return new_shs, new_opacity, new_cov



def init_filled_particles(pos, shs, cov, opacity, new_pos):
    # Reshape shs to a 2D array (num_particles, attributes)
    pos = t2j(pos)
    shs = t2j(shs)
    cov = t2j(cov)
    opacity = t2j(opacity)
    new_pos = t2j(new_pos)


    shs = shs.reshape(pos.shape[0], -1)
    
    # Reshape position, covariance, and opacity arrays
    pos = pos.reshape(-1, 3)
    cov = cov.reshape(-1, 6)
    opacity = opacity.reshape(-1,1)

    # Initialize new_shs with the mean of `shs`, repeated for the number of new positions
    new_shs = jnp.mean(shs, axis=0).repeat(new_pos.shape[0]).reshape(new_pos.shape[0], -1)
    
    # Calculate attributes of the closest particles
    kdtree = build_kd_tree(pos)

    new_shs, new_opacity, new_cov = get_attr_from_closest(new_pos,kdtree, pos, shs, opacity, cov)

    # Concatenate original and new attributes
    shs_combined = jnp.concatenate([shs, new_shs], axis=0).reshape(-1, shs.shape[1] // 3, 3)
    opacity_combined = jnp.concatenate([opacity, new_opacity.reshape(-1, 1)], axis=0)
    cov_combined = jnp.concatenate([cov, new_cov], axis=0)

    shs_combined = j2t(shs_combined)
    opacity_combined = j2t(opacity_combined)
    cov_combined = j2t(cov_combined)

    return shs_combined, opacity_combined, cov_combined


def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    shifts = jnp.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    
    for shift in shifts:
        node_pos = (index + shift) * grid_dx
        dist = pos - node_pos
        gaussian_weight += jnp.exp(-0.5 * jnp.dot(dist, cov @ dist))
        # print("Calculated log gweight: ",  -0.5 * jnp.dot(dist, cov @ dist))
    return opacity * gaussian_weight / 8.0



def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    boundary: list = None,
    smooth: bool = False,
):
    pos_clone = jnp.copy(pos)
    
    # Apply boundary conditions
    if boundary is not None:
        assert len(boundary) == 6
        mask = jnp.ones(pos_clone.shape[0], dtype=bool)
        max_diff = 0.0
        
        for i in range(3):
            mask = jnp.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = jnp.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = jnp.maximum(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        grid_dx = max_diff / grid_n
        new_origin = jnp.array([boundary[0], boundary[2], boundary[4]])
        pos = pos - new_origin

    # JAX arrays instead of Taichi fields
    grid = jnp.zeros((grid_n, grid_n, grid_n), dtype=int)
    grid_density = jnp.zeros((grid_n, grid_n, grid_n), dtype=float)
    particles = jnp.zeros((max_samples, 3), dtype=float)
    fill_num = 0

    x, y, z = jnp.meshgrid(jnp.arange(grid_n), jnp.arange(grid_n), jnp.arange(grid_n), indexing='ij')

    
    
    # Compute density_field
    grid, grid_density = densify_grids(pos, opacity, cov, grid, grid_density, grid_dx)
    print("Max grid density: ", jnp.max(grid_density))
    # Fill dense grids


    

    fill_num , grid , particles = fill_dense_grids(
        grid,
        grid_density,
        grid_dx,
        density_thres,
        particles,
        0,
        max_particles_per_cell,
        rng_key=jax.random.PRNGKey(0)
    )
    print("after dense grids: ", fill_num)

    

    # Smooth density_field (optional)
    if smooth:
        df = np.array(grid_density)
        smoothed_df = mcubes.smooth(df, method = "constrained" ,  max_iters=500).astype(np.float32)
        grid_density = jnp.array(smoothed_df)
        print("smooth finished")

    
    isomin = float(grid_density.min())
    isomax = float(grid_density.max())

    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=grid_density.flatten(),
        isomin=isomin,
        isomax=isomax,
        opacity=0.1,  # Adjust opacity for better visualization
        surface_count=15,  # Number of isosurfaces
    ))

    fig.show()

    # Fill internal grids
    particles , fill_num = internal_filling(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
        key=jax.random.PRNGKey(0)
    )
    print("after internal grids: ", fill_num)

    # Concatenate new particles with the original ones
    particles_tensor = particles[:fill_num]
    
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
        
    particles_tensor = jnp.concatenate([pos_clone, particles_tensor], axis=0)

    return particles_tensor


 
def densify_grids(init_particles, opacity, cov_upper, grid, grid_density, grid_dx , batch_size=1024):
    print("Shape of init_particles: ", init_particles.shape)
    print("Shape of cov: ", cov_upper.shape)
    print("Max value in cov", jnp.max(cov_upper))

    def cov_matrix(cov_upper_row):
        cov = jnp.array([
            [cov_upper_row[0], cov_upper_row[1], cov_upper_row[2]],
            [cov_upper_row[1], cov_upper_row[3], cov_upper_row[4]],
            [cov_upper_row[2], cov_upper_row[4], cov_upper_row[5]]
        ]).T
        return cov

    def sym_eig_and_r(cov_in):
        # Perform eigen decomposition
        sig, Q = jnp.linalg.eigh(cov_in)
        
        # Ensure positive definiteness by clamping eigenvalues
        sig = jnp.maximum(sig, 1e-8)

        # Inverse sqrt of the eigenvalues matrix
        sig_inv = jnp.diag(1.0 / sig)
        
        # Recalculate the covariance matrix
        cov_new = Q @ sig_inv @ Q.T

        # Maximum radius in voxel units
        r = jnp.ceil(jnp.max(jnp.sqrt(jnp.abs(sig))) / grid_dx).astype(int)
        return cov_new, r

    def body_1(pi):
        pos = init_particles[pi]
        x, y, z = pos
        
        # Compute grid indices
        i = jnp.floor(x / grid_dx).astype(int)
        j = jnp.floor(y / grid_dx).astype(int)
        k = jnp.floor(z / grid_dx).astype(int)

        grid_pos = jnp.array([i, j, k])


        # Covariance matrix from upper-triangle
        cov = cov_matrix(cov_upper[pi])
        # Eigen decomposition and radius calculation
        cov, r = sym_eig_and_r(cov)

        
        return grid_pos, cov, r 


    def compute_filtered_voxel_positions(n_particles, r_all, voxel_base_all, batch_size, batch_no):
        # Calculate the maximum radius to cover all possible offsets
        max_r = jnp.max(r_all)
        offset_range = jnp.arange(-max_r, max_r + 1)

        # Generate offsets from the meshgrid
        dx, dy, dz = jnp.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
        offsets = jnp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)  # Shape: (n_offsets, 3)

        # Determine start and end indices for the current batch
        start_idx = batch_no * batch_size
        end_idx = jnp.minimum(start_idx + batch_size, n_particles)

        # Slice r_all and voxel_base_all for the current batch
        r_batch = r_all[start_idx:end_idx]  # Shape: (batch_size,)
        voxel_base_batch = voxel_base_all[start_idx:end_idx]  # Shape: (batch_size, 3)

        # Expand dimensions to prepare for broadcasting within the batch
        r_batch_expanded = r_batch[:, None, None]  # Shape: (batch_size, 1, 1)
        voxel_base_batch_expanded = voxel_base_batch[:, None, :]  # Shape: (batch_size, 1, 3)

        # Compute voxel positions by adding offsets to each particle's voxel_base
        voxel_pos_batch = voxel_base_batch_expanded + offsets  # Shape: (batch_size, n_offsets, 3)

        # Create mask to filter positions based on each particle's radius r
        within_range = (jnp.abs(offsets[:, 0][None, :]) <= r_batch_expanded) & \
                    (jnp.abs(offsets[:, 1][None, :]) <= r_batch_expanded) & \
                    (jnp.abs(offsets[:, 2][None, :]) <= r_batch_expanded)

        # print("within_range: ", within_range)
        # print("Shape of within_range: ", within_range.shape)
        # Get the indices of the valid positions for each particle in the batch
        valid_indices = jnp.nonzero(within_range)  # Returns tuple of indices

        # print("Shape of voxel_pos_batch: ", voxel_pos_batch.shape)
        # print("Length of valid_indices: ", len(valid_indices))
        # print("Shape of valid_indices: ", valid_indices[0].shape)
        # Extract the valid positions and particle indices
        filtered_voxel_positions_batch = voxel_pos_batch[valid_indices[0],valid_indices[2]]  
        # print("filtered_voxel_positions_batch: ", filtered_voxel_positions_batch)
        particle_indices_expanded = (start_idx + valid_indices[0]).reshape(-1, 1)
        
        # Concatenate voxel positions with particle indices
        filtered_voxel_positions_with_indices = jnp.concatenate(
            [filtered_voxel_positions_batch, particle_indices_expanded], axis=-1
        )  # Shape: (valid_positions_count, 4)

        return filtered_voxel_positions_with_indices

    vmap_body_1 = jax.vmap(body_1)
    grid_pos_all, cov_all, r_all  = vmap_body_1(jnp.arange(init_particles.shape[0]))
    grid = grid.at[grid_pos_all[...,0],grid_pos_all[...,1],grid_pos_all[...,2]].add(1)
    print("Max particles in grid: ", jnp.max(grid))

    for k in range(0,(-(-grid_pos_all.shape[0]//batch_size))):
        voxel_pos_all = compute_filtered_voxel_positions(init_particles.shape[0], r_all, grid_pos_all,batch_no=k, batch_size=batch_size)
        def body_fun(i,vp):  
            dx , dy, dz , pi = vp[i]     
            # print("dx: ", dx , "dy: ", dy, "dz: ", dz, "pi: ", pi)
            # Define conditions to check grid bounds
            in_bounds = (0 <= dx) & (dx < grid_density.shape[0]) & \
                        (0 <= dy) & (dy < grid_density.shape[1]) & \
                        (0 <= dz) & (dz < grid_density.shape[2])
            
            # Calculate density only if in bounds, otherwise return 0
            density = jnp.where(
                in_bounds,
                compute_density(jnp.array([dx, dy, dz]), init_particles[pi], opacity[pi], cov_all[pi], grid_dx)[0],
                0.0
            )

            return density, jnp.array([dx, dy, dz])
        

                    

        vmap_body = jax.vmap(body_fun, in_axes=(0,None))
                    
        density , voxel_pos = vmap_body(jnp.arange(voxel_pos_all.shape[0]),voxel_pos_all)
        grid_density = grid_density.at[voxel_pos[...,0],voxel_pos[...,1],voxel_pos[...,2]].add(density)
    return grid,grid_density 

    

        

       
    


def collision_search(grid_density, index, dir_type, size, threshold):
    # Define the search direction based on `dir_type`
    directions = jnp.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    direction = directions[dir_type]

    # Update index based on the initial direction
    index = index + direction

    def cond_fn(state):
        idx, flag = state
        i, j, k = idx
        within_bounds = (jnp.max(idx) < size) & (jnp.min(idx) >= 0)
        return within_bounds & ~flag

    def body_fn(state):
        idx, flag = state
        # Check if density at the current index is above the threshold
        flag = jnp.where(grid_density[tuple(idx)] > threshold, True, flag)
        return idx + direction, flag

    # Initialize the state with the index and a flag indicating no collision detected
    initial_state = (index, False)

    # Run the loop until the condition fails or a collision is detected
    _, collision_detected = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return collision_detected


def collision_times(grid, grid_density, index, dir_type, size, threshold):
    # Define directions
    directions = jnp.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    times = jnp.where((dir_type > 5) | (dir_type < 0), 1, 0)
    
    # Exit early if direction is invalid
    if times == 1:
        return times
    
    direction = directions[dir_type]
    
    # Initialize state variables
    state = grid[tuple(index)] > 0
    index = index + direction

    def cond_fn(state_tuple):
        idx, _, _ = state_tuple
        i, j, k = idx
        within_bounds = (jnp.max(idx) < size) & (jnp.min(idx) >= 0)
        return within_bounds

    def body_fn(state_tuple):
        idx, times, state = state_tuple
        i, j, k = idx
        
        # Determine the new state based on grid density threshold
        new_state = grid_density[tuple(idx)] > threshold
        times += jnp.where((new_state != state) & ~state, 1, 0)
        return idx + direction, times, new_state

    # Initialize the state for the loop
    initial_state = (index, times, state)
    
    # Execute the loop
    _, collision_times, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return collision_times


def fill_dense_grids(
    grid,
    grid_density,
    grid_dx,
    density_thres,
    new_particles,
    start_idx,
    max_particles_per_cell,
    rng_key
):
    # Step 1: Compute `diff_all` (number of particles needed per cell based on density threshold)
    def compute_diff(density, grid_val):
        return jnp.maximum(0, max_particles_per_cell - grid_val) * (density > density_thres)

    # Vectorized calculation of `diff_all` across grid
    diff_all = jax.vmap(compute_diff)(grid_density.ravel(), grid.ravel())  # Shape: (n_cells,)
    # Step 2: Repeat grid indices based on `diff_all`
    grid_shape = grid.shape
    grid_indices = jnp.array(jnp.meshgrid(
        jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]), jnp.arange(grid_shape[2]),
        indexing="ij"
    )).reshape(3, -1).T  # Shape: (n_cells, 3)
    
    # Expanding grid indices based on `diff_all`
    repeated_indices = jnp.repeat(grid_indices, diff_all, axis=0)  # Shape: (total_particles, 3)
    
    # Step 3: Generate random displacements
    total_particles = repeated_indices.shape[0]
    random_disps = jax.random.uniform(rng_key, shape=(total_particles, 3)) * grid_dx  # Shape: (total_particles, 3)

    # Step 4: Compute new particle positions and concatenate with `new_particles`
    new_particle_positions = repeated_indices * grid_dx + random_disps  # Shape: (total_particles, 3)

    # Step 5: Update grid where particles were added
    def update_grid_val(grid_val, diff):
        return jnp.where(diff > 0, max_particles_per_cell, grid_val)
    
    updated_grid = jax.vmap(update_grid_val)(grid.ravel(), diff_all).reshape(grid_shape)

    # Step 6: Update start_idx based on the number of new particles added
    new_start_idx = start_idx + total_particles
    print("total particles: ", total_particles)
    print("start_idx: ", start_idx)

    return new_start_idx, updated_grid, new_particle_positions 


def internal_filling(
    grid,
    grid_density,
    grid_dx,
    new_particles,
    start_idx,
    max_particles_per_cell,
    exclude_dir,
    ray_cast_dir,
    threshold,
    key,
):
    grid_shape = grid.shape[0]
    indices = jnp.indices(grid.shape).reshape(3, -1).T  # Flattened grid index
    print("Shape of indices: ", indices.shape)
    # indices = jnp.meshgrid( jnp.arange(grid_shape), jnp.arange(grid_shape), jnp.arange(grid_shape)  , indexing="ij")
    def process_cell(idx):
        i, j, k = idx
        cell_filled = grid[i, j, k]
        index = jnp.array([i, j, k])
        
        # Check collision for non-excluded directions
        hit_check_fn = lambda d: (d != exclude_dir) & collision_search( grid_density, index , d , grid_shape,threshold)
        collision_hit = jax.vmap(hit_check_fn)(jnp.arange(6)).all()

        # Check hit times along ray_cast_dir
        hit_times = collision_times(grid,grid_density,index, ray_cast_dir, grid_shape , threshold)
        
        print("Collision hit: ", collision_hit , "Hit times: ", hit_times , "Cell filled: ", cell_filled , "Index: ", index)
       

        

        return jnp.maximum(0,max_particles_per_cell - cell_filled)*((cell_filled == 0) & collision_hit & (hit_times % 2 == 1))
            

    # Process all cells and flatten results
    diff_all = jax.vmap(process_cell,in_axes=(0))(indices)
    repeated_indices = jnp.repeat(indices, diff_all, axis=0)  # Shape: (total_particles, 3)
    
    # Step 3: Generate random displacements
    total_particles = repeated_indices.shape[0]
    random_disps = jax.random.uniform(key, shape=(total_particles, 3)) * grid_dx  # Shape: (total_particles, 3)

    # Step 4: Compute new particle positions and concatenate with `new_particles`
    new_particle_positions = repeated_indices * grid_dx + random_disps  # Shape: (total_particles, 3)



    # Concatenate with existing new_particles and update start index
    new_particles = jnp.concatenate([new_particles, new_particle_positions], axis=0)
    new_start_idx = new_particles.shape[0]
    
    return new_particles, new_start_idx


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