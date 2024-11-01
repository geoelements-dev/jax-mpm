import sys
sys.path.append("gaussian-splatting")
import os 
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
import argparse
import torch
from mpm_solver import *
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

   
    if not os.path.exists(args.model_path) :
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)


    device = "cuda:0"


    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
   
    
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    )

    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"].to(device=device)
    init_cov = params["cov3D_precomp"].to(device=device)
    init_screen_points = params["screen_points"].to(device=device)
    init_opacity = params["opacity"].to(device=device)
    init_shs = params["shs"].to(device=device)

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]



    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

  
    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device=device)
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    


    print("Max x pos : ", torch.max(transformed_pos[:, 0]))
    print("Min x pos : ", torch.min(transformed_pos[:, 0]))
    print("Max y pos : ", torch.max(transformed_pos[:, 1]))
    print("Min y pos : ", torch.min(transformed_pos[:, 1]))
    print("Max z pos : ", torch.max(transformed_pos[:, 2]))
    print("Min z pos : ", torch.min(transformed_pos[:, 2]))
    

    gs_num = transformed_pos.shape[0]
    filling_params = preprocessing_params["particle_filling"]
    
    

   

    

    if filling_params is not None:
        print("Filling internal particles...")
        transformed_pos , init_opacity , init_cov = reduce_particles_in_cell(transformed_pos, init_opacity, init_cov, 
                                                                             material_params["grid_lim"] / filling_params["n_grid"], 
                                                                             filling_params["n_grid"], 
                                                                             filling_params["reduce_particles_to"])

        mpm_init_pos = j2t(fill_particles(
            pos=t2j(transformed_pos),
            opacity=t2j(init_opacity),
            cov=t2j(init_cov),
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_particles_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        )).to(device=device)

        
    else:
        mpm_init_pos = transformed_pos.to(device=device)


    data_np = mpm_init_pos.cpu().detach().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the particles
    ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], c=data_np[:, 2], cmap='viridis')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Scatter Plot of Particles")

    plt.show()
        
    # data_np = mpm_init_pos.cpu().detach().numpy()
    # np.savez("particle_positions.npz", positions=data_np)


    # Printing min,max and mean positions 

    print("Min position: ", torch.min(mpm_init_pos, 0))
    print("Max position: ", torch.max(mpm_init_pos, 0))
    print("Mean position: ", torch.mean(mpm_init_pos, 0))


    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        uniform=material_params["material"] == "sand",
    ).to(device=device)



    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity


  



    mpm_solver = load_initial_data_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov, n_grid=150, grid_lim=2,init_cov=True)
    mpm_solver['mpm_model'],mpm_solver['mpm_state'] = set_parameters_dict(mpm_solver['mpm_model'],mpm_solver['mpm_state'], material_params)

    mpm_solver['mpm_state'],mpm_solver['mpm_model'] = finalize_mu_lam(mpm_solver['mpm_state'],mpm_solver['mpm_model']) # set mu and lambda from the E and nu input
    mpm_solver = add_grid_boundaries(mpm_solver)
    mpm_solver = add_surface_collider( mpm_solver , (0, 0.5, 0), (0.0,1.0,0.0), 'slip', 0.1)
    mpm_solver = add_surface_collider( mpm_solver , (0, 1.5, 0), (0.0,-1.0,0.0), 'slip', 0.1)
    mpm_solver = add_surface_collider( mpm_solver , (0.5335, 0, 0), (1.0,0.0,0.0), 'slip', 0.1)
    mpm_solver = add_surface_collider( mpm_solver , (1.4665, 0, 0), (-1.0,0.0,0.0), 'slip', 0.1)
    mpm_solver = add_surface_collider( mpm_solver , (0, 0, 0.9558), (0.0,0.0,1.0), 'slip', 0.1)

    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )


    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            p2g2p(mpm_solver,frame, substep_dt)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        mpm_solver['mpm_state'] = compute_R_from_F(mpm_solver['mpm_state'])

        if args.render_img:
            pos = j2t(mpm_solver['mpm_state']['particle_x'])
            cov3D = j2t(mpm_solver['mpm_state']['particle_cov'])
            rot = j2t(mpm_solver['mpm_state']['particle_R'])
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos.float(),
                means2D=init_screen_points.float(),
                shs=None,
                colors_precomp=colors_precomp.float(),
                opacities=opacity.float(),
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D.float(),
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        # fps = int(1.0 / time_params["frame_dt"])
        fps = 5
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )





    
