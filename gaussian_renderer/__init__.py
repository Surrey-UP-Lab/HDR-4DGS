#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
from torch import Tensor

import math
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: Tensor, scaling_modifier: float = 1.0,
           override_color=None, hist_luminance: Tensor = None, train: bool = True, iteration: int = 0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
        if pipe.compute_cov3D_python:
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
        else:
            _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        if pc.gaussian_dim == 3 or pc.force_sh_3d:
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        elif pc.gaussian_dim == 4:
            dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
            sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
        else:
            raise ValueError("Unknown Gaussian dimension")
        # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        colors_precomp = sh2rgb + 0.5
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]

    time = torch.tensor(viewpoint_camera.timestamp, dtype=torch.float32, device="cuda")

    if iteration == 19999:
        print(f'{iteration}')
    
    if hist_luminance is not None:
        with torch.no_grad():
            index = int(time * hist_luminance.shape[0]) - 1 if time == 1 else int(time * hist_luminance.shape[0])

            assert index >= 0, "Index must be >= 0, but got {}".format(index)
            hist_luminance[index] = colors_precomp.mean(dim=0).cpu()  # store hdr colors

            if index >= 20:
                bef_lum = hist_luminance[index-20:index].detach().requires_grad_(False)
            elif index == 0:
                bef_lum = hist_luminance[index:index+1].detach().requires_grad_(False)
            else:
                bef_lum = hist_luminance[0:index].detach().requires_grad_(False)

            bef_lum = bef_lum.cuda()
    else:
        bef_lum = None

    exp_time = torch.tensor(viewpoint_camera.exp_time, dtype=torch.float32, device="cuda")
    colors_hdr = torch.exp(colors_precomp)
    colors_ldr = pc.tone_mapper(colors_precomp, bef_lum, time, exp_time, iteration)
    colors = torch.cat([colors_hdr, colors_ldr], dim=-1)
    
    rendered, radii, depth, alpha, flow, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors,
            flow_2d=flow_2d,
            opacities=opacity,
            ts=ts,
            scales=scales,
            scales_t=scales_t,
            rotations=rotations,
            rotations_r=rotations_r,
            cov3D_precomp=cov3D_precomp)
    
    rendered_hdr_image, rendered_image = rendered.split([3, 3], dim=0)
    
    log_hdr_img = torch.log(rendered_hdr_image + 1e-5).permute(1, 2, 0)  # [400, 400, 3]
    ldr_image = pc.tone_mapper(log_hdr_img, bef_lum, time, exp_time, iteration)
    ldr_image = ldr_image.permute(2, 0, 1)  # [3, 400, 400]
    
    del bef_lum
    
    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "luminance_bank": hist_luminance,
            "render_hdr": rendered_hdr_image,
            "extra_image": ldr_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow}
