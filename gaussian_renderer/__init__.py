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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(
    viewpoint_camera,               # 相机对象
    pc : GaussianModel,             # GaussianModel对象
    pipe,                           # 流水线参数对象
    bg_color : torch.Tensor,        # 渲染背景
    scaling_modifier = 1.0,         # 控制高斯体们的尺寸, 缩放因子
    separate_sh = False,            # 渲染时是否使用稀疏 Adam 加速器, 还是说用默认的 SH 算子
    override_color = None,          # 渲染覆盖颜色 (若不指定则会用 sh 系数计算颜色)
    use_trained_exp=False           # 渲染时是否启动曝光补偿
):
    '''
    看了这个函数我目前有一些想法:
    首先通过这个函数, 可以得到：
        1. 渲染的 2D 图像
        2. 每个高斯点的屏幕坐标 (含梯度信息)
        3. 每个高斯点投影到图像上时的像素半径
        4. 渲染的反深度图
    
    那么有了渲染的 2D 图像和渲染的反深度图, 就可以在 train.py 里跟 gt 计算 loss 了。然后即可反向传播影响到这里的 pc, 也就是 GaussianModel 里的参数
    那 "每个高斯点的屏幕坐标 (含梯度信息)" 和 "每个高斯点投影到图像上时的像素半径" 这俩玩意有什么用呢?
    其实就是给 train.py 里稠密化 & 剪枝提供信息用的
    '''

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # 这个 screenspace_points 我的理解就是给每个高斯体在屏幕上分配一个可微分的占位张量
    # 到时候后面渲染写入时, pytorch 能跟踪并保存梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0    # [N 3]
    try:
        # 这里是确保 screenspace_points 一定是有梯度可微分的 (不同版本 pytorch 可能不需要这句话)
        screenspace_points.retain_grad()
    except:
        pass

    # viewpoint_camera.FoVx 是水平视场角（以弧度为单位），表示相机能看到左右两侧的总角度; viewpoint_camera.FoVy 同理
    # 所以 tanfovx 是在单位深度 (相机前方 1 个单位距离) 处得到的半宽度
    # 所以 tanfovy 是在单位深度 (相机前方 1 个单位距离) 处得到的半高度
    # 画个图就清楚了, 很简单
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 这里是把渲染需要知道的东西打包到一个配置对象里
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),        # 输出图像的像素高
        image_width=int(viewpoint_camera.image_width),          # 输出图像的像素宽
        tanfovx=tanfovx,                                        # 单位深度处的半宽度
        tanfovy=tanfovy,                                        # 单位深度处的半高度
        bg=bg_color,                                            # 背景颜色
        scale_modifier=scaling_modifier,                        # 控制高斯体们的尺寸, 缩放因子
        # 知识点补充:
        # 先用视图矩阵 viewmatrix 把世界坐标系下的点弄到相机坐标系
        # 再用投影矩阵 projmatrix 把相机坐标系下的点投影到屏幕坐标系
        viewmatrix=viewpoint_camera.world_view_transform,       # 视图矩阵
        projmatrix=viewpoint_camera.full_proj_transform,        # 投影矩阵
        sh_degree=pc.active_sh_degree,                          # 球谐函数阶数
        campos=viewpoint_camera.camera_center,                  # 相机在世界里的坐标
        prefiltered=False,              # 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
        debug=pipe.debug,               # 如果 True，光栅化器会多打印点位数据、可视化中间结果，方便调试
        antialiasing=pipe.antialiasing  # 是否开启抗锯齿
    )

    # 创建光栅化器实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz            # 取出所有高斯体的 3D 坐标 [N 3]
    means2D = screenspace_points    # 把之前创建的, 可微的占位张量赋给 means2D
    opacity = pc.get_opacity        # 取出所有高斯体的不透明度

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 下面三个都是占位, 也就是要么直接算 3x3 的 cov3D_precomp, 那么直接读出 scales, rotations, 协方差在后续动态算
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 如果提供了预先计算的颜色，则使用它们。否则，如果需要在 Python 中从 SHs 预计算颜色，那么就这样做。如果不是，那么 SH -> RGB 转换将由光栅化器完成。
    shs = None
    colors_precomp = None

    # 如果用户没指定颜色
    if override_color is None:
        # 如果需要在 Python 中从 SHs 预计算颜色
        if pipe.convert_SHs_python:
            # pc.get_features 的形状是 [N D 3], D 表示 sh 系数个数, 3 表示 rgb 三通道
            # shs_view 就是 shape 变为了 [N 3 D]
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)

            # 计算每个高斯中心指向相机的单位方向向量
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # [N 3]
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # 单位化, [N 3]
            
            # eval_sh 用球谐函数在该方向上评估颜色, 得到 [N 3] 的 RGB, 即每个高斯点的颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # +0.5 偏移并 clamp 确保非负
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        # 如果不在 Python 端转换, 则直接把 SH 系数给 shs, 或者把 dc,rest 分量给 dc, shs. 有了这些变量之后再转换
        else:
            # 如果启动了稀疏 Adam 加速器, 就 dc, rest 分量分开传
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    # 如果用户指定了颜色
    else:
        colors_precomp = override_color

    # 开始用光栅化器实例渲染
    # 拿到三个东西: 渲染图像, 半径, 反深度图像
    if separate_sh:
        # BUG: separate_sh 我感觉是不支持的, 因为这里传了 dc 参数进去, 但是 GaussianRasterizer 的 forward 函数里并没有用到 dc 这个参数
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,                          # 所有高斯体的 3D 坐标 [N 3]
            means2D = means2D,                          # [N 3] 的屏幕空间占位张量, 光栅器会往里写入 (x, y) 投影坐标
            dc = dc,                                    # 低频 sh 系数
            shs = shs,                                  # 高频 sh 系数
            colors_precomp = colors_precomp,            # 预先计算好的 RGB 颜色 (若有)
            opacities = opacity,                        # 所有高斯体的不透明度
            scales = scales,                            # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations = rotations,                      # 每个高斯体的旋转变量
            cov3D_precomp = cov3D_precomp               # 预先计算好的协方差矩阵 (若有)
        )
    else:
        (
            rendered_image,
            radii,
            depth_image,
            gauss_sum,
            gauss_count,
        ) = rasterizer(
            means3D = means3D,                          # 所有高斯体的 3D 坐标 [N 3]
            means2D = means2D,                          # [N 3] 的屏幕空间占位张量, 光栅器会往里写入 (x, y) 投影坐标
            shs = shs,                                  # sh 系数
            colors_precomp = colors_precomp,            # 预先计算好的 RGB 颜色 (若有)
            opacities = opacity,                        # 所有高斯体的不透明度
            scales = scales,                            # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations = rotations,                      # 每个高斯体的旋转变量
            cov3D_precomp = cov3D_precomp               # 预先计算好的协方差矩阵 (若有)
        )
        
    # Apply exposure to rendered image (training only)
    # 是否对整张 RGB 图像做一次曝光补偿
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # 返回成图之前, 把数值范围 clamp 一下, 保证正确
    rendered_image = rendered_image.clamp(0, 1)

    assert radii.shape == gauss_sum.shape and radii.shape == gauss_count.shape, "gauss_sum and gauss_count should have the same shape as radii"
    out = {
        "render": rendered_image,                       # 渲染的 2D 图像
        "viewspace_points": screenspace_points,         # screenspace_points 这个张量里面此时已经写入了每个高斯的屏幕坐标 (内含梯度信息)
        "visibility_filter" : (radii > 0).nonzero(),    # 从所有高斯体中筛出“在屏幕上真实可见”的那些，返回它们的索引
        "radii": radii,                                 # 每个高斯点投影到图像上时的像素半径
        "depth" : depth_image,                           # 渲染的反深度图
        "gauss_sum": gauss_sum,
        "gauss_count": gauss_count,
    }
    return out
