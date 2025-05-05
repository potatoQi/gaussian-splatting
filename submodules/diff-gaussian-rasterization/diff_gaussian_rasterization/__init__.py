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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    # 调用 torch.autograd.Function 的子类 _RasterizeGaussians 的 apply 方法来执行渲染操作
    # apply 会依次执行: _RasterizeGaussians.forward 和 _RasterizeGaussians.backward
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

'''
知识点补充:
可以发现, 这个类继承于 torch.autograd.Function, 只要是继承这个类的, 其实就是 python 层面自己写的一个自定义算子
自定义算子的写法跟普通继承 nn.Module 有几个区别:
1. 不需要写 __init__
2. 外部调用时, 只能用 类名.apply(...) 来调用, 不能实例化 (本质就可以把其理解为一个函数过程)
3. 必须要实现 forward 和 backward 方法, 这俩方法参数不是用 self, 而是用 ctx (ctx 是 context 的缩写, 上下文)
4. forward 和 backward 头上必须要有 @staticmethod 装饰器
5. forward 方法的参数是 ctx + apply(...) 里的 ...
6. backward 方法的参数是 ctx + forward return 值的梯度, 相当于 backward 方法就是要实现接收结果对前向 return 那几个值的梯度, 然后用这几个梯度计算对 forward 输入参数的梯度的功能罢了
7. backward 方法的 return 值必须是一个元组, 且元素个数严格等于在 forward 里定义的输入参数的数量, 如果该输入不需要梯度, 就返回 None
'''
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,            # 所有高斯体的 3D 坐标 [N 3]
        means2D,            # [N 3] 的屏幕空间占位张量, 光栅器会往里写入 (x, y) 投影坐标
        sh,                 # sh 系数
        colors_precomp,     # 预先计算好的 RGB 颜色 (若有)
        opacities,          # 所有高斯体的不透明度
        scales,             # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
        rotations,          # 每个高斯体的旋转变量
        cov3Ds_precomp,     # 预先计算好的协方差矩阵 (若有)
        raster_settings,    # 渲染需要的配置参数, 具体内容看 class GaussianRasterizationSettings
    ):

        # Restructure arguments the way that the C++ lib expects them
        # 把所有参数重新打包成一个元组, 方便传递给 C++ 端
        args = (
            raster_settings.bg,                 # 背景颜色
            means3D,                            # 所有高斯体的 3D 坐标 [N 3]
            colors_precomp,                     # 预先计算好的 RGB 颜色 (若有)
            opacities,                          # 所有高斯体的不透明度
            scales,                             # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations,                          # 每个高斯体的旋转变量
            raster_settings.scale_modifier,     # 控制高斯体们的尺寸, 缩放因子
            cov3Ds_precomp,                     # 预先计算好的协方差矩阵 (若有)
            raster_settings.viewmatrix,         # 视图矩阵
            raster_settings.projmatrix,         # 投影矩阵
            raster_settings.tanfovx,            # 单位深度处的半宽度
            raster_settings.tanfovy,            # 单位深度处的半高度
            raster_settings.image_height,       # 输出图像的像素高
            raster_settings.image_width,        # 输出图像的像素宽
            sh,                                 # sh 系数
            raster_settings.sh_degree,          # 球谐函数阶数
            raster_settings.campos,             # 相机在世界里的坐标
            raster_settings.prefiltered,        # 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
            raster_settings.antialiasing,       # 是否开启抗锯齿
            raster_settings.debug               # 如果 True，光栅化器会多打印点位数据、可视化中间结果，方便调试
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)
        # num_rendered: 记录每个像素实际上被多少个高斯点贡献了颜色
        # NOTE: ? ↑
        # color: 三通道 RGB 图像
        # radii: 高斯体投影到屏幕上的像素半径
        # geomBuffer: 存储每个像素对应哪些高斯点索引
        # NOTE: ?↑
        # binningBuffer: 加速查找哪些点落在哪个屏幕区域
        # NOTE: ?↑
        # imgBuffer: 临时存放中间的像素数据，方便分批处理。这些 Buffer 在前向存好，为后向梯度计算提供随机访问。
        # NOTE: ?↑
        # invdepths: 反深度图

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        # 把那些需要存储梯度反向更新的变量都放到 ctx.save_for_backward 里
        ctx.save_for_backward(
            colors_precomp,         # 预先计算好的 RGB 颜色 (若有)
            means3D,                # 所有高斯体的 3D 坐标 [N 3]
            scales,                 # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations,              # 每个高斯体的旋转变量
            cov3Ds_precomp,         # 预先计算好的协方差矩阵 (若有)
            radii,                  # 高斯体投影到屏幕上的像素半径
            sh,                     # sh 系数
            opacities,              # 所有高斯体的不透明度
            geomBuffer,             # 存储每个像素对应哪些高斯点索引
            # NOTE: ?↑
            binningBuffer,          # 加速查找哪些点落在哪个屏幕区域
            # NOTE: ?↑
            imgBuffer               # 临时存放中间的像素数据，方便分批处理。这些 Buffer 在前向存好，为后向梯度计算提供随机访问。
            # NOTE: ?↑
        )
        
        return color, radii, invdepths

    @staticmethod
    def backward(
        ctx,
        grad_out_color,             # 对渲染图 color 的梯度
        _,                          # 对 radii 的梯度, 这里不需要, 所以用 _ 占位
        grad_out_depth              # 对反深度图 invdepths 的梯度
    ):
        # 恢复对应 forward 里通过 ctx.save_for_backward 保存的变量
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        # 同样的, 把需要的参数重新打包成一个元组顺序, 方便传递给 C++ 端
        args = (
            raster_settings.bg,                 # 背景颜色
            means3D,                            # 所有高斯体的 3D 坐标 [N 3]
            radii,                              # 高斯体投影到屏幕上的像素半径
            colors_precomp,                     # 预先计算好的 RGB 颜色 (若有)
            opacities,                          # 所有高斯体的不透明度
            scales,                             # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations,                          # 每个高斯体的旋转变量
            raster_settings.scale_modifier,     # 控制高斯体们的尺寸, 缩放因子
            cov3Ds_precomp,                     # 预先计算好的协方差矩阵 (若有)
            raster_settings.viewmatrix,         # 视图矩阵
            raster_settings.projmatrix,         # 投影矩阵
            raster_settings.tanfovx,            # 单位深度处的半宽度
            raster_settings.tanfovy,            # 单位深度处的半高度
            grad_out_color,                     # 对渲染图 color 的梯度
            grad_out_depth,                     # 对反深度图 invdepths 的梯度
            sh,                                 # sh 系数
            raster_settings.sh_degree,          # 球谐函数阶数
            raster_settings.campos,             # 相机在世界里的坐标
            geomBuffer,                         # 存储每个像素对应哪些高斯点索引
            # NOTE: ?↑
            num_rendered,                       # 记录每个像素实际上被多少个高斯点贡献了颜色
            # NOTE: ?↑
            binningBuffer,                      # 加速查找哪些点落在哪个屏幕区域
            # NOTE: ?↑
            imgBuffer,                          # 临时存放中间的像素数据，方便分批处理。这些 Buffer 在前向存好，为后向梯度计算提供随机访问。
            # NOTE: ?↑
            raster_settings.antialiasing,       # 是否开启抗锯齿
            raster_settings.debug               # 如果 True，光栅化器会多打印点位数据、可视化中间结果，方便调试
        )

        # Compute gradients for relevant tensors by invoking backward method
        # 拿到 8 个梯度, 这 8 个梯度就分别对应着 forward 输入的前 8 个参数
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,           # 对所有高斯体的 3D 坐标 [N 3] 的梯度
            grad_means2D,           # 对 [N 3] 的屏幕空间占位张量的梯度
            grad_sh,                # 对 sh 系数的梯度
            grad_colors_precomp,    # 对预先计算好的 RGB 颜色 (若有) 的梯度
            grad_opacities,         # 对所有高斯体的不透明度的梯度
            grad_scales,            # 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度
            grad_rotations,         # 对每个高斯体的旋转变量的梯度
            grad_cov3Ds_precomp,    # 对预先计算好的协方差矩阵 (若有) 的梯度
            None,                   # 对 raster_settings 的梯度, 这里不需要, 所以用 None 占位
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int               # 输出图像的像素高
    image_width: int                # 输出图像的像素宽
    tanfovx : float                 # 单位深度处的半宽度
    tanfovy : float                 # 单位深度处的半高度
    bg : torch.Tensor               # 背景颜色
    scale_modifier : float          # 控制高斯体们的尺寸, 缩放因子
    viewmatrix : torch.Tensor       # 视图矩阵
    projmatrix : torch.Tensor       # 投影矩阵
    sh_degree : int                 # 球谐函数阶数
    campos : torch.Tensor           # 相机在世界里的坐标
    prefiltered : bool              # 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
    debug : bool                    # 如果 True，光栅化器会多打印点位数据、可视化中间结果，方便调试
    antialiasing : bool             # 是否开启抗锯齿

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings  # 拿到渲染需要的配置参数, 具体内容看上面的 class GaussianRasterizationSettings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(
        self,
        means3D,                    # 所有高斯体的 3D 坐标 [N 3]
        means2D,                    # [N 3] 的屏幕空间占位张量, 光栅器会往里写入 (x, y) 投影坐标
        opacities,                  # 所有高斯体的不透明度
        shs = None,                 # sh 系数
        colors_precomp = None,      # 预先计算好的 RGB 颜色 (若有)
        scales = None,              # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
        rotations = None,           # 每个高斯体的旋转变量
        cov3D_precomp = None        # 预先计算好的协方差矩阵 (若有)
    ):
        raster_settings = self.raster_settings

        # 要不就给 shs, 要不就给 colors_precomp, 不能都给, 也不能都不给, 否则报错
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        # 要不就给 scales, rotations, 要不就给 cov3D_precomp, 不能都给, 也不能都不给, 否则报错
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 保证上述变量所有为 None 的, 替换为空张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 准备进入 cppcuda 咯~
        return rasterize_gaussians(
            means3D,            # 所有高斯体的 3D 坐标 [N 3]
            means2D,            # [N 3] 的屏幕空间占位张量, 光栅器会往里写入 (x, y) 投影坐标
            shs,                # sh 系数
            colors_precomp,     # 预先计算好的 RGB 颜色 (若有)
            opacities,          # 所有高斯体的不透明度
            scales,             # 每个高斯体的尺度 (在 xyz 轴的缩放长度)
            rotations,          # 每个高斯体的旋转变量
            cov3D_precomp,      # 预先计算好的协方差矩阵 (若有)
            raster_settings,    # 渲染需要的配置参数, 具体内容看上面的 class GaussianRasterizationSettings
        )

