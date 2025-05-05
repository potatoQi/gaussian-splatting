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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 模型预测的 scale 是实数尺度的, 所以模型预测的 scale 需要 activation 一下 (即 exp 运算一下) 才能得到真实的 scale
        # 所以就定义了 scaling_activation 和 scaling_inverse_activation 函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation     # NOTE: ?

        # 模型预测的 opacity 是实数尺度的, 所以模型预测的 opacity 需要 activation 一下 (即 sigmoid 运算一下) 才能得到真实的 opacity
        # 所以就定义了 opacity_activation 和 opacity_inverse_activation 函数
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # 模型预测的旋转向量是任意 4 维可正可负的向量, 所以需要 activation 一下 (即 normalize 运算一下) 才能得到真实的旋转向量
        # 真实的旋转向量是一个单位向量
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0                   # 当前球谐函数的阶数
        self.optimizer_type = optimizer_type        # 优化器类型 (default or sparse_adam)
        self.max_sh_degree = sh_degree              # 最大球谐阶数, 就等于用户传进来的值

        self._xyz = torch.empty(0)                  # 存储所有高斯点的坐标
        self._scaling = torch.empty(0)              # 存储每个点尺度的 logits 值 (在 xyz 轴的缩放长度)
        self._opacity = torch.empty(0)              # 存储每个点不透明度的 logits 值
        self._rotation = torch.empty(0)             # 存储每个点的 logits 旋转变量 (每个旋转变量是 4 维) (协方差矩阵, 控制旋转方向)
        # 这里补个知识点
        # 描述一个高斯点的属性: 坐标, 协方差, 不透明度, 颜色
        # 其中颜色由球谐函数控制, 说简单点, 从物理直觉上理解
        # _features_dc 翻译叫直流分量, _features_rest 翻译叫高频分量, 太抽象了
        # 其实理解为基础底色和高频细节就好了
        # 举个例子, 想象场景里每个高斯点就像一颗微小的灯泡, 它不仅有一个基础的颜色亮度(DC分量), 还能根据观察方向或光照方向出现明暗变化(高频分量)
        # 至于它们的 shape, dc 分量是 [N 3 1], 高频分量是 [N 3 (l+1)^2 - 1] (这个 -1 就是减掉的 dc 分量里的 1)
        self._features_dc = torch.empty(0)          # dc 分量
        self._features_rest = torch.empty(0)        # 高频分量

        self.max_radii2D = torch.empty(0)           # 维护每个高斯点在所有视角中投影到像素平面后的最大半径
        self.xyz_gradient_accum = torch.empty(0)    # 维护每个高斯点的梯度累积的数值
        self.denom = torch.empty(0)                 # 维护每个高斯点的梯度累积的次数
        self.optimizer = None                       # 优化器
        self.percent_dense = 0                      # 启动稠密化的场景覆盖比例阈值
        self.spatial_lr_scale = 0                   # 训练集的空间尺度半径 (场景范围的大小)
        self.setup_functions()      # 定义一些函数

    # 把一些优化器相关参数定义好
    def training_setup(self, training_args):
        # percent_dense 的意思是如果某个点覆盖了场景范围的百分比小于此值（尺度足够小）, 它才有稠密化的其中一项资格 (默认 0.01)
        self.percent_dense = training_args.percent_dense

        # self.get_xyz 是一个 (N, 3) 的 tensor, 代表了所有高斯点的坐标
        # 所以这句话就是给每个高斯点开一个 "梯度累计变量", 初始值全为 0
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # [n 1]

        # 同理, 给每个点开一个梯度次数累计变量, 初始值全为 0
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # [n 1]

        # 这里定义这个 l 有啥用? 为什么要这么定义呢?
        # 因为 l 等下要传给 torch.optim.Adam(l) 里面, 还记得以前还想训额外可学习参数时候我的做法吗: torch.optim.Adam([self.tmp, model.parameters()], ...)
        # 一样的道理, 这里的 l 就是自己配置了下, 首先 l 是个 list, 这没问题
        # 然后里面每一个是一个 dict, params 就是优化器要优化的参数, lr 就是优化这个参数的学习率, name 就是给这个参数起一个名字
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 定义优化器 (虽然 lr 都设置的 0, 但是会优先用 l 中的配置)
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 定义关于 xyz 的学习率调度函数
        # Q: 在 self.optimizer 里不是已经添加了 _xyz 的优化参数和学习率了吗? 为啥又要搞一个学习率调度函数?
        # A: 添加是肯定要添加的, 不然优化器就不会优化 _xyz 了, 但是默认的学习率调度规则我不喜欢啊, 我想自定义, 所以我就定义个函数
        # 到时候直接通过调度函数得到学习率, 然后动态的修改 optimizer 里关于 _xyz 的学习率就好了
        # BUG: 我觉得这里 lr_delay_mult 传了跟没传一样, 因为没传 lr_delay_steps
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
        # 定义一个关于曝光度的优化器
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        # 定义关于曝光度的学习率调度函数, 这些参数默认都是 0, 所以相当于默认是不开启曝光补偿的
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations
        )

    # 这个函数相当于执行了 __init__ + training_setup, 把该设定的东西都设定好了
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # 保存模型参数 (return 的东西就是 train.py 里 ckpt load 出来的 model_params)
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    # 根据当前迭代步数, 更新 _xyz 和 _exposure 的学习率
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:    # 遍历 exposure_optimizer 的参数组 (只添加了一组: _exposure)
                param_group['lr'] = self.exposure_scheduler_args(iteration) # 把学习率按照调度函数更新

        for param_group in self.optimizer.param_groups: # 遍历 optimizer 参数组
            if param_group["name"] == "xyz":    # 找到 _xyz 参数组
                lr = self.xyz_scheduler_args(iteration) # 把学习率按照调度函数更新
                param_group['lr'] = lr
                return lr

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 统计下高斯点们的梯度信息 (梯度累计和, 梯度累计次数)
    def add_densification_stats(
        self,
        viewspace_point_tensor,     # 视角下高斯点的坐标
        update_filter               # 在当前视角下可见高斯点的索引
    ):
        # 首先 torch.norm(input, p=2, dim, keepdim) 是计算 input 在指定维度的 L2 范数, keepdim 表示是否保留被归约的维度, 将其保留为 1
        # 然后这里的 viewspace_point_tensor.grad shape 是 [N 3], 代表了每个高斯点在视角下 xyz 的梯度
        # 所以 viewspace_point_tensor.grad[update_filter,:2] 就是取出可见的高斯点的 x 和 y 的梯度, shape 为 [n 2]
        # 为什么不取 z 的梯度? 因为图片上算 loss 只关注 2D 图片
        # 所以 torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) 其实就是得到一个 shape 为 [n 1] 的 tensor,
        # 每个元素是可见高斯点的: \sqrt{\frac{\partial L}{\partial x}^2 + \frac{\partial L}{\partial y}^2}
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1  # 梯度累计次数+1, 因为 train.py 里进行了一次迭代

    def densify_and_prune(
        self,
        max_grad,               # 梯度阈值, 只有在这次反向传播中梯度贡献超过这个阈值的, 才会在它们周围长出新点
        min_opacity,            # 不透明度下限阈值
        extent,                 # 场景空间的尺度长度
        max_screen_size,        # 剪枝半径阈值, 任何投影半径大于这个值的高斯点, 都会被剪掉
        radii                   # 每个高斯点投影到图像上时的像素半径
    ):
        grads = self.xyz_gradient_accum / self.denom    # [N 1]
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        # 先 clone 一批高梯度, 小尺寸的点
        self.densify_and_clone(grads, max_grad, extent)
        # 再 split 一批高梯度, 大尺寸的点
        self.densify_and_split(grads, max_grad, extent)

        # BUG: 执行到这里的时候 .max_radii2D 已经作废了吧, 全是 0, 那么后边根据 max_radii2D 进行剪枝的意义是什么呢?

        # 这里的 prune_mask 是一个 bool 数组, 代表了每个高斯点是否需要被剪枝
        # 把那些不透明度小于阈值的剪掉
        prune_mask = (self.get_opacity < min_opacity).squeeze() # [N]

        # 如果指定了 max_screen_size, 还要把 “那些历史上在任一视角下投影到屏幕的最大半径 > max_screen_size 的点” 和 "超过边界" 的点也剪掉
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # self.get_scaling.max(dim=1).values 的 shape 是 [N], 表示每个点在 xyz 轴上最大的缩放系数
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 剪枝
        self.prune_points(prune_mask)

        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def densify_and_clone(
        self,
        grads,                  # 梯度 (x,y 综合后的梯度) [N 1]
        grad_threshold,         # 梯度阈值 (高于这个的且尺度比场景空间尺度小很多的都进行 clone)
        scene_extent            # 场景空间的尺度长度 (自身 xyz 缩放尺度比这尺度小很多的且梯度高于 grad_threshold 的都进行 clone)
    ):
        # 把那些梯度大于 grad_threshold 的点的 mask 设为 True
        # torch.norm(grads, dim=-1) 在这里其实只是起到 [N 1] -> [N] 的作用
        # torch.where 我是觉得没必要写
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        # 上面的 mask 已经把梯度大于阈值的筛出来了, 但是还要叠加一层尺度够小的条件
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        # 取出满足条件的点的坐标, 缩放长度, 旋转, 不透明度, 颜色, 投影半径
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # 将这些点 clone 一份
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densification_postfix(
        self,
        new_xyz,                    # 要 clone 的点的坐标
        new_features_dc,            # 要 clone 的点的 dc 分量
        new_features_rest,          # 要 clone 的点的 高频分量
        new_opacities,              # 要 clone 的点的不透明度
        new_scaling,                # 要 clone 的点的 xyz 轴缩放长度
        new_rotation,               # 要 clone 的点的旋转向量
        new_tmp_radii               # 要 clone 的点的投影到像素平面的半径
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }

        # cat_tensors_to_optimizer 这个方法会对字典里每一组参数, 和模型里原来的同名参数在第零维度做 concat (把新点拼到旧点后面)
        # 同时在 optimizer 里, 为这些新增的参数位置分配一套新的动量／二阶动量缓存 (exp_avg、exp_avg_sq)
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 更新这些参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新与“点数量”相关的统计张量
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii)) # clone 后全部点的投影半径这玩意得马上更新, 因为后边剪枝需要用到这玩意
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # clone 完后, 梯度全部清零
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # clone 完后, 梯度全部清零
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # clone 完后, max_radii2D 也全部清零

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_and_split(
        self,
        grads,                  # 梯度 (x,y 综合后的梯度) [N 1]
        grad_threshold,         # 梯度阈值 (高于这个的且尺度比场景空间尺度大很多的都进行 split)
        scene_extent,           # 场景空间的尺度长度 (自身 xyz 缩放尺度比这尺度大很多的且梯度高于 grad_threshold 的都进行 split)
        N=2                     # 每个点分裂的数量 (默认 2, 也就是分裂成 2 个点)
    ):
        # 这里的 n_init_points 代表最新的点数量 (因为前面可能因为 clone 导致点数量变多了), 此时 grads 的 shape 仍然是没 clone 之前的
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()  # 把没 clone 之前的点的 grads 放入对应位置, 新 clone 的点 grads 是 0

        # 把那些梯度大于阈值同时尺度大于阈值的点的 mask 设为 True
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        # 在选出符合条件的点的基础上, 再 copy N 份
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)  # [M, 3] -> [M*N , 3]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)    # 采样出微小偏移, [M*N, 3]
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 通过 build_rotation 得到世界坐标系下的旋转矩阵, [M, 3, 3] -> [M*N, 3, 3]

        # 把这 M*N 个点的坐标, 协方差, 不透明度, 颜色, 投影半径都选出来
        # 这个 M*N 个点的坐标叠加了偏差, 尺度是在父尺度上缩小了
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) # torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) 得到世界坐标系下的偏移向量
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 将这些点 clone 一份
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 把那些已经老的, 之前用来复制新点的老点给剪掉
        # prune_filter 就是最新版 (因为刚刚才 clone 过 M*N 个点) 的 mask
        prune_filter = torch.cat((
            selected_pts_mask,  # 首先是未 clone 之前的 mask 数组
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool) # 然后是刚刚 clone 的 M*N 个点的 mask (都是 False)
        ))
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # _prune_optimizer 这个方法会对 optimizer 里把每个参数张量（位置、尺度、旋转、特征、不透明度等）按照 valid_points_mask 做索引，只留被保留的行
        # 同时把对应的 Adam 动量 (exp_avg) 和二阶动量 (exp_avg_sq) 也筛成相同长度，确保优化状态对齐。
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 这几个玩意估计都是 0, 因为前面刚做过 densification_postfix, 除了 tmp_radii 以外的都已经清空了
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置所有高斯点的不透明度
    def reset_opacity(self):
        # 得到一个 shape 跟 _opacity 一样的 tensor, 里面的值全都 <= 0.01
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 找到 optimzer 里 name 为 'opacity' 的参数组, 然后把里面的值换为 opacities_new, 梯度信息清空
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # 在 optimizer 优化器里找到 name 为 name 的参数组, 把参数值里的值换为 tensor, 并把梯度信息清空, 最后返回该参数组名字和对应值的一个 dict
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 通过 optimizer.state.get 拿到的是 group["params"][0] 的 state, 也就是 group["params"][0] 的梯度信息
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)          # 清空
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)       # 清空

                # 把原先 group["params"][0] 的梯度信息删掉
                del self.optimizer.state[group['params'][0]]
                # group["params"][0] 换为新的值 (tensor)
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                # 把重置后的 group["params"][0] 的梯度信息重新赋值
                self.optimizer.state[group['params'][0]] = stored_state
                # 这里就是开个 dict 记录下这个参数组的名字和对应的值
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    









    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(
        self,
        pcd : BasicPointCloud,      # 点云对象
        # BUG: 下面这个 int 是啥意思? 不是应该是 list 吗? 我觉得作者写错了
        cam_infos : int,            # 训练集相机对象列表
        spatial_lr_scale : float    # 训练集的空间尺度半径 (场景范围的大小)
    ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
