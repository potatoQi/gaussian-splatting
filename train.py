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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(
    dataset,                        # 数据加载参数对象
    opt,                            # 优化器参数对象
    pipe,                           # 流水线参数对象
    testing_iterations,         # test 的时间点
    saving_iterations,          # save model 的时间点
    checkpoint_iterations,      # save ckpt 的时间点
    checkpoint,                 # 断点续训的 ckpt 路径
    debug_from,                 # debug 开关开启的时间点
):

    # 检查开启 sparse_adam 的条件
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 初始的迭代步
    first_iter = 0
    # 设置一下输出目录, 保存当前实验配置文件, 返回一个 tensorboard 对象
    tb_writer = prepare_output_and_logger(dataset)


    # 实例化一个可学习的三维高斯点云模型实例
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # 封装了数据集（图像、相机参数、可选深度/掩码）和高斯模型，提供训练和测试时按视角渲染的方法
    scene = Scene(dataset, gaussians)
    # 把 opt 优化器参数在 gaussians 中设置好
    gaussians.training_setup(opt)


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        # 用 .restore() 方法恢复 ckpt 中高斯模型的参数
        gaussians.restore(model_params, opt)

    # 背景颜色用白底还是黑底
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # shape: [3]

    # 创建两个 CUDA 事件, 用来后面打点从而算出迭代的时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 是否启动 稀疏 Adam 加速
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    # 得到一个函数 depth_l1_weight(step), 会在第 step=0 返回 init, step=max_steps 返回 final, 中间以指数形式衰减
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,   # 默认是 1.0
        opt.depth_l1_weight_final,  # 默认是 0.01
        max_steps=opt.iterations
    )

    # 优化高斯模型的过程中, 需要反复从不同角度把模型渲染出来, 跟 gt 照片做对比
    # 所以 sceme.getTrainCameras() 这个方法会返回一个训练用到的视角列表
    # 每一个视角对象里, 包含了相机的内参、外参、图像、深度图、掩码等信息
    # 深度图就是每个像素到相机的距离, 如果这个距离用欧式距离表达就是线性深度, 如果用欧式距离的倒数表达就是反深度图。深度图本质就是 shape 为 (1 h w) 的图
    # 为什么需要深度图? 因为模型不仅要学会渲染出看起来颜色和 gt 一样的画面, 还需要恢复正确的三维结构
    # 这里的 viewpoint_stack 就是一个视角对象列表, 里面每个元素都是一个 Viewpoint 对象
    viewpoint_stack = scene.getTrainCameras().copy()
    # 视角对象列表的下标索引列表
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # 用于平滑记录主光度损失
    ema_loss_for_log = 0.0
    # 用于平滑记录深度正则项的损失
    ema_Ll1depth_for_log = 0.0

    # 创建一个进度条对象
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # 如果 GUI 对象还没没浏览器建立连接, 就尝试初始化连接
        if network_gui.conn == None:
            network_gui.try_connect()   # 我看了里面的实现, 本质就是 listener.accept(), 它写了个异常捕获, 如果失败就 pass, 不会报错
        # 如果本次迭代 GUI 对象已经成功建立连接, 就开始与前端交互
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                # custom_cam: 前端发来的自定义相机视角, 表示用户想看这个视角的渲染结果
                # do_training: 告诉后端渲染完这帧之后, 是否退出与前端交互继续进行训练
                # pipe.convert_SHs_python: 是否使用 python 代码来计算 SH, 还是用 C++ 的实现 (允许从前端临时改用后端的算子实现方式)
                # pipe.compute_cov3D_python: 是否使用 python 代码来计算 cov3D, 还是用 C++ 的实现 (允许从前端临时改用后端的算子实现方式)
                # keep_alive: 是否保持连接, 还是断开连接
                # scaling_modifer: 前端传来的缩放比例, 允许用户在前端调整渲染的缩放比例
                
                if custom_cam != None:
                    net_image = render(
                        custom_cam,                                 # 视角
                        gaussians,                                  # 高斯模型
                        pipe,                                       # 流水线参数对象
                        background,                                 # 渲染背景 shape: [3]
                        scaling_modifier=scaling_modifer,           # 渲染缩放比例
                        use_trained_exp=dataset.train_test_exp,     # 渲染时是否启动曝光补偿 (知识点补充: 曝光补偿是因为可能拍出的照片亮度不统一, 会影响到训练. 所以用一个全局亮度缩放因子，来校正训练中渲染值和真实图像在明暗上的一致性. 让模型专注于捕捉颜色、结构和几何。)
                        separate_sh=SPARSE_ADAM_AVAILABLE           # 当使用稀疏 Adam 加速时, 会给每个高斯点维护单独的球谐通道, 否则用统一的 SH 算子
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)   # -> [h w c]
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                # 发送渲染结果到前端
                network_gui.send(net_image_bytes, dataset.source_path)
                # 是否退出与前端的交互
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            # 如果发生了任何异常, 就把连接断了, 跳出 while
            except Exception as e:
                network_gui.conn = None

        # 打个点
        iter_start.record()

        # 根据当前迭代步, 更新 _xyz 和 _exposure 的学习率
        gaussians.update_learning_rate(iteration)

        # 每 1000 步把 SH 阶数提高一级
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机无放回地从 viewpoint_stack 里抽取一个训练视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # 更新调试模式开启状态
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 定义渲染背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 渲染!
        render_pkg = render(
            viewpoint_cam,      # 视角
            gaussians,          # 模型
            pipe,               # 流水线参数对象
            bg,                 # 渲染背景 shape: [3]
            use_trained_exp=dataset.train_test_exp,     # 渲染时是否启动曝光补偿
            separate_sh=SPARSE_ADAM_AVAILABLE           # 渲染时是否使用稀疏 Adam 加速器, 还是说用默认的 SH 算子
        )
        
        # 渲染出来的 rgb 图
        image = render_pkg["render"]  # tensor, [c, h, w], 分布范围: [0, 1], float32
        # 所有高斯点在视角下的坐标
        viewspace_point_tensor = render_pkg["viewspace_points"]  # tensor, [N, 3]
        # 一个 bool, 表示每个高斯点在当前视角下是否可见
        visibility_filter = render_pkg["visibility_filter"]  # tensor, [N], bool
        # 每个高斯点投影到图像上时的像素半径
        radii = render_pkg["radii"]  # tensor, [N]
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 如果有透明度掩码, 就把渲染出来的图像乘以透明度掩码
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # 如果导入了 ssim 计算的库 (用 cuda 计算 ssim 的库), 就用 cuda 算一下 ssim
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        # 否则用 python 实现算一下 ssim
        else:
            ssim_value = ssim(image, gt_image)
        # loss = (1 - λ)L1 + λ(1 - SSIM)     (SSIM 是越高越好, 所以要 1 - SSIM)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 深度损失
        Ll1depth_pure = 0.0
        # 如果当前视角的深度图可信, 且当前迭代步的深度损失权重大于 0, 就计算深度损失
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]  # 拿到渲染的反深度图
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()    # 拿到 gt 反深度图
            depth_mask = viewpoint_cam.depth_mask.cuda()    # 拿到深度图掩码

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure    # 深度损失要乘上当前迭代步的深度损失权重 (越要后期这个权重越小)
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        # 打个点
        iter_end.record()


        # 下面开始进行 日志、模型保存、稠密化、优化器步进等辅助工作
        with torch.no_grad():
            # 更新下 ema 的 loss 们
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})  # 每 10  步刷新一次进度条尾注
                progress_bar.update(10)  # 推进进度条 10 步
            if iteration == opt.iterations:
                progress_bar.close()

            # 打下日志
            training_report(
                tb_writer,                              # tensorboard 对象
                iteration,                              # 当前迭代步
                Ll1,                                    # 光度损失
                loss,                                   # 总损失
                l1_loss,                                # 计算 L1 损失的函数
                iter_start.elapsed_time(iter_end),      # 每次迭代耗时

                testing_iterations,                     # test 的时间点, 如果到了该 test 的时间点那么会在打完报告后 test 一下
                scene,                                  # 场景对象
                render,                                 # 渲染函数
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),    # 渲染参数
                # 这里传的是是否开启曝光补偿的开关, 实际上这个开关也用来控制是否启用 Synthetic‐NeRF 训练评估模式
                # Synthetic‐NeRF 训练评估模式 就是对于每一张图, 左边用来训练, 右边用来评估
                # 在这套代码里, 启用曝光补偿 = 启用 Synthetic‐NeRF 训练评估模式, 不启用 Synthetic‐NeRF 训练评估模式 = 不启用曝光补偿
                dataset.train_test_exp
            )

            # 保存下 Gaussian model
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 稠密化 & 剪枝 (用于在训练早期自动往场景里长出更多高斯点以填补稀疏区域, 同时必要时减掉过渡聚集的点)
            if iteration < opt.densify_until_iter:
                # 补充个知识点, 如果 a, b 是一维 tensor, 正常情况下, c = a[b] 也是一个一维 tensor
                # c 这个tensor里面下标为 i 的元素值是 a tensor 中下标为 b[i] 的值 (花式索引)
                # 但是如果 b 的类型是 bool, 那么把 a tensor 按下标过一遍, b[i] == True 的元素会被保留, b[i] == False 的元素会被丢弃
                # 也就是 b 类型为 bool 的时候, c = a[b] 也会得到一个一维 tensor, 但其长度可能会比 a 短 (因为为 False 的都会丢掉了)

                # 所以下面这句话就是这么个意思: gaussians.max_radii2D[visibility_filter] 是一个一维 tensor, 里面记录着每个在当前视角下看得见高斯点
                # 在所有迭代步中自己出现过的最大半径
                # 所以维护方法就是自己与当前自己的半径取 max 咯. 即打竞赛里的 max_val = max(max_val, now_val)
                # 所以下面这句话的效果就是 gaussians.max_radii2D 本身会被修改——只有在 visibility_filter==True 的位置上，它的值会被替换成新的最大半径。
                # 但是 gaussians.max_radii2D 这个一维 tensor 的长度没变
                gaussians.max_radii2D[visibility_filter] = \
                    torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                
                # 统计下高斯点们的梯度信息 (梯度累计和, 梯度累计次数)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 定义要剪枝的高斯点半径阈值 (这里 iteration > opt.opacity_reset_interval 是因为迭代初期不设定剪枝阈值, 让高斯点们自由生长)
                    # 以及为啥要用 opt.opacity_reset_interval 这个跟透明度挂钩的属性作为门槛, 因为每隔 opt.opacity_reset_interval 步就会把高斯点的透明度重置
                    # 所以 opt.opacity_reset_interval 这个时间点是高斯点们已经生长的差不多了且透明度也稳定了的一个时间点
                    # 本质上我认为其实可以多定义一个自定义参数控制何时开启剪枝阈值的, 但是作者没有这么做
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 稠密化 & 剪枝
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,     # 梯度阈值, 只有在这次反向传播中梯度贡献超过这个阈值的, 才会在它们周围长出新点
                        0.005,                          # 不透明度下限阈值 (低于这个值的高斯点会被剪掉)
                        scene.cameras_extent,           # 场景空间的尺度长度
                        size_threshold,                 # 剪枝半径阈值, 任何投影半径大于这个值的高斯点, 都会被剪掉
                        radii                           # 每个高斯点投影到图像上时的像素半径
                    )
                
                # 每隔 opt.opacity_reset_interval 步就把高斯点的透明度重置
                # 以及如果是白底背景的话, 在进行第一次 稠密化 & 剪枝 前也会重置透明度一次
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # 优化曝光补偿优化器的参数
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    # 补充个知识点:
                    # 渲染器先判断每个高斯点在相机坐标系里是否落在视锥内, 且没有被前面的点遮挡, 从而得出 visibility_filter
                    # 对于通过了 visibility_filter 的点, 它们的半径会被渲染器计算出来, 作为 radii
                    # 当开启稀疏 Adam 加速时, 只利用那些像素级别 radii > 0 (而非几何级别可见但像素级别不可见) 的点进行参数优化
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # 优化高斯模型的参数
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存下 ckpt
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # 如果没有输出保存路径, 则将保存到 ./output/unique_str[0:10] 目录下
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # 把当前参数配置写入到 cfg_args 文件中并保存到输出目录下
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 实例化一个 tensorboard 对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,                  # tensorboard 对象
    iteration,                  # 当前迭代步
    Ll1,                        # 光度损失
    loss,                       # 总损失
    l1_loss,                    # 计算 L1 损失的函数
    elapsed,                    # 当前次迭代耗时
    testing_iterations,         # test 的时间点, 如果到了该 test 的时间点那么会在打完报告后 test 一下
    scene : Scene,              # 场景对象
    renderFunc,                 # 渲染函数
    renderArgs,                 # 渲染参数, 是一个元组 (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    train_test_exp              # 同时代表双重含义: 是否启用曝光补偿 / 是否启用 Synthetic‐NeRF 训练评估模式
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 如果到了 test 的时间点, 就测一把
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            # 这里拿的是 getTestCameras(), 之前是 getTrainCameras()
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            # 从训练集中抽 5 张代表图, 做可视化对比, 确认模型对已见视角的质量并无退化。
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # l1_loss 和 psnr 都清零
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs
                        )["render"], 0.0, 1.0
                    )   # tensor, [c, h, w], 范围: [0, 1], float32
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)   # 训练里没 clamp, 这里 clamp 了, 其实没必要我觉得
                    
                    # 如果启用 Synthetic‐NeRF 训练评估模式, 就只用右半边图片做 test
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    # 可视化前 5 张的渲染图与原图对比图
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],        # -> [1 c h w]
                            global_step=iteration
                        )
                        # 因为每次 getTestCameras() / getTrainCameras() 拿到的视角是一样的, 所以 gt 视角图片只用可视化到 tensorboard 一次就行了
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 当前模型高斯点的透明度直方图
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)   # 当前模型的高斯点数量
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)            # 注册数据加载相关参数
    op = OptimizationParams(parser)     # 注册优化器相关参数
    pp = PipelineParams(parser)         # 注册渲染相关参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])  # 解析所有命令行参数, 把值填到 args 对象里
    args.save_iterations.append(args.iterations)    # 迭代次数的最后一次一定会保存一次 Gaussian model
    
    print("Optimizing " + args.model_path)

    # 设置是否在控制台输出内容 + 设置随机种子 + 指定 GPU 设备
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    # 是否把 pytorch 的异常信息打印出来 (当程序出错时)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),               # 数据加载参数对象
        op.extract(args),               # 优化器参数对象
        pp.extract(args),               # 流水线参数对象
        args.test_iterations,       # test 的时间点
        args.save_iterations,       # save model 的时间点
        args.checkpoint_iterations, # save ckpt 的时间点
        args.start_checkpoint,      # 断点续训的 ckpt 路径
        args.debug_from,            # debug 开关开启的时间点    
    )

    # All done
    print("\nTraining complete.")
