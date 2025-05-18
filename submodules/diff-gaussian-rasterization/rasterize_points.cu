/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<
	int,			// num_rendered: 记录每个像素实际上被多少个高斯点贡献了颜色
	torch::Tensor,	// color: 三通道 RGB 图像 
	torch::Tensor,	// radii: 高斯体投影到屏幕上的像素半径
	torch::Tensor,	// geomBuffer: 存储每个像素对应哪些高斯点索引
	torch::Tensor,	// binningBuffer: 加速查找哪些点落在哪个屏幕区域
	torch::Tensor,	// imgBuffer: 临时存放中间的像素数据，方便分批处理。这些 Buffer 在前向存好，为后向梯度计算提供随机访问。
	torch::Tensor,	// invdepth: 反深度图
	torch::Tensor,	// accum_alpha: 每个 pixel 的剩余透射率
	torch::Tensor,	// gauss_sum
	torch::Tensor,	// gauss_count
	torch::Tensor,	// last_contr_gauss
	torch::Tensor	// out_depths
>
RasterizeGaussiansCUDA(
	// 这里传的是地址, 省空间; 同时用了 const, 所以同时避免了不小心的修改操作
	const torch::Tensor& background,		// 背景颜色
	const torch::Tensor& means3D,			// 所有高斯体的 3D 坐标 [P 3]
    const torch::Tensor& colors,			// 预先计算好的 RGB 颜色 (若有)
    const torch::Tensor& opacity,			// 所有高斯体的不透明度
	const torch::Tensor& scales,			// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const torch::Tensor& rotations,			// 每个高斯体的旋转变量
	const float scale_modifier,				// 控制高斯体们的尺寸, 缩放因子
	const torch::Tensor& cov3D_precomp,		// 预先计算好的协方差矩阵 (若有)
	const torch::Tensor& viewmatrix,		// 视图矩阵
	const torch::Tensor& projmatrix,		// 投影矩阵
	const float tan_fovx, 					// 单位深度处的半宽度
	const float tan_fovy,					// 单位深度处的半高度

    const int image_height,					// 输出图像的像素高
    const int image_width,					// 输出图像的像素宽
	const torch::Tensor& sh,				// sh 系数 [P M D(3)]
	const int degree,						// sh 的阶数
	const torch::Tensor& campos,			// 相机在世界里的坐标
	const bool prefiltered,					// 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
	const bool antialiasing,				// 是否开启抗锯齿
	const bool debug						// 是否开启调试模式
) {
	// 传入的 means3D 的 shape 必须满足 [P 3]
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);	// 点的数量
	const int H = image_height;
	const int W = image_width;

	// 创建俩 options(), 用来新建 tensor 时指定 dtype, device 啥的
	// 分别弄了个 int 和 float 的
	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	// 创建占位 tensor, 渲染图像和反深度图; 后边往里填东西, 最后要 return 的
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);	// [3 H W]
	torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);	// 我不懂这句话存在的意义是什么, 后边又赋值了一遍
	float* out_invdepthptr = nullptr;	// 声明一个 float 类型指针

	out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();	// [1 H W]
	out_invdepthptr = out_invdepth.data<float>();	// out_invdepthptr 指向 out_invdepth tensor 的首元素

	// 占位 tensor, 每个高斯点投影半径
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));	// [P]
	
	// 创建一个 device, 放到 gpu 上
	torch::Device device(torch::kCUDA);
	// 创建一个 options(), 数据类型为 uint8
	torch::TensorOptions options(torch::kByte);

	// 这里创建了三个空 tensor, geomBuffer, binningBuffer, imgBuffer; 数据类型为 uint8, 放在 gpu 上
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

	// 弄三个扩容函数, 分别对应 geomBuffer, binningBuffer, imgBuffer
	// 举个例子:
	// char* ptr = geomFunc(100); // 这时候 geomBuffer 的 size 就变成了 100, 且 ptr 里存的是 geomBuffer 的首地址
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	// 渲染动作的总次数
	int rendered = 0;

	// 定义一个指针, 接收每个 pixel 的剩余透射率 (要往里填东西)
	torch::Tensor accum_alpha = torch::empty({1, H, W}, float_opts.device(torch::kCUDA));
	float* accum_alpha_ptr = accum_alpha.data_ptr<float>();

	torch::Tensor gauss_sum = torch::zeros({P}, float_opts.device(torch::kCUDA));
	float* gauss_sum_ptr = gauss_sum.data_ptr<float>();
	torch::Tensor gauss_count = torch::zeros({P}, int_opts.device(torch::kCUDA));
	int* gauss_count_ptr = gauss_count.data_ptr<int>();

	torch::Tensor last_contr_gauss = torch::full({1, H, W}, -1, int_opts.device(torch::kCUDA));
	int* last_contr_gauss_ptr = last_contr_gauss.data_ptr<int>();

	torch::Tensor out_depths = torch::empty({P}, float_opts.device(torch::kCUDA));
	float* out_depths_ptr = out_depths.data_ptr<float>();

	if(P != 0) {
		// M 是每个高斯点球谐系数的数量
		int M = 0;
		if(sh.size(0) != 0) {
			M = sh.size(1);
		}

		// 这个函数实现在 cuda_rasterizer/rasterizer_impl.cu 里; 声明在 cuda_rasterizer/rasterizer.h 里
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,		// geomBuffer 扩容函数 & 返回指针 (要往里填东西)
			binningFunc,	// binningBuffer 扩容函数 & 返回指针 (要往里填东西)
			imgFunc,		// imgBuffer 扩容函数 & 返回指针 (要往里填东西)
			P,			// 高斯点数量
			degree,		// sh 的阶数
			M,			// sh 系数的数量
			background.contiguous().data<float>(),			// 背景颜色
			W,												// 图像宽度
			H,												// 图像高度
			means3D.contiguous().data<float>(),				// 所有高斯体的 3D 坐标 [P 3]
			sh.contiguous().data_ptr<float>(),				// sh 系数 [P M D(3)]
			colors.contiguous().data<float>(), 				// 预先计算好的 RGB 颜色 (若有)
			opacity.contiguous().data<float>(), 			// 所有高斯体的不透明度
			scales.contiguous().data_ptr<float>(),			// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
			scale_modifier,									// 控制高斯体们的尺寸, 缩放因子
			rotations.contiguous().data_ptr<float>(),		// 每个高斯体的旋转变量
			cov3D_precomp.contiguous().data<float>(), 		// 预先计算好的协方差矩阵 (若有)
			viewmatrix.contiguous().data<float>(), 			// 视图矩阵
			projmatrix.contiguous().data<float>(),			// 投影矩阵
			campos.contiguous().data<float>(),				// 相机在世界里的坐标
			tan_fovx,										// 单位深度处的半宽度
			tan_fovy,										// 单位深度处的半高度
			prefiltered,									// 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
			
			out_color.contiguous().data<float>(),	// 渲染图像 (要往里填东西)
			out_invdepthptr,						// 反深度图 (要往里填东西)
			antialiasing,									// 是否开启抗锯齿
			radii.contiguous().data<int>(),			// 每个高斯点投影半径 (要往里填东西)
			debug,											// 是否开启调试模式
			accum_alpha_ptr,							// 每个 pixel 的剩余透射率 (要往里填东西)
			gauss_sum_ptr,
			gauss_count_ptr,
			last_contr_gauss_ptr,
			out_depths_ptr
		);
	}

	return std::make_tuple(
		rendered,		// 渲染动作的总次数
		out_color,		// [3 H W], 渲染图
		radii,			// [P], 每个高斯点投影半径
		geomBuffer,
		binningBuffer,
		imgBuffer,
		out_invdepth,	// [1 H W], 反深度图
		accum_alpha,		// [1 H W], 每个 pixel 的剩余透射率
		gauss_sum,
		gauss_count,
		last_contr_gauss,
		out_depths
	);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,			// 背景颜色
	const torch::Tensor& means3D,				// 所有高斯体的 3D 坐标 [P 3]
	const torch::Tensor& radii,					// 每个高斯点投影半径 [P]
    const torch::Tensor& colors,				// 预先计算好的 RGB 颜色 (若有)
	const torch::Tensor& opacities,				// 所有高斯体的不透明度
	const torch::Tensor& scales,				// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const torch::Tensor& rotations,				// 每个高斯体的旋转变量
	const float scale_modifier,					// 控制高斯体们的尺寸, 缩放因子
	const torch::Tensor& cov3D_precomp,			// 预先计算好的协方差矩阵 (若有)
	const torch::Tensor& viewmatrix,			// 视图矩阵
    const torch::Tensor& projmatrix,			// 投影矩阵
	const float tan_fovx,						// 单位深度处的半宽度
	const float tan_fovy,						// 单位深度处的半高度
    const torch::Tensor& dL_dout_color,					// loss 对渲染 RGB 图像的梯度 [3 H W] (backward 的输入参数)
	const torch::Tensor& dL_dout_invdepth,				// loss 对反深度图的梯度 [1 H W]	  (backward 的输入参数)
	const torch::Tensor& sh,					// sh 系数 [P M(16) D(3)], M 通常是 (degree+1)^2
	const int degree,							// sh 的阶数
	const torch::Tensor& campos,				// 相机在世界里的坐标
	const torch::Tensor& geomBuffer,
	const int R,										// 渲染动作的总次数, 也就是有多少个 (tileID, depth) pairs
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,					// 是否开启抗锯齿
	const bool debug							// 是否开启调试模式
) {
	const int P = means3D.size(0);			// 高斯点的数量
	const int H = dL_dout_color.size(1);	// 图像的高度
	const int W = dL_dout_color.size(2);	// 图像的宽度
	
	int M = 0;	// sh 系数的数量
	if(sh.size(0) != 0) {	
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());			// loss 对高斯点 3D 坐标的梯度 [P 3]
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());			// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());	// loss 对高斯点 RGB 颜色的梯度 [P 3]
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());			// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());			// loss 对高斯体不透明度的梯度
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());				// loss 对预先计算好的协方差的矩阵 (若有) 的梯度 (要么走这条路, 要么走下面的 dL_dscales, dL_drotations)
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());				// loss 对 sh 系数的梯度 [P M D]
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());				// loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());			// loss 对每个高斯体的旋转变量的梯度 [P 4]
	torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());			// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	
	// 答疑:
	// 为什么相比于 return 的那几个梯度, 多了一个 dL_dconic 和 dL_dinvdepths? 我们在 .apply(...) 的参数也妹有这俩输入啊
	// 这是因为, conic 和 invdepths 是中间量, 还记得 conic 和 invdepths 是怎么计算出来的吗?
	// conic 是由 conv3D 算出来的, conv3D 又是由 scales, rotations, invdepths 算出来的。所以我们当然需要 dL_dconic, 这样才能链式传递到我们想要的 dL_dscales, dL_drotations 上面
	// invdepths 是 view.z, view 又是由 means3D 坐标算出来的, 所以我们当然需要 dL_dinvdepths, 这样才能链式传递到我们想要的 dL_dmeans3D 上面

	float* dL_dout_invdepthptr = nullptr;	// loss 对反深度图的梯度 [1 H W] 的指针
	float* dL_dinvdepthsptr = nullptr;		// loss 对每个高斯体投影深度 view.z 的梯度 [P 1] 的指针

	// 只有 forward 有反深度图时才去计算 dL_dinvdepths
	if(dL_dout_invdepth.size(0) != 0) {
		dL_dout_invdepthptr = dL_dout_invdepth.data<float>();

		dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
		dL_dinvdepths = dL_dinvdepths.contiguous();
		dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	}

	if(P != 0) {  
		CudaRasterizer::Rasterizer::backward(
			// 基本尺寸参数
			P,											// 高斯点数量
			degree,										// sh 的阶数
			M,											// sh 系数的数量
			R,											// 前向所有 (tileID, depth) pairs 的个数

			// 前向需要的常量输入 (只读)
			background.contiguous().data<float>(),		// 背景颜色
			W,											// 图像宽度
			H, 											// 图像高度
			means3D.contiguous().data<float>(),			// 所有高斯体的 3D 坐标 [P 3]
			sh.contiguous().data<float>(),				// sh 系数 [P M D]
			colors.contiguous().data<float>(),			// 预先计算好的 RGB 颜色 (若有)
			opacities.contiguous().data<float>(),		// 所有高斯体的不透明度
			scales.data_ptr<float>(),					// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
			scale_modifier,								// 控制高斯体们的尺寸, 缩放因子
			rotations.data_ptr<float>(),				// 每个高斯体的旋转变量
			cov3D_precomp.contiguous().data<float>(),	// 预先计算好的协方差矩阵 (若有)
			viewmatrix.contiguous().data<float>(),		// 视图矩阵
			projmatrix.contiguous().data<float>(),		// 投影矩阵
			campos.contiguous().data<float>(),			// 相机在世界里的坐标
			tan_fovx,									// 单位深度处的半宽度
			tan_fovy,									// 单位深度处的半高度

			// 前向算出的中间量
			radii.contiguous().data<int>(),										// 每个高斯点投影半径 [P]
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),		// geomBuffer
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),		// binningBuffer
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),		// imgBuffer

			// 从 Python 端传回的 loss 对 forward 输出的梯度
			dL_dout_color.contiguous().data<float>(),		// loss 对渲染 RGB 图像的梯度 [3 H W]
			dL_dout_invdepthptr,							// loss 对反深度图的梯度 [1 H W]

			// 需要往里写入值的梯度
			dL_dmeans2D.contiguous().data<float>(),				// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
			dL_dconic.contiguous().data<float>(),  				// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
			dL_dopacity.contiguous().data<float>(),				// loss 对高斯体不透明度的梯度 [P 1]
			dL_dcolors.contiguous().data<float>(),				// loss 对高斯点 RGB 颜色的梯度 [P 3]
			dL_dinvdepthsptr,									// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
			dL_dmeans3D.contiguous().data<float>(),				// loss 对高斯点 3D 坐标的梯度 [P 3]
			dL_dcov3D.contiguous().data<float>(),				// loss 对预先计算好的协方差的矩阵 (若有) 的梯度 [P 6]
			dL_dsh.contiguous().data<float>(),					// loss 对 sh 系数的梯度 [P M D]
			dL_dscales.contiguous().data<float>(),				// loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
			dL_drotations.contiguous().data<float>(),			// loss 对每个高斯体的旋转变量的梯度 [P 4]

			// 额外开关
			antialiasing,			// 是否开启抗锯齿
			debug					// 是否开启调试模式
		);
	}

	return std::make_tuple(
		dL_dmeans2D,			// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
		dL_dcolors,				// loss 对高斯点 RGB 颜色的梯度 [P 3]
		dL_dopacity,			// loss 对高斯体不透明度的梯度
		dL_dmeans3D,			// loss 对高斯点 3D 坐标的梯度
		dL_dcov3D,				// loss 对预先计算好的协方差的矩阵 (若有) 的梯度
		dL_dsh,					// loss 对 sh 系数的梯度
		dL_dscales,				// loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度
		dL_drotations			// loss 对每个高斯体的旋转变量的梯度
	);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix
) { 
	const int P = means3D.size(0);
	
	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
	
	if(P != 0) {
		CudaRasterizer::Rasterizer::markVisible(P,
			means3D.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>());
	}
	
	return present;
}
