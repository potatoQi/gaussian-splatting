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
	torch::Tensor	// invdepth: 反深度图
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
	const torch::Tensor& sh,				// sh 系数
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
	// NOTE: 我暂时不知道这三个 tensor 的具体代表什么, 有什么用
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

	// 弄三个扩容函数, 分别对应 geomBuffer, binningBuffer, imgBuffer
	// 举个例子:
	// char* ptr = geomFunc(100); // 这时候 geomBuffer 的 size 就变成了 100, 且 ptr 里存的是 geomBuffer 的首地址
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	// NOTE: 这个变量表示什么意思我也不理解目前
	int rendered = 0;

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
			sh.contiguous().data_ptr<float>(),				// sh 系数
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
			debug											// 是否开启调试模式
		);
	}

	return std::make_tuple(
		rendered,		// NOTE: ?
		out_color,		// [3 H W], 渲染图
		radii,			// [P], 每个高斯点投影半径
		geomBuffer,		// NOTE: ?
		binningBuffer,	// NOTE: ?
		imgBuffer,		// NOTE: ?
		out_invdepth	// [1 H W], 反深度图
	);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
