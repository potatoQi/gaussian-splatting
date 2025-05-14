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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	int P,							// 高斯点数量
	int D,							// sh 的阶数
	int M,							// sh 的系数数量
	const float* orig_points,		// 高斯点的 3D 坐标 [P 3]
	const glm::vec3* scales,		// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const float scale_modifier,		// 控制高斯体们的尺寸, 缩放因子
	const glm::vec4* rotations,		// 每个高斯体的旋转变量
	const float* opacities,			// 所有高斯体的不透明度
	const float* shs,				// sh 系数
	bool* clamped,							// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位 (待写入)
	const float* cov3D_precomp,		// 预先计算好的协方差矩阵 (若有)
	const float* colors_precomp,	// 预先计算好的 RGB 颜色 (若有)
	const float* viewmatrix,		// 视图矩阵
	const float* projmatrix,		// 投影矩阵
	const glm::vec3* cam_pos,		// 相机在世界里的坐标
	const int W,					// 图像宽度
	const int H,					// 图像高度
	const float tan_fovx,			// 单位深度处的半宽度
	const float tan_fovy,			// 单位深度处的半高度
	const float focal_x,			// x 轴焦距
	const float focal_y,			// y 轴焦距
	int* radii,								// [P] 每个高斯点投影半径 (要往里填东西)
	float2* points_xy_image,				// [P 2] 输出的 2D 投影中心
	float* depths,							// [P] 每个高斯点的深度 (要往里填东西)
	float* cov3Ds,							// [P 6] 输出的协方差矩阵 (若没预传 cov3D_precomp, 就输出到这里)
	float* rgb,								// [P 3] 输出的投影点 RGB 颜色 (如果没预传 colors_precomp, 就输出到这里)
	float4* conic_opacity,					// 输出的 2D 协方差矩阵的逆矩阵 和 输入的透明度
	const dim3 grid,				// 所需 blocks 数量
	uint32_t* tiles_touched,				// [P] 输出的每个投影点影响到的 tile 数量
	bool prefiltered,				// 是否开启预滤波
	bool antialiasing,				// 是否开启抗锯齿
	float* out_depths
) {
	// 分配一个 thread 编号
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;	// 第 idx 个高斯点在相机空间的坐标
	// in_frustum(...) 会把第 idx 个高斯点的世界坐标 orig_points 乘视图矩阵 viewmatrix 变换到相机空间, 结果存到 p_view 里
	// 如果这个点的 p_view.z 太近了, 就返回 False
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	// projmatrix 矩阵是观测变换和投影变换的叠加, 所以下面就是把点的世界坐标变换到屏幕空间, 并且归一化到 [-1, 1] 立方体里面
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// cov3D 指针指向第 idx 个高斯点的那 6 个浮点数 (6 个数构成了协方差矩阵)
	// 算 conv3D 的作用是算 conv2D, 算出 conv2D 之后, 就可以算出 2D 椭圆的主轴长度
	const float* cov3D;
	if (cov3D_precomp != nullptr) {
		cov3D = cov3D_precomp + idx * 6;
	}
	else {
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 算出 2D 协方差矩阵 [2x2] 的, 不过因为是对称矩阵, 所以只需要存 3 个数
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 下面是抗锯齿才会用到的, 最终算出的 h_convolution_scaling 会影响传入的 opacities, 我暂且不管
	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	// 2D 协方差矩阵的特征值
	const float det = det_cov_plus_h_cov;
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	// conic 是 2D 协方差的逆矩阵
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 这里就是算出长轴的一半 my_radius, 待会用来作为近似圆的半径
	// 可是为什么 my_radius 还要乘 3 呢? 因为无论是 3D 高斯, 还是投影的 2D 高斯, 为了方便理解, 我们分别把他俩理解为 3D 椭圆和 2D 椭圆
	// 可是他们的参数是控制一个高斯分布参数的参数, 所以他们本质上是一个无限大的椭圆, 只不过越靠中心的椭圆出现的概率越大, 越远离中心的椭圆出现的概率越小
	// 所以说, 我们求出的原本的 my_radius 对应着的是 Iσ, 为了覆盖 >99% 的高斯分布, 我们需要乘上 3 (3 个标准差保证几乎能覆盖所有的高斯分布)
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	// p_proj 是压缩到 [-1, 1] 立方体里面的坐标 (NDC), ndc2Pix() 的作用就是把 ndc 坐标映射到像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// grid 是用 16x16 的 thread 覆盖 [H W] 所需要的 blocks 数量, 其实也就是 tile 用二维坐标表示的数量
	getRect(
		point_image,	// 圆心
		my_radius,		// 半径
		rect_min,	// 二维坐标索引 (表示覆盖的 tile 们的左上角的 tile 的二维坐标)
		rect_max,	// 二维坐标索引 (表示覆盖的 tile 们的右下角的 tile 的二维坐标各自加 1) (比如实际覆盖的右下角是 (2,2), 那么 rect_max 就是 (3,3))
		grid			// grid.x 是 tile 二维 x 坐标的上界, grid.y 是 tile 是二维 y 坐标的上界
	);
	// 如果这个圆一个 tile 都不覆盖, 那就没必要继续了
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 这一段代码的逻辑就是: 如果外面没给预先算好的颜色, 就送 sh 系数里现场算一次 rgb 颜色, 然后存到 rgb 缓冲里
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(
			idx,						// 当前是第 idx 个高斯点
			D,							// sh 的阶数
			M,							// sh 的系数总数
			(glm::vec3*)orig_points,	// 高斯点的世界坐标
			*cam_pos,					// 相机的世界坐标
			shs,						// sh 系数
			clamped						// 表示该 idx 个高斯点的 R/G/B 通道的值有没有被截断
		);
		// 该高斯点的 R/G/B 通道的值存下来
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// 噢所以它拿的深度值其实是相机坐标系下的 z 轴值, 而不是 ndc 坐标系里的 z 轴值
	depths[idx] = p_view.z;
	out_depths[idx] = p_view.z;
	// my_radius 是近似圆的半径
	radii[idx] = my_radius;
	// point_image 是高斯点投影到屏幕上的坐标
	points_xy_image[idx] = point_image;

	// 把第 idx 个高斯点传入的透明度取出来
	float opacity = opacities[idx];
	// 前三个分量是 2D 协方差矩阵的逆矩阵, 第四个分量是透明度; 相当于打包起来保存
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

	// 计算这个高斯点影响到的 tile 的数量
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,				// [tileID, 2] tile 的涉及范围, 值是 point_list_keys 中的索引
	const uint32_t* __restrict__ point_list,		// 对应着 point_list_keys 中负责的高斯投影圆心的索引
	int W,											// 图像宽度
	int H,											// 图像高度
	const float2* __restrict__ points_xy_image,		// [P 2] 高斯投影圆心的坐标
	const float* __restrict__ features,				// [P 3] 高斯投影圆心的 RGB 颜色
	const float4* __restrict__ conic_opacity,		// 高斯投影圆心的 2D 协方差矩阵的逆矩阵 和 输入的不透明度
	float* __restrict__ final_T,							// 每个 pixel 的剩余透射率 (要往里填东西)
	uint32_t* __restrict__ n_contrib,						// 实际影响到该 pixel 的高斯体实例数量 (要往里填东西)
	const float* __restrict__ bg_color,				// 背景颜色
	float* __restrict__ out_color,							// 渲染图像 (要往里填东西)
	const float* __restrict__ depths,				// [P] 高斯投影深度
	float* __restrict__ invdepth,							// 反深度图 (要往里填东西)
	float* __restrict__ gauss_sum,
	int* __restrict__ gauss_count,
	int* __restrict__ last_contr_gauss,
	float* __restrict__ out_accum_alpha
) {
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// 二维下标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 一维下标
	uint32_t pix_id = W * pix.y + pix.x;
	// 二维下标 (浮点版)
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	// 这个线程是否在范围内
	bool inside = pix.x < W&& pix.y < H;
	// 那些边界外的 thread, 就让它们的 done=True, 后边不让它们去执行渲染操作, 只让它们去执行一下 fetching 工作就好了
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// block.group_index().y * horizontal_blocks + block.group_index().x 是当前 thread 所在 tile 的编号
	// range[tileID].x ~ range[tileID].y 是当前 tile 负责的高斯点的索引范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// rounds 表示如果一轮处理 BLOCK_SIZE 个高斯体实例, 全部处理完 range.y - range.x 个高斯体实例, 需要多少轮 (range 是左闭右开的)
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 这个 tile 一共有多少个高斯体实例需要遍历
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// 这里比较巧妙的, 既然是并行所有像素去计算, 为什么前面还要计算这个像素属于哪个 tile 呢?
	// 原因就是一个 tile 内需要读取的高斯体实例信息是一样的。因为当时排序就是先按 tile 从小到大, 再按 depth 从小到大
	// 所以一个 tile 里有 256 个 pixel 嘛, 这些 pixel 既然都属于一个 tile, 所以需要用到的高斯体实例信息都是一样的
	// 那么这些信息存一次就好, 不需要存 256 次; shared memory 就是解决这个问题, share memory 的作用域是一个 block, 刚好对应着一个 tile
	// 即只有这个 block 里任意一个 thread 存了数据, 那么其它的 threads 也能看到

	// 定义占位好; collected_id 是高斯体实例的索引, collected_xy 是高斯体投影圆心的坐标, collected_conic_opacity 是高斯体实例的 2D 协方差逆矩阵 + 透明度
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	// T 是剩余透射率, 初始为 1.0 表示光线还未被任何高斯体吸收, 后面每遇到一个不透明度为 α 的高斯体, 就 T *= (1 - α)
	float T = 1.0f;
	// contributor 表示影响到该 pixel 的高斯体实例数量
	uint32_t contributor = 0;
	// 记录最后一个对该 pixel 产生贡献的高斯体实例数量编号 (1 <= last_contributor <= contributor)
	uint32_t last_contributor = 0;
	// 该 pixel CHANNEL 通道的值
	float C[CHANNELS] = { 0 };
	// 该 pixel 的反深度值
	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {	// 一次处理 BLOCK_SIZE 个高斯体实例 
		// End if entire block votes that it is done rasterizing
		// 因为有可能覆盖到这个 tile 的高斯体数量非常之多, 但是一个 tile 只有 BLOCK_SIZE 个 thread, 所以一次最多只能处理 BLOCK_SIZE 个高斯体实例
		int num_done = __syncthreads_count(done);	// 统计当前 block 中有多少个 thread 已经是 done
		if (num_done == BLOCK_SIZE) // 如果当前 block (tile) 中所有的 thread 都已经是 done 了, 那就 break
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 下面其实就是给每个 pixel 分配一个高斯体实例, 让这个 pixel 去读取高斯体实例的信息
		int progress = i * BLOCK_SIZE + block.thread_rank();	// 进度: 表示处理到第几个高斯体实例了
		if (range.x + progress < range.y) { // 如果这个 pixel 负责的高斯体实例在该 tile 要处理的高斯体实例范围内
			// 取出的 coll_id 是高斯体的 idx
			int coll_id = point_list[range.x + progress];
			// 当前这个 pixel 把 idx 这个高斯体实例存在自己下标的 shared memory 里
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		// 保证当前 tile 里所有 thread 都完成上述运算才进行下一步
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) { // j 是遍历当前 shared memory 中的高斯体实例 (如果当前 pixel 已经渲染完毕就不进这里)
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// 下面就是计算出一个 power, 这个 power 的取值跟是否这个高斯实例会对当前 pixel 产生影响有关
			float2 xy = collected_xy[j];	// 该高斯体实例的投影圆心坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// 运行到这里说明光线已经砸到了当前高斯体头上
			int idx_gaussian = collected_id[j];	// 该高斯体实例的索引
			// 把 test_T 累加到下标为 idx_gaussian 的高斯体上去, 然后该高斯体的 count++
			// 这里用原子操作是因为可能有多个 pixel 同时命中同一个 idx_gaussian, 而 += 这个操作会先读一个旧值然后加新值再存回去, 同时命中会导致数据错乱
			// 所以就用原子操作
			atomicAdd(&gauss_sum[idx_gaussian], T);
			atomicAdd(&gauss_count[idx_gaussian], 1);

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// alpha=1 是不透明, alpha=0 是透明
			// 计算该高斯体对当前 pixel 的不透明度 = 高斯体本身的不透明度 * 系数 (越远系数越小)
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue; // 如果 alpha < 1/255, 说明该高斯体对于该 pixel 来说太透明了, 没影响, 直接 continue
			float test_T = T * (1 - alpha); // 1-α 代表应该削弱的比例, T 是该 pixel 剩余透射率
			if (test_T < 0.0001f) { // 该 pixel 已经不能继续透射光线了, 即不会再接受当前以及以后别的高斯体影响, 即渲染完成, done=True
				done = true;
				continue;
			}

			// TEST: 功能 1 的测试代码, 若开启请把上面两行原子操作注释掉
			// atomicAdd(&gauss_sum[idx_gaussian], test_T);
			// atomicAdd(&gauss_count[idx_gaussian], 1);

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				//features[collected_id[j] * CHANNELS + ch] 是该高斯投影圆心 ch 通道的值
				// 这个值首先要受到在该 pixel 处的不透明度的影响, 其次要受到该 pixel 剩余透射率的影响
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			// 同理, 累计该 pixel 处的反深度值
			if(invdepth)
				expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			last_contr_gauss[pix_id] = idx_gaussian;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside) {	// 运行到这里, 如果 inside=True, 说明该 pixel 已经渲染完毕 
		final_T[pix_id] = T;
		out_accum_alpha[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 颜色 + 背景颜色

		if (invdepth)
			invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid,				// (grid.x, grid.y, 1)--屏幕被划分为多少个 tile
	const dim3 block,				// (16, 16, 1)--每个 tile 被划分为多少个 thread
	const uint2* ranges,			// [tileID, 2] tile 的涉及范围, 值是 point_list_keys 中的索引
	const uint32_t* point_list,		// 对应着 point_list_keys 中负责的高斯投影圆心的索引
	int W,							// 图像宽度
	int H,							// 图像高度
	const float2* means2D,			// [P 2] 高斯投影圆心的 2D 坐标
	const float* colors,			// [P 3] 高斯投影圆心的 RGB 颜色
	const float4* conic_opacity,	// 高斯投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
	float* final_T,							// 每个 pixel 的剩余透射率 (要往里填东西)
	uint32_t* n_contrib,					// 实际影响到该 pixel 的高斯体实例数量 (要往里填东西)
	const float* bg_color,			// 背景颜色
	float* out_color,						// 渲染图像 (要往里填东西)
	float* depths,					// [P] 高斯投影深度
	float* depth,							// 反深度图 (要往里填东西)
	float* gauss_sum,
	int* gauss_count,
	int* last_contr_gauss,
	float* out_accum_alpha
) {
	// 并行处理每个像素
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W,
		H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth,
		gauss_sum,
		gauss_count,
		last_contr_gauss,
		out_accum_alpha
	);
}

void FORWARD::preprocess(
	int P,							// 高斯点数量
	int D,							// sh 的阶数
	int M,							// sh 的系数数量
	const float* means3D,			// 高斯点的 3D 坐标 [P 3]
	const glm::vec3* scales,		// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const float scale_modifier,		// 控制高斯体们的尺寸, 缩放因子
	const glm::vec4* rotations,		// 每个高斯体的旋转变量
	const float* opacities,			// 所有高斯体的不透明度
	const float* shs,				// sh 系数
	bool* clamped,							// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位 (待写入)
	const float* cov3D_precomp,		// 预先计算好的协方差矩阵 (若有)
	const float* colors_precomp,	// 预先计算好的 RGB 颜色 (若有)
	const float* viewmatrix,		// 视图矩阵
	const float* projmatrix,		// 投影矩阵
	const glm::vec3* cam_pos,		// 相机在世界里的坐标
	const int W,					// 图像宽度
	const int H,					// 图像高度
	const float focal_x,			// x 轴焦距
	const float focal_y,			// y 轴焦距
	const float tan_fovx,			// 单位深度处的半宽度
	const float tan_fovy,			// 单位深度处的半高度
	int* radii,								// [P] 每个高斯点投影半径 (要往里填东西)
	float2* means2D,						// [P 2] 输出的 2D 投影中心
	float* depths,							// [P] 每个高斯点的深度 (要往里填东西)
	float* cov3Ds,							// [P 6] 输出的协方差矩阵 (如果没预传 cov3D_precomp, 就输出到这里)
	float* rgb,								// [P 3] 输出的投影点 RGB 颜色 (如果没预传 colors_precomp, 就输出到这里)
	float4* conic_opacity,					// 输出的 2D 协方差矩阵的逆矩阵 和 输入的透明度
	const dim3 grid,				// 所需 blocks 数量
	uint32_t* tiles_touched,				// [P] 输出的每个投影点影响到的 tile 数量
	bool prefiltered,				// 是否开启预滤波
	bool antialiasing,				// 是否开启抗锯齿
	float* out_depths
) {
	// <NUM_CHANNELS> 是模板参数, 会传给 preprocessCUDA 的 C
	// lauch kernel, 有 (P+255)/256 个 block, 每个 block 有 256 个 thread
	// 一共有 P+255 个 thread, 保证了覆盖 P 个点
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		D,
		M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W,
		H,
		tan_fovx,
		tan_fovy,
		focal_x,
		focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing,
		out_depths
	);
}
