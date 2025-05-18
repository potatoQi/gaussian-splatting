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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,									// 高斯点数量
	const float2* points_xy,				// [P 2] 高斯点的 2D 坐标
	const float* depths,					// [P] 高斯点的 2D 投影深度
	const uint32_t* offsets,				// [P] 前缀和数组，指示每个高斯点实例化自己影响到的 tile 时的写入起始偏移
	uint64_t* gaussian_keys_unsorted,				// 输出: [num_rendered]  tile ID(前 32 位) + depth(后 32 位)
	uint32_t* gaussian_values_unsorted,				// 输出: [num_rendered]  影响这个 tile 的高斯点的编号
	int* radii,								// [P] 每个高斯点投影半径
	dim3 grid								// (gridX,gridY,1)——屏幕被划分成多少个 tile
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0) {
		// Find this Gaussian's offset in buffer for writing keys/values.
		// off 表示当前这个高斯点实例化自己影响到的 tile 时的写入起始偏移
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++) {
			for (int x = rect_min.x; x < rect_max.x; x++) {
				// grid.x 是一行的数量, y 是在第几行(从 0 开始), x 是在第几列(从 0 开始)
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;		// tile ID + depth
				gaussian_values_unsorted[off] = idx;	// off 这个 tile 是 idx 这个高斯点影响的
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,	// geomBuffer 扩容函数 & 返回指针 (要往里填东西)
	std::function<char* (size_t)> binningBuffer,	// binningBuffer 扩容函数 & 返回指针 (要往里填东西)
	std::function<char* (size_t)> imageBuffer,		// imageBuffer 扩容函数 & 返回指针 (要往里填东西)
	const int P,			// 高斯点数量
	const int D,			// sh 的阶数
	const int M,			// sh 系数的数量
	const float* background,				// 背景颜色
	const int width,						// 图像宽度
	const int height,						// 图像高度
	const float* means3D,					// 高斯点的 3D 坐标 [P 3]
	const float* shs,						// sh 系数 [P M D(3)]
	const float* colors_precomp,			// 预先计算好的 RGB 颜色 (若有)
	const float* opacities,					// 所有高斯体的不透明度
	const float* scales,					// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const float scale_modifier,				// 控制高斯体们的尺寸, 缩放因子
	const float* rotations,					// 每个高斯体的旋转变量
	const float* cov3D_precomp,				// 预先计算好的协方差矩阵 (若有)
	const float* viewmatrix,				// 视图矩阵
	const float* projmatrix,				// 投影矩阵
	const float* cam_pos,					// 相机在世界里的坐标
	const float tan_fovx,					// 单位深度处的半宽度
	const float tan_fovy,					// 单位深度处的半高度
	const bool prefiltered,					// 表示你是否已经在别的地方对颜色做过“预滤波”（模糊、降采样）处理。这里设为 False，让 rasterizer 自己来处理
	float* out_color,			// 渲染图像 (要往里填东西)
	float* depth,				// 反深度图 (要往里填东西)
	bool antialiasing,						// 是否开启抗锯齿
	int* radii,					// 每个高斯点投影半径 (要往里填东西)
	bool debug,								// 是否开启 debug 模式
	float* out_accum_alpha,// 每个 pixel 的剩余透射率 (要往里填东西)
	float* gauss_sum,
	int* gauss_count,
	int* last_contr_gauss,
	float* out_depths
) {
	// 想象一个针孔相机, 光线从物体经过针孔, 打到后面的成像平面, 焦距就是针孔到成像平面之间的距离
	// f 越大, 成像平面离针孔远，投影的物体看上去“更大”、视野更窄（长焦）。
	// f 越小, 成像平面离针孔近，投影的物体看上去“更小”、视野更宽（广角）。
	// 下面计算的 focal_x/focal_y 就是在针孔相机的放射角固定的前提下, 为了使得成像平面大小是 height x width 的 x/y 焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// required<GeometryState>(P) 是一个编译时/运行时帮助函数, 用来计算如果要存下 P 个高斯点的 GeometryState 需要的字节大小
	// GeometryState 是一个结构体, 里面存放了高斯点的各种信息: 坐标协方差颜色不透明度等
	size_t chunk_size = required<GeometryState>(P);

	// 在 geometryBuffer 中申请 chunk_size 大小的内存, 并返回指向这块内存的指针, 用来存放 P 个点的 GeometryState
	char* chunkptr = geometryBuffer(chunk_size);

	// 简单理解就行, fromChunk 是一个静态方法, 它接受一个原始内存指针 chunkptr 和高斯点数量 P, 然后返回一个把原始内存结构化好的 GeometryState 对象
	// 后续往里填东西就行
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 正常 radii 传进来的是一块区域内存的指针, 占位用的, 待填充
	// 可是如果 radii 没分配内存, 那么就用 geomState.internal_radii 这个指针
	if (radii == nullptr) {
		radii = geomState.internal_radii;
	}

	// 这里名字起的比较诡异, 这里的 block 其实就是说一个 block 分配 16x16x1 个 thread
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	// 这里计算所需的 blocks 数量 (注意这里第一维是列)
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// required<ImageState>(width * height) 是一个编译时/运行时帮助函数, 用来计算如果要存下 width * height 个像素的 ImageState 需要的字节大小
	size_t img_chunk_size = required<ImageState>(width * height);
	// 在 imageBuffer 中申请 img_chunk_size 大小的内存, 并返回指向这块内存的指针, 用来存放 width * height 个像素的 ImageState
	char* img_chunkptr = imageBuffer(img_chunk_size);
	// 简单理解就行, fromChunk 是一个静态方法, 它接受一个原始内存指针 img_chunkptr 和像素数量 width * height, 然后返回一个把原始内存结构化好的 ImageState 对象
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	/*
		到了这里, 我们已经有:
			GeometryState geomState: 里面存放了高斯点的各种信息: 坐标协方差颜色不透明度等
			ImageState imgState: 里面存放了像素的各种信息: 像素颜色、深度等
	*/

	// 如果不是在做标准的 RGB 渲染, 那么就需要提供预先计算好的颜色. 否则报错
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 把高斯体从世界坐标下的参数转换到屏幕空间下的各种中间量, 为后续真正的光栅化 render 做准备
	CHECK_CUDA(
		FORWARD::preprocess(
			P,							// 高斯点数量
			D,							// sh 的阶数
			M,							// sh 系数的数量
			means3D,					// 高斯点的 3D 坐标 [P 3]
			(glm::vec3*)scales,			// 每个高斯体的尺度 (在 xyz 轴的缩放长度) (这里的 glm::vec3* 意思是把原本 float* 指向一大片连续浮点数的指针, 解释为 glm::vec3*, 即在代码里可以直接写 scales[i].x)
			scale_modifier,				// 控制高斯体们的尺寸, 缩放因子
			(glm::vec4*)rotations,		// 每个高斯体的旋转变量
			opacities,					// 所有高斯体的不透明度
			shs,						// sh 系数 [P M D(3)]
			geomState.clamped,						// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位 (待写入)
			cov3D_precomp,				// 预先计算好的协方差矩阵 (若有)
			colors_precomp,				// 预先计算好的 RGB 颜色 (若有)
			viewmatrix,					// 视图矩阵
			projmatrix,					// 投影矩阵
			(glm::vec3*)cam_pos,		// 相机在世界里的坐标
			width,						// 图像宽度
			height,						// 图像高度
			focal_x,					// x 轴焦距
			focal_y,					// y 轴焦距
			tan_fovx,					// 单位深度处的半宽度
			tan_fovy,					// 单位深度处的半高度
			radii,									// [P] 每个高斯点投影半径 (要往里填东西)
			geomState.means2D,						// [P 2] 输出的 2D 投影中心
			geomState.depths,						// [P] 输出的高斯投影深度
			geomState.cov3D,						// [P 6] 输出的协方差矩阵 (如果没预传 cov3D_precomp, 就输出到这里)
			geomState.rgb,							// [P 3] 输出的投影点 RGB 颜色 (如果没预传 colors_precomp, 就输出到这里)
			geomState.conic_opacity,				// 输出的 2D 协方差矩阵的逆矩阵 和 输入的透明度
			tile_grid,					// (gridX,gridY,1)——屏幕被划分成多少个 tile
			geomState.tiles_touched,				// [P] 输出的每个投影点影响到的 tile 数量
			prefiltered,				// 是否开启预滤波
			antialiasing,				// 是否开启抗锯齿
			out_depths
		),
		debug
	)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 下面就可以理解算出 geomState.tiles_touched 这个数组的前缀和数组, 存到 geomState.point_offsets 里
	// 作用就是后边用于给每个高斯点分配缓存区用的
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
			geomState.scanning_space,
			geomState.scan_size,
			geomState.tiles_touched,
			geomState.point_offsets,
			P
		),
		debug
	)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// 下面这段就是我们待会要进行多少次渲染, 渲染的总次数就是 num_rendered, 也就是 geomState.point_offsets[P - 1], 即所有高斯点会影响到的 tiles 数量的总和
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(
			&num_rendered,
			geomState.point_offsets + P - 1,
			sizeof(int),
			cudaMemcpyDeviceToHost
		),
		debug
	);

	// required<BinningState>(num_rendered) 是一个编译时/运行时帮助函数, 用来计算如果要存下 num_rendered 个渲染操作的 BinningState 需要的字节大小
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	// 在 binningBuffer 中申请 binning_chunk_size 大小的内存, 并返回指向这块内存的指针, 用来存放 num_rendered 个渲染操作的 BinningState
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	// 简单理解就行, fromChunk 是一个静态方法, 它接受一个原始内存指针 binning_chunkptr 和渲染操作数量 num_rendered, 然后返回一个把原始内存结构化好的 BinningState 对象
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 并且处理 P 个高斯点, 去找出每个高斯点影响到的 tile, 然后统计下信息
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,											// 高斯点数量
		geomState.means2D,							// [P 2] 高斯点的 2D 坐标
		geomState.depths,							// [P] 高斯点的 2D 投影深度
		geomState.point_offsets,					// [P] 前缀和数组，指示每个高斯点实例化自己影响到的 tile 时的写入起始偏移
		binningState.point_list_keys_unsorted,				// 输出: [num_rendered]  tile ID(前 32 位) + depth(后 32 位)
		binningState.point_list_unsorted,					// 输出: [num_rendered]  影响这个 tile 的高斯点的编号
		radii,										// [P] 每个高斯点投影半径
		tile_grid									// (gridX,gridY,1)——屏幕被划分成多少个 tile
	)
	CHECK_CUDA(, debug)

	// 下面这段代码就是对前面的 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 进行基数排序
	// 让它们最终按照 "tile 从小到大, 同一 tile 内再按 depth 从小到大" 的顺序排列
	// 结果存到 binningState.point_list_keys 和 binningState.point_list 里
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 把 imgState.ranges 这个数组清零, 这个数组的作用是存储每个 tile 的高斯点的范围
	// 这里作者写的比较令人迷惑, 因为 imgState 定义的时候是以像素为单位定义的, 但是为什么这里又用来存储 tile 的涉及范围了呢?
	// 其实是这样的, imgState 里有三块同样大小为 width * height 的内存区域, 分别是 accum_alpha, n_contrib 和 ranges
		// accu_alpha 和 n_contrib 是用来存储每个像素的 alpha 和贡献值的
		// ranges (uint2 数组) 用来存储 tile 的涉及范围, 也就是说只有 tile_grid 的内存区域会被用到, 剩下的不会用到
		// 所以说把 ranges 记录在 imgState 只是为了方便, 其实是可以单独开辟一块内存的
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	// Identify start and end of per-tile workloads in sorted list
	// imgState.ranges[t] 中的 t 表示 tildID
	// 前面的 binningState.point_list_keys 不是已经排好序, 存储了 (tildeID, depth) 的键值对了吗?
	// imgState.ranges[t].x 表示 t 这个 tile 在 point_list_keys 中的起始位置
	// imgState.ranges[t].y 表示 t 这个 tile 在 point_list_keys 中的结束位置+1
	// 比如 point_list_keys = [(0, 1), (0, 2), (0, 3), (1, 0)], 那么 imgState.ranges[0].x = 0, imgState.ranges[0].y = 3 (左闭右开)
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
		);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
			tile_grid,					// (gridX,gridY,1)——屏幕被划分成多少个 tile
			block,						// (16,16,1)——每个 tile 里有多少个 thread
			imgState.ranges,			// [tileID 2] tile 的涉及范围, 值是 point_list_keys 中的索引
			binningState.point_list,	// 对应着 point_list_keys 中负责的高斯投影圆心的索引
			width,						// 图像宽度
			height,						// 图像高度
			geomState.means2D,			// [P 2] 高斯投影圆心的 2D 坐标
			feature_ptr,				// [P 3] 高斯投影圆心的 RGB 颜色
			geomState.conic_opacity,	// 高斯投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
			imgState.accum_alpha,				// 每个 pixel 的剩余透射率 (要往里填东西)
			imgState.n_contrib,					// 实际影响到该 pixel 的高斯体实例数量 (要往里填东西)
			background,					// 背景颜色
			out_color,							// 渲染图像 (要往里填东西)
			geomState.depths,			// [P] 高斯投影深度
			depth,								// 反深度图 (要往里填东西)
			gauss_sum,
			gauss_count,
			last_contr_gauss,
			out_accum_alpha
		),
		debug
	)

	// 渲染动作的总次数
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	// 基本尺寸参数
	const int P,				// 高斯点数量
	const int D,				// sh 的阶数
	const int M,				// sh 系数的数量
	const int R,				// 前向所有 (tileID, depth) pairs 的个数

	// 前向需要的常量输入 (只读)
	const float* background,			// 背景颜色
	const int width,					// 图像宽度
	const int height,					// 图像高度
	const float* means3D,				// 高斯点的 3D 坐标 [P 3]
	const float* shs,					// sh 系数 [P M D]
	const float* colors_precomp,		// 预先计算好的 RGB 颜色 (若有)
	const float* opacities,				// 所有高斯体的不透明度
	const float* scales,				// 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const float scale_modifier,			// 控制高斯体们的尺寸, 缩放因子
	const float* rotations,				// 每个高斯体的旋转变量
	const float* cov3D_precomp,			// 预先计算好的协方差矩阵 (若有)
	const float* viewmatrix,			// 视图矩阵
	const float* projmatrix,			// 投影矩阵
	const float* campos,				// 相机在世界里的坐标
	const float tan_fovx,				// 单位深度处的半宽度
	const float tan_fovy,				// 单位深度处的半高度

	// 前向算出的中间量
	const int* radii,				// [P] 每个高斯点投影半径
	char* geom_buffer,				// geomBuffer
	char* binning_buffer,			// binningBuffer
	char* img_buffer,				// imageBuffer

	// 从 Python 端传回的 loss 对 forward 输出的梯度
	const float* dL_dpix,			// loss 对渲染 RGB 图像的梯度 [3 H W]
	const float* dL_invdepths,		// loss 对反深度图的梯度 [1 H W]

	// 需要往里写入值的梯度
	float* dL_dmean2D,			// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	float* dL_dconic,			// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
	float* dL_dopacity,			// loss 对高斯点不透明度的梯度 [P 1]
	float* dL_dcolor,			// loss 对高斯点 RGB 颜色的梯度 [P 3]
	float* dL_dinvdepth,		// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	float* dL_dmean3D,			// loss 对高斯点 3D 坐标的梯度 [P 3]
	float* dL_dcov3D,			// loss 对预先计算好的协方差的矩阵 (若有) 的梯度 [P 6]
	float* dL_dsh,				// loss 对 sh 系数的梯度 [P M D]
	float* dL_dscale,			// loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
	float* dL_drot,				// loss 对每个高斯体的旋转变量的梯度 [P 4]

	// 额外开关
	bool antialiasing,	// 是否开启抗锯齿
	bool debug			// 是否开启 debug 模式
) {
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);			// 管理高斯点信息的地方
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);		// 放 (tildeID, depth), 对应高斯idx 的地方
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);	// 放像素信息的地方 (以及每个 tile 负责的 pairs 的范围)

	if (radii == nullptr) {
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);	// y 轴焦距
	const float focal_x = width / (2.0f * tan_fovx);	// x 轴焦距

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);		// 一共需要的 block 数量 (tile 数量)
	const dim3 block(BLOCK_X, BLOCK_Y, 1);	// 一个 block 有 16x16 个 thread

	// 下面这个函数就是把 dL_dpix, dL_invdepths 反传到几个输出身上, 过程需要用到一些前向过程已经计算出来的量
	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
			tile_grid,					// (gridX,gridY,1)——屏幕被划分成多少个 tile
			block,						// (16,16,1)——每个 tile 里有多少个 thread
			imgState.ranges,			// [tileID 2] tile 负责的 pairs 范围, 值是 point_list_keys 中的索引 (左闭右开)
			binningState.point_list,	// 对应着 point_list_keys 中高斯体的索引 idx
			width,						// 图像宽度
			height,						// 图像高度
			background,					// 背景颜色
			geomState.means2D,			// [P 2] 高斯投影圆心的 2D 坐标
			geomState.conic_opacity,	// 高斯投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
			color_ptr,					// [P 3] 高斯投影圆心的 RGB 颜色
			geomState.depths,			// [P] 高斯投影深度
			imgState.accum_alpha,		// [P] 每个 pixel 的剩余透射率
			imgState.n_contrib,			// 实际影响到该 pixel 的高斯体实例数量, 换句话说, 该像素光线上穿过的高斯体数量
			dL_dpix,					// loss 对渲染 RGB 图像的梯度 [3 H W]
			dL_invdepths,				// loss 对反深度图的梯度 [1 H W]
			(float3*)dL_dmean2D,				// 输出: loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
			(float4*)dL_dconic,					// 输出: loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
			dL_dopacity,						// 输出: loss 对高斯点不透明度的梯度 [P 1]
			dL_dcolor,							// 输出: loss 对高斯点 RGB 颜色的梯度 [P 3]
			dL_dinvdepth						// 输出: loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
		),
		debug
	);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(
			P,							// 高斯点数量
			D,							// sh 的阶数
			M,							// sh 系数的数量
			(float3*)means3D,			// [P 3] 高斯点的 3D 坐标 [P 3]
			radii,						// [P] 每个高斯点投影半径
			shs,						// [P M D] sh 系数
			geomState.clamped,			// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位
			opacities,					// 所有高斯体的不透明度
			(glm::vec3*)scales,			// [P 3] 每个高斯体的尺度 (在 xyz 轴的缩放长度)
			(glm::vec4*)rotations,		// [P 4] 每个高斯体的旋转变量
			scale_modifier,				// 控制高斯体们的尺寸, 缩放因子
			cov3D_ptr,					// 3D 协方差矩阵
			viewmatrix,					// 视图矩阵
			projmatrix,					// 投影矩阵	
			focal_x,					// x 轴焦距
			focal_y,					// y 轴焦距
			tan_fovx,					// 单位深度处的半宽度
			tan_fovy,					// 单位深度处的半高度
			(glm::vec3*)campos,			// 相机在世界里的坐标
			(float3*)dL_dmean2D,		// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
			dL_dconic,					// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
			dL_dinvdepth,				// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
			dL_dopacity,				// loss 对高斯点不透明度的梯度 [P 1]
			(glm::vec3*)dL_dmean3D,				// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
			dL_dcolor,					// loss 对高斯点 RGB 颜色的梯度 [P 3]
			dL_dcov3D,							// 输出: loss 对高斯体 3D 协方差的梯度 [P 6]
			dL_dsh,								// 输出: loss 对 sh 系数的梯度 [P M D]
			(glm::vec3*)dL_dscale,				// 输出: loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
			(glm::vec4*)dL_drot,				// 输出: loss 对每个高斯体的旋转变量的梯度 [P 4]
			antialiasing	// 是否开启抗锯齿
		),
		debug
	);
}
