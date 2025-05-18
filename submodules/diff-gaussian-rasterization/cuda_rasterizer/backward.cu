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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(
	int P,							// 高斯体数量
	const float3* means,			// [P 3] 高斯体的 3D 坐标
	const int* radii,				// [P] 每个高斯点的投影半径
	const float* cov3Ds,			// 3D 协方差矩阵
	const float h_x,				// x 轴焦距
	const float h_y,				// y 轴焦距
	const float tan_fovx,			// 单位深度处的半宽度
	const float tan_fovy,			// 单位深度处的半高度
	const float* view_matrix,		// 视图矩阵
	const float* opacities,			// 高斯体的不透明度
	const float* dL_dconics,		// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
	float* dL_dopacity,				// loss 对高斯点不透明度的梯度 [P 1]
	const float* dL_dinvdepth,		// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	float3* dL_dmeans,						// 输出: loss 对高斯体 3D 坐标的梯度 [P 3]
	float* dL_dcov,							// 输出: loss 对高斯体 3D 协方差的梯度 [P 6]
	bool antialiasing
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 拿出当前高斯体的协方差矩阵 (6 个数)
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];	// 取出世界坐标 (3维)
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };	// 取出 2D 协方差逆矩阵
	float3 t = transformPoint4x3(mean, view_matrix);	// 得到当前高斯体的相机坐标 (3维)
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	// Account for inverse depth gradients
	if (dL_dinvdepth)
	dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,							// 高斯体数量
	int D,							// SH 阶数
	int M,							// SH 系数数量
	const float3* means,			// [P 3] 高斯体的 3D 坐标
	const int* radii,				// [P] 每个高斯点的投影半径
	const float* shs,				// [P M D] sh 系数
	const bool* clamped,			// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped
	const glm::vec3* scales,		// [P 3] 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const glm::vec4* rotations,		// [P 4] 每个高斯体的旋转变量
	const float scale_modifier,		// 控制高斯体们的尺寸, 缩放因子
	const float* proj,				// 投影矩阵
	const glm::vec3* campos,		// 相机在世界里的坐标
	const float3* dL_dmean2D,		// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	glm::vec3* dL_dmeans,					// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
	float* dL_dcolor,				// loss 对高斯点 RGB 颜色的梯度 [P 3]
	float* dL_dcov3D,				// loss 对高斯体 3D 协方差的梯度 [P 6]
	float* dL_dsh,							// 输出: loss 对 sh 系数的梯度 [P M D]
	glm::vec3* dL_dscale,					// 输出: loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
	glm::vec4* dL_drot,						// 输出: loss 对每个高斯体的旋转变量的梯度 [P 4]
	float* dL_dopacity				// loss 对高斯点不透明度的梯度 [P 1]
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 取出高斯体的世界坐标 (3 维)
	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	// 拿到未归一化的 ndc 坐标
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);	// 归一化系数

	// 下面这段就是把 dL_dmean2D 梯度反传到 dL_dmeans
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;
	dL_dmeans[idx] += dL_dmean;

	// 下面这句就是把 dL_dcolor 梯度反传到 dL_dsh, dL_dmeans
	if (shs)
		computeColorFromSH(
			idx,					// 当前高斯体的索引
			D,						// SH 阶数
			M,						// SH 系数数量
			(glm::vec3*)means,		// 高斯体的 3D 坐标
			*campos,				// 相机在世界里的坐标
			shs,					// [P M D] sh 系数
			clamped,				// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped
			(glm::vec3*)dL_dcolor,	// loss 对高斯点 RGB 颜色的梯度 [P 3]
			(glm::vec3*)dL_dmeans,			// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
			(glm::vec3*)dL_dsh				// 输出: loss 对 sh 系数的梯度 [P M D]
		);

	// 下面这句就是把 dL_dcov3D 梯度反传到 dL_dscale, dL_drot
	if (scales)
		computeCov3D(
			idx,					// 当前高斯体的索引
			scales[idx],			// [3] 当前高斯体的尺度 (在 xyz 轴的缩放长度)
			scale_modifier,			// 控制高斯体们的尺寸, 缩放因子
			rotations[idx],			// [4] 当前高斯体的旋转变量
			dL_dcov3D,				// loss 对高斯体 3D 协方差的梯度 [P 6]
			dL_dscale,						// 输出: loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
			dL_drot							// 输出: loss 对每个高斯体的旋转变量的梯度 [P 4]
		);
}

// Backward version of the rendering procedure.
template <uint32_t C, uint32_t DIM>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,				// [tileID 2] tile 负责的 pairs 范围, 值是 point_list_keys 中的索引 (左闭右开)
	const uint32_t* __restrict__ point_list,		// 对应着 point_list_keys 中高斯体的索引 idx
	int W,											// 图像宽度
	int H,											// 图像高度
	const float* __restrict__ bg_color,				// 背景颜色
	const float2* __restrict__ points_xy_image,		// [P 2] 高斯投影圆心的 2D 坐标
	const float4* __restrict__ conic_opacity,		// 高斯投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
	const float* __restrict__ colors,				// [P 3] 高斯体投影圆心的 RGB 颜色
	const float* __restrict__ depths,				// [P] 高斯投影深度
	const float* __restrict__ final_Ts,				// [P] 每个 pixel 的剩余透射率
	const uint32_t* __restrict__ n_contrib,			// [P] 该像素光线上穿过的高斯体数量
	const float* __restrict__ dL_dpixels,			// loss 对渲染 RGB 图像的梯度 [3 H W]
	const float* __restrict__ dL_invdepths,			// loss 对反深度图的梯度 [1 H W]
	const float* __restrict__ dL_reps,				// loss 对自定义特征的梯度 [NUM_DIM H W]
	float3* __restrict__ dL_dmean2D,						// 输出: loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	float4* __restrict__ dL_dconic2D,						// 输出: loss 对高斯点 2D 协方差矩阵的梯度 [P 2 2]
	float* __restrict__ dL_dopacity,						// 输出: loss 对高斯体不透明度的梯度 [P 1]
	float* __restrict__ dL_dcolors,							// 输出: loss 对高斯点 RGB 颜色的梯度 [P 3]
	float* __restrict__ dL_dinvdepths,						// 输出: loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	float* __restrict__ dL_dreps,							// 输出: loss 对自定义特征的梯度 [P NUM_DIM]
	const float* __restrict__ regs					// [P M] 高斯点的自定义特征
) {
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// 当前 thread 对应的 pixel 的二维坐标
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 当前 thread 的一维坐标
	const uint32_t pix_id = W * pix.y + pix.x;
	// 当前 thread 对应的 pixel 的二维坐标 (浮点)
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	
	// 取出这个 pixel 所在的 tile 负责的 pairs 的范围
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	// 要做的高斯体数量
	int toDo = range.y - range.x;

	// 如果一次做 BLOCK_SIZE 个高斯体, 那么做完负责的高斯体需要的轮数
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// 下面就是 share-memory, 用处就是每次 256 个 pixel 会把收集到的数据填到 share-memory 里面
	__shared__ int collected_id[BLOCK_SIZE];				// 高斯体 idx
	__shared__ float2 collected_xy[BLOCK_SIZE];				// 高斯体投影圆心的 2D 坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];	// 高斯体投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
	__shared__ float collected_colors[C * BLOCK_SIZE];		// 高斯体投影圆心的 RGB 颜色
	__shared__ float collected_regs[DIM * BLOCK_SIZE];		// 高斯点的自定义特征
	__shared__ float collected_depths[BLOCK_SIZE];			// 高斯体投影深度

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;	// 拿到当前像素的剩余透光率

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;									// 当前 pixel 要处理的高斯体数量
	const int last_contributor = inside ? n_contrib[pix_id] : 0;	// 对当前 pixel 产生贡献的最后一个高斯体编号 (编号从 1 开始)

	float dL_dpixel[C];			// 用来存 loss 对当前 pixel RGB 颜色的梯度 [3]
	float dL_rep[DIM];			// 用来存 loss 对当前 pixel 自定义特征的梯度 [NUM_DIM]
	float dL_invdepth;			// 用来存 loss 对当前 pixel 反深度的梯度 [1]
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		for (int i = 0; i < DIM; i++)
			dL_rep[i] = dL_reps[i * H * W + pix_id];
		if(dL_invdepths)
			dL_invdepth = dL_invdepths[pix_id];
	}
	float accum_rec[C] = { 0 };		// 一个过程量, 后续算梯度会用到
	float accum_rep[DIM] = { 0 };	// 一个过程量, 后续算梯度会用到
	float accum_invdepth_rec = 0;	// 一个过程量, 后续算梯度会用到

	// 下面这三个量, 是在反向遍历的时候, 用来保存上一个被处理到的高斯体的一些属性
	float last_alpha = 0;			// 上一个高斯体的 alpha
	float last_color[C] = { 0 };	// 上一个高斯体的颜色
	float last_rep[DIM] = { 0 };	// 上一个高斯体的自定义特征
	float last_invdepth = 0;		// 上一个高斯体的反深度


	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// 因为: x_{pixel} = (x_{ndc} x 0.5 + 0.5) * W
	// 所以: x_{pixel} 对 x_{ndc} 的导数 = 0.5 * W
	// 所以下面是存了俩导数
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();		// 进度: 表示处理到第几个高斯体实例了
		if (range.x + progress < range.y) {		// 如果这个 pixel 负责的高斯体实例在该 tile 要处理的高斯体实例范围内
			const int coll_id = point_list[range.y - progress - 1];		// 倒着拿
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			for (int i = 0; i < DIM; i++)
				collected_regs[i * BLOCK_SIZE + block.thread_rank()] = regs[coll_id * DIM + i];
			if(dL_invdepths)
				collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
			contributor--;	// contributor-- 对应着 forward 相同地方的 contributor++
			// 只处理真正最该 pixel 有贡献的高斯体实例
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			// power 就是二位高斯表达式中 e 头上上个系数
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// 按理说 power 绝对是 <= 0 的, 这里是个保险性检查
			if (power > 0.0f)
				continue;

			const float G = exp(power);	// 高斯概率值
			const float alpha = min(0.99f, con_o.w * G);  // 当前高斯体在此 pixel 上的不透明度 (= 自己的不透明度 * 高斯概率值)
			// 那些对于该 pixel 的不透明影响度过小的高斯直接 continue (它们在前向过程中也会被 continue 的)
			if (alpha < 1.0f / 255.0f)
				continue;

			// ------------ 计算 dL/dc_{now}, dL/dinvdepth_{now}, dL/dα_{now}, dL/dopacity_{now} --------------
			// 算出 T_{t-1}
			T = T / (1.f - alpha);
			// dC/dc_{now} = αT_{t-1}
			const float dchannel_dcolor = alpha * T;

			// 算 dL/dc_{now} 和 dL/dα_{now}
			const int global_id = collected_id[j]; // 当前高斯体的 idx
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++) {
				const float dL_dchannel = dL_dpixel[ch];

				// 算 dL/dα_{now}
				const float c = collected_colors[ch * BLOCK_SIZE + j];	// 当前高斯体的颜色 c
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

				// 算 dL/dc_{now}
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			for (int ch = 0; ch < DIM; ch++) {
				const float dL_dchannel = dL_rep[ch];

				// 算 dL/dα_{now}
				const float c = collected_regs[ch * BLOCK_SIZE + j];	// 当前高斯体的自定义特征 c
				accum_rep[ch] = last_alpha * last_rep[ch] + (1.f - last_alpha) * accum_rep[ch];
				last_rep[ch] = c;
				dL_dalpha += (c - accum_rep[ch]) * dL_dchannel;

				// 算 dL/dc_{now}
				atomicAdd(&(dL_dreps[global_id * DIM + ch]), dchannel_dcolor * dL_dchannel);
			}

			// 算 dL/dinvdepth_{now} 和 dL/dα_{now}
			if (dL_dinvdepths) {
				// 算 dL/dα_{now}
				const float invd = 1.f / collected_depths[j];
				accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
				last_invdepth = invd;
				dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;

				// 算 dL/dinvdepth_{now}
				atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}

			dL_dalpha *= T;
			last_alpha = alpha;

			// 算 dL/dα_{now}
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// 算 dL/dopacity_{now}
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			// ------------ 计算 dL_dmean2D, dL_dconic2D --------------
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);	// 这里的 0.5 不是 bug, 原因见: https://github.com/graphdeco-inria/gaussian-splatting/issues/1096
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
		}
	}
}

void BACKWARD::preprocess(
	int P,							// 高斯体数量
	int D,							// SH 阶数
	int M,							// SH 系数数量
	const float3* means3D,			// [P 3] 高斯体的 3D 坐标
	const int* radii,				// [P] 每个高斯点的投影半径
	const float* shs,				// [P M D] sh 系数
	const bool* clamped,			// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位
	const float* opacities,			// 高斯体的不透明度
	const glm::vec3* scales,		// [P 3] 每个高斯体的尺度 (在 xyz 轴的缩放长度)
	const glm::vec4* rotations,		// [P 4] 每个高斯体的旋转变量
	const float scale_modifier,		// 控制高斯体们的尺寸, 缩放因子
	const float* cov3Ds,			// 3D 协方差矩阵
	const float* viewmatrix,		// 视图矩阵
	const float* projmatrix,		// 投影矩阵
	const float focal_x,			// x 轴焦距
	const float focal_y,			// y 轴焦距
	const float tan_fovx,			// 单位深度处的半宽度
	const float tan_fovy,			// 单位深度处的半高度
	const glm::vec3* campos,		// 相机在世界里的坐标
	const float3* dL_dmean2D,		// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	const float* dL_dconic,			// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
	const float* dL_dinvdepth,		// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	float* dL_dopacity,				// loss 对高斯点不透明度的梯度 [P 1]
	glm::vec3* dL_dmean3D,					// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
	float* dL_dcolor,				// loss 对高斯点 RGB 颜色的梯度 [P 3]
	float* dL_dcov3D,						// 输出: loss 对高斯体 3D 协方差的梯度 [P 6]
	float* dL_dsh,							// 输出: loss 对 sh 系数的梯度 [P M D]
	glm::vec3* dL_dscale,					// 输出: loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
	glm::vec4* dL_drot,						// 输出: loss 对每个高斯体的旋转变量的梯度 [P 4]
	bool antialiasing	// 是否开启抗锯齿
) {
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// P 个高斯体并行算
	// 利用 dL_dconic, dL_dopacity, dL_dinvdepth, 梯度反传到 dL_dmean3D, dL_dconv3D
	// 为什么要用 dL_dinvdepth 呢? 因为深度 view.z 是经过视图矩阵变换得到的
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,						// 高斯体数量
		means3D,				// [P 3] 高斯体的 3D 坐标
		radii,					// [P] 每个高斯点的投影半径
		cov3Ds,					// 3D 协方差矩阵
		focal_x,				// x 轴焦距
		focal_y,				// y 轴焦距
		tan_fovx,				// 单位深度处的半宽度
		tan_fovy,				// 单位深度处的半高度
		viewmatrix,				// 视图矩阵
		opacities,				// 高斯体的不透明度
		dL_dconic,				// loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
		dL_dopacity,			// loss 对高斯点不透明度的梯度 [P 1]
		dL_dinvdepth,			// loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
		(float3*)dL_dmean3D,			// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
		dL_dcov3D,						// 输出: loss 对高斯体 3D 协方差的梯度 [P 6]
		antialiasing
	);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	// P 个高斯体并行算
	// 利用 dL_dmean2D, dL_dcolor, dL_dconv3D, 梯度反传到 dL_dmean3D, dL_dsh, dL_dscale, dL_drot
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,						// 高斯体数量
		D,						// SH 阶数
		M,						// SH 系数数量
		(float3*)means3D,		// [P 3] 高斯体的 3D 坐标
		radii,					// [P] 每个高斯点的投影半径
		shs,					// [P M D] sh 系数
		clamped,				// [P 3] 每个高斯点的 R/G/B 通道的值是否被 clamped 的标志位
		(glm::vec3*)scales,		// [P 3] 每个高斯体的尺度 (在 xyz 轴的缩放长度)
		(glm::vec4*)rotations,	// [P 4] 每个高斯体的旋转变量
		scale_modifier,			// 控制高斯体们的尺寸, 缩放因子
		projmatrix,				// 投影矩阵
		campos,					// 相机在世界里的坐标
		(float3*)dL_dmean2D,	// loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
		(glm::vec3*)dL_dmean3D,			// 输出: loss 对高斯点 3D 坐标的梯度 [P 3]
		dL_dcolor,				// loss 对高斯点 RGB 颜色的梯度 [P 3]
		dL_dcov3D,				// loss 对高斯体 3D 协方差的梯度 [P 6]
		dL_dsh,							// 输出: loss 对 sh 系数的梯度 [P M D]
		dL_dscale,						// 输出: loss 对每个高斯体的尺度 (在 xyz 轴的缩放长度) 的梯度 [P 3]
		dL_drot,						// 输出: loss 对每个高斯体的旋转变量的梯度 [P 4]
		dL_dopacity				// loss 对高斯点不透明度的梯度 [P 1]
	);
}

void BACKWARD::render(
	const dim3 grid,				// (gridX,gridY,1)——屏幕被划分成多少个 tile
	const dim3 block,				// (16,16,1)——每个 tile 里有多少个 thread
	const uint2* ranges,			// [tileID 2] tile 负责的 pairs 范围, 值是 point_list_keys 中的索引 (左闭右开)
	const uint32_t* point_list,		// 对应着 point_list_keys 中高斯体的索引 idx
	int W,							// 图像宽度
	int H,							// 图像高度
	const float* bg_color,			// 背景颜色
	const float2* means2D,			// [P 2] 高斯投影圆心的 2D 坐标
	const float4* conic_opacity,	// 高斯投影椭圆的 2D 协方差矩阵的逆矩阵 + 3D 高斯体的不透明度
	const float* colors,			// [P 3] 高斯投影圆心的 RGB 颜色
	const float* depths,			// [P] 高斯投影深度
	const float* final_Ts,			// [P] 每个 pixel 的剩余透射率
	const uint32_t* n_contrib,		// [P] 该像素光线上穿过的高斯体数量
	const float* dL_dpixels,		// loss 对渲染 RGB 图像的梯度 [3 H W]
	const float* dL_invdepths,		// loss 对反深度图的梯度 [1 H W]
	const float* dL_reps,			// loss 对自定义特征的梯度 [NUM_DIM H W]
	float3* dL_dmean2D,						// 输出: loss 对 ndc 空间的高斯点 2D 坐标的梯度 [P 2]
	float4* dL_dconic2D,					// 输出: loss 对高斯点 2D 协方差逆矩阵的梯度 [P 2 2]
	float* dL_dopacity,						// 输出: loss 对高斯点不透明度的梯度 [P 1]
	float* dL_dcolors,						// 输出: loss 对高斯点 RGB 颜色的梯度 [P 3]
	float* dL_dinvdepths,					// 输出: loss 对每个高斯体投影深度 view.z 的梯度 [P 1]
	float* dL_dreps,						// 输出: loss 对自定义特征的梯度 [P NUM_DIM]
	const float* reps				// [NUM_DIM H W] 自定义特征
) {
	// 每个像素并行去做
	renderCUDA<NUM_CHANNELS, NUM_DIM> << <grid, block >> >(
		ranges,
		point_list,
		W,
		H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_reps,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths,
		dL_dreps,
		reps
	);
}
