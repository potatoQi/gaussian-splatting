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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);                  // 前向: rasterize_points.cu 里的 RasterizeGaussiansCUDA 函数
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA); // 反向: rasterize_points.cu 里的 RasterizeGaussiansBackwardCUDA 函数
  m.def("mark_visible", &markVisible);
}