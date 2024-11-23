#ifndef CUDA_PROCESSING_H
#define CUDA_PROCESSING_H

#include "types.h"
#include <cuda_runtime.h>
#include <utility>

pointcloud downsamplePointCloud(const pointcloud &input,
                                float downsample_factor);
pointcloud farthestPointSampling(const pointcloud &input, float ratio,
                                 bool random_start);
std::pair<std::vector<int64_t>, std::vector<int64_t>> radius(const pointcloud &x_cloud, const pointcloud &y_cloud,
                                                             float radius, int max_num_neighbors);
#endif // CUDA_PROCESSING_H