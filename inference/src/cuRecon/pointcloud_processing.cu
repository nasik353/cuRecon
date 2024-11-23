#include "cuRecon/pointcloud_processing.h"
#include "cuRecon/types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS 256 
#define WARP_SIZE 32

__forceinline__ __device__ int64_t get_example_idx(int64_t idx,
                                                   const int64_t *ptr,
                                                   const int64_t num_examples) {
  for (int64_t i = 0; i < num_examples; i++) {
    if (ptr[i + 1] > idx)
      return i;
  }
  return num_examples - 1;
}

__global__ void downsamplePointCloudKernel(const point *input, point *output,
                                           int input_size,
                                           float downsample_factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int output_size = static_cast<int>(input_size / downsample_factor);

  if (idx < output_size) {
    int input_idx = static_cast<int>(idx * downsample_factor);
    output[idx] = input[input_idx];
  }
}

pointcloud downsamplePointCloud(const pointcloud &input,
                                float downsample_factor) {
  int input_size = input.data.size();
  int output_size = static_cast<int>(input_size / downsample_factor);

  point *d_input;
  point *d_output;
  cudaMalloc(&d_input, input_size * sizeof(point));
  cudaMalloc(&d_output, output_size * sizeof(point));

  cudaMemcpy(d_input, input.data.data(), input_size * sizeof(point),
             cudaMemcpyHostToDevice);

  int blocks = (output_size + THREADS - 1) / THREADS;

  downsamplePointCloudKernel<<<blocks, THREADS>>>(d_input, d_output, input_size,
                                                  downsample_factor);
  cudaDeviceSynchronize();

  pointcloud output;
  output.data.resize(output_size);

  cudaMemcpy(output.data.data(), d_output, output_size * sizeof(point),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}

__global__ void
fps_kernel(const point *__restrict__ src, const int *__restrict__ ptr,
           const int *__restrict__ out_ptr, const int *__restrict__ start,
           float *__restrict__ dist, int *__restrict__ out, int dim) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int start_idx = ptr[bid];
  const int end_idx = ptr[bid + 1];

  __shared__ float best_dist[THREADS];
  __shared__ int best_idx[THREADS];

  if (tid == 0) {
    out[out_ptr[bid]] = start_idx + start[bid];
  }

  for (int m = out_ptr[bid] + 1; m < out_ptr[bid + 1]; m++) {
    __syncthreads();
    int prev_idx = out[m - 1]; 

    float max_dist = -1.0f; 
    int max_idx = 0;

    for (int n = start_idx + tid; n < end_idx; n += THREADS) {
      float dx = src[prev_idx].x - src[n].x;
      float dy = src[prev_idx].y - src[n].y;
      float dz = src[prev_idx].z - src[n].z;
      float dist_to_prev = dx * dx + dy * dy + dz * dz;

      if (dist_to_prev < dist[n]) {
        dist[n] = dist_to_prev;
      }

      if (dist[n] > max_dist) {
        max_dist = dist[n];
        max_idx = n;
      }
    }

    best_dist[tid] = max_dist;
    best_idx[tid] = max_idx;
    __syncthreads();

    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        if (best_dist[tid + offset] > best_dist[tid]) {
          best_dist[tid] = best_dist[tid + offset];
          best_idx[tid] = best_idx[tid + offset];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      out[m] = best_idx[0];
    }
  }
}

pointcloud farthestPointSampling(const pointcloud &input, float ratio,
                                 bool random_start) {
  int input_size = input.data.size();
  int dim = 3;        
  int batch_size = 1;
  int output_size = static_cast<int>(input_size * ratio);

  point *d_input;
  int *d_ptr;
  int *d_out_ptr;
  int *d_start;
  float *d_dist;
  int *d_out;

  cudaMalloc(&d_input, input_size * sizeof(point));
  cudaMalloc(&d_ptr, (batch_size + 1) * sizeof(int));
  cudaMalloc(&d_out_ptr, (batch_size + 1) * sizeof(int));
  cudaMalloc(&d_start, batch_size * sizeof(int));
  cudaMalloc(&d_dist, input_size * sizeof(float));
  cudaMalloc(&d_out, output_size * sizeof(int));

  cudaMemcpy(d_input, input.data.data(), input_size * sizeof(point),
             cudaMemcpyHostToDevice);

  std::vector<int> ptr = {0, input_size};
  std::vector<int> out_ptr = {0, output_size};
  cudaMemcpy(d_ptr, ptr.data(), (batch_size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_ptr, out_ptr.data(), (batch_size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  std::vector<int> start(batch_size, 0);
  if (random_start) {
    for (int i = 0; i < batch_size; i++) {
      start[i] = rand() % (ptr[i + 1] - ptr[i]);
    }
  }
  cudaMemcpy(d_start, start.data(), batch_size * sizeof(int),
             cudaMemcpyHostToDevice);

  std::vector<float> dist(input_size,
                          1e10f); 
  cudaMemcpy(d_dist, dist.data(), input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  fps_kernel<<<batch_size, THREADS>>>(d_input, d_ptr, d_out_ptr, d_start,
                                      d_dist, d_out, dim);
  cudaDeviceSynchronize();

  std::vector<int> out_indices(output_size);
  cudaMemcpy(out_indices.data(), d_out, output_size * sizeof(int),
             cudaMemcpyDeviceToHost);

  pointcloud output;
  output.data.resize(output_size);
  for (int i = 0; i < output_size; i++) {
    output.data[i] = input.data[out_indices[i]];
  }

  cudaFree(d_input);
  cudaFree(d_ptr);
  cudaFree(d_out_ptr);
  cudaFree(d_start);
  cudaFree(d_dist);
  cudaFree(d_out);

  return output;
}

__global__ void radius_kernel(const point *__restrict__ x,
                              const point *__restrict__ y,
                              const int64_t *__restrict__ ptr_x,
                              const int64_t *__restrict__ ptr_y,
                              int64_t *__restrict__ row,
                              int64_t *__restrict__ col, const float r_squared,
                              const int n, const int m, const int num_examples,
                              const int max_num_neighbors) {

  const int n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  int count = 0;
  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    float dx = x[n_x].x - y[n_y].x;
    float dy = x[n_x].y - y[n_y].y;
    float dz = x[n_x].z - y[n_y].z;
    float dist = dx * dx + dy * dy + dz * dz;

    if (dist < r_squared) {
      row[n_y * max_num_neighbors + count] = n_y;
      col[n_y * max_num_neighbors + count] = n_x;
      count++;
    }

    if (count >= max_num_neighbors)
      break;
  }
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
radius(const pointcloud &x_cloud, const pointcloud &y_cloud, float radius,
       int max_num_neighbors) {
  int n = x_cloud.data.size();
  int m = y_cloud.data.size();

  std::vector<int64_t> ptr_x = {
      0, n};
  std::vector<int64_t> ptr_y = {
      0, m};

  int num_examples = ptr_x.size() - 1;

  std::vector<int64_t> row(m * max_num_neighbors, -1);
  std::vector<int64_t> col(m * max_num_neighbors, -1);

  point *d_x, *d_y;
  int64_t *d_ptr_x, *d_ptr_y, *d_row, *d_col;

  cudaMalloc(&d_x, n * sizeof(point));
  cudaMalloc(&d_y, m * sizeof(point));
  cudaMalloc(&d_ptr_x, ptr_x.size() * sizeof(int64_t));
  cudaMalloc(&d_ptr_y, ptr_y.size() * sizeof(int64_t));
  cudaMalloc(&d_row, row.size() * sizeof(int64_t));
  cudaMalloc(&d_col, col.size() * sizeof(int64_t));

  cudaMemcpy(d_x, x_cloud.data.data(), n * sizeof(point),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y_cloud.data.data(), m * sizeof(point),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr_x, ptr_x.data(), ptr_x.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr_y, ptr_y.data(), ptr_y.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  dim3 BLOCKS((m + THREADS - 1) / THREADS);
  radius_kernel<<<BLOCKS, THREADS>>>(d_x, d_y, d_ptr_x, d_ptr_y, d_row, d_col,
                                     radius * radius, n, m, num_examples,
                                     max_num_neighbors);
  cudaDeviceSynchronize();

  cudaMemcpy(row.data(), d_row, row.size() * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(col.data(), d_col, col.size() * sizeof(int64_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_ptr_x);
  cudaFree(d_ptr_y);
  cudaFree(d_row);
  cudaFree(d_col);

  std::vector<int64_t> valid_row, valid_col;
  for (size_t i = 0; i < row.size(); i++) {
    if (row[i] != -1) {
      valid_row.push_back(row[i]);
      valid_col.push_back(col[i]);
    }
  }

  return std::make_pair(valid_row, valid_col);
}