#include <cuda_runtime.h>

// ── Forward ──────────────────────────────────────────────────────────────────

__global__ void relu_forward_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

//  Z[row][col] += b[row]
__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        int row = i % rows;
        Z[i] += b[row];
    }
}

// ── Backward ─────────────────────────────────────────────────────────────────

// dZ[i] = (Z[i] > 0) ? dA[i] : 0     (ReLU backward using pre-activation Z)
__global__ void relu_backward_kernel(const float* Z, const float* dA, float* dZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dZ[i] = (Z[i] > 0.0f) ? dA[i] : 0.0f;
    }
}

// db[row] = sum over cols of dZ[row][col]
// dZ is row-major (rows x cols), db is (rows x 1)
// Each thread handles one row
__global__ void sum_cols_kernel(const float* dZ, float* db, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += dZ[row * cols + c];
        }
        db[row] = sum;
    }
}

// ── Wrappers llamados desde C++ ──────────────────────────────────────────────

extern "C" {

void launch_relu_forward(const float* in, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_forward_kernel<<<grid, block>>>(in, out, n);
}

void launch_relu_backward(const float* Z, const float* dA, float* dZ, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(Z, dA, dZ, n);
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    int n = rows * cols;
    int block = 256;
    int grid = (n + block - 1) / block;
    add_bias_kernel<<<grid, block>>>(Z, b, rows, cols);
}

void launch_sum_cols(const float* dZ, float* db, int rows, int cols) {
    int block = 256;
    int grid = (rows + block - 1) / block;
    sum_cols_kernel<<<grid, block>>>(dZ, db, rows, cols);
}

} // extern "C"