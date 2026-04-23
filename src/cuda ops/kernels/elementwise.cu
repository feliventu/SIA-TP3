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