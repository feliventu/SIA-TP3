#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../gpu_tensor.cuh"

// C = alpha * A . B + beta * C
void gpu_gemm(cublasHandle_t handle,
            const GpuTensor& A,
            const GpuTensor& B,
            GpuTensor& C,
            float alpha = 1.0f, float beta = 0.0f)
{
             
        cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C.rows, C.cols, A.cols,    // M, N, K
        &alpha,
        A.data, A.rows,            // A y su "leading dimension"
        B.data, B.rows,            // B
        &beta,
        C.data, C.rows);   


}


void gpu_gemm_tensor_cores(cublasHandle_t handle,
                            const GpuTensor& A,
                            const GpuTensor& B,
                            GpuTensor& C)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        C.rows, C.cols, A.cols,
        &alpha,
        A.data, CUDA_R_16F, A.rows,   // float16 input
        B.data, CUDA_R_16F, B.rows,   // float16 input
        &beta,
        C.data, CUDA_R_32F, C.rows,   // float32 output (no perdés precisión)
        CUBLAS_COMPUTE_16F,            // activa Tensor Cores
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}