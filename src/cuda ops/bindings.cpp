#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "gpu_tensor.cuh"

namespace py = pybind11;

// Handle persistente (evita crear/destruir en cada llamada)
static cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (g_handle == nullptr) {
        cublasCreate(&g_handle);
    }
    return g_handle;
}

/*
 * Matmul para matrices ROW-MAJOR (como las de NumPy).
 *
 * cuBLAS espera column-major. El truco es que si A y B están en row-major,
 * cuBLAS las ve como A^T y B^T en column-major.
 * Queremos C = A·B (row-major).
 * En column-major eso equivale a: C^T = B^T · A^T
 * Entonces llamamos: cublas(..., N, M, K, B, N, A, K, C, N)
 */
py::array_t<float> matmul(py::array_t<float> A, py::array_t<float> B, bool use_tensor_cores) {
    auto a = A.request();
    auto b = B.request();
    
    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    
    GpuTensor d_A(M, K), d_B(K, N), d_C(M, N);
    
    // CPU -> GPU
    d_A.upload((float*)a.ptr);
    d_B.upload((float*)b.ptr);
    
    cublasHandle_t handle = get_handle();
    float alpha = 1.0f, beta = 0.0f;
    
    if (use_tensor_cores) {
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,              // dimensiones invertidas (row-major trick)
            &alpha,
            d_B.data, CUDA_R_32F, N,   // B primero (era segundo)
            d_A.data, CUDA_R_32F, K,   // A segundo (era primero)
            &beta,
            d_C.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,              // dimensiones invertidas
            &alpha,
            d_B.data, N,          // B primero
            d_A.data, K,          // A segundo
            &beta,
            d_C.data, N);
    }
    
    // GPU -> CPU
    auto result = py::array_t<float>({M, N});
    d_C.download((float*)result.request().ptr);
    
    return result;
}

extern void gpu_augment(const float *d_src_img, float *d_dst_img, 
                 int m, float alpha, float sigma, float rot_range, float scale_range);

py::array_t<float> augment_images(py::array_t<float> images, float alpha, float sigma, float rot, float scale) {
    auto buf = images.request();
    
    // Expecting shape (m, 784), C-contiguous
    int m = buf.shape[0];
    int features = buf.shape[1];
    
    if (features != 784) {
        throw std::runtime_error("Las imágenes deben tener 784 features (28x28).");
    }

    GpuTensor d_src(m, features);
    GpuTensor d_dst(m, features);
    
    d_src.upload((float*)buf.ptr);
    
    gpu_augment(d_src.data, d_dst.data, m, alpha, sigma, rot, scale);
    
    auto result = py::array_t<float>({m, features});
    d_dst.download((float*)result.request().ptr);
    
    return result;
}

PYBIND11_MODULE(cuda_ops, m_py) {
    m_py.def("matmul", &matmul, "GPU matrix multiply (row-major compatible)",
          py::arg("A"), py::arg("B"), py::arg("use_tensor_cores") = false);
    
    m_py.def("augment_images", &augment_images, "Deforma imágenes en GPU (Afín + Elástica)",
          py::arg("images"), py::arg("alpha"), py::arg("sigma"), py::arg("rot"), py::arg("scale"));
}

