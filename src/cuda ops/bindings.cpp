#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <stdexcept>

#include "gpu_tensor.cuh"

namespace py = pybind11;

// ── Kernel launchers (definidos en elementwise.cu) ───────────────────────────
extern "C" {
    void launch_relu_forward(const float* in, float* out, int n);
    void launch_relu_backward(const float* Z, const float* dA, float* dZ, int n);
    void launch_add_bias(float* Z, const float* b, int rows, int cols);
    void launch_sum_cols(const float* dZ, float* db, int rows, int cols);
}

// ── Augmentation (definido en augmentation.cu) ──────────────────────────────
extern void gpu_augment(const float *d_src_img, float *d_dst_img,
                 int m, float alpha, float sigma, float rot_range, float scale_range);

// ── Handle cuBLAS persistente ────────────────────────────────────────────────
static cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (g_handle == nullptr) {
        cublasCreate(&g_handle);
    }
    return g_handle;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Función matmul standalone (backward compatible — no se toca)
// ═══════════════════════════════════════════════════════════════════════════════

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
            N, M, K,
            &alpha,
            d_B.data, CUDA_R_32F, N,
            d_A.data, CUDA_R_32F, K,
            &beta,
            d_C.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B.data, N,
            d_A.data, K,
            &beta,
            d_C.data, N);
    }

    // GPU -> CPU
    auto result = py::array_t<float>({M, N});
    d_C.download((float*)result.request().ptr);

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MlpContext — forward y backward completos en GPU
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * Row-major GEMM helper.
 * Computes C(M,N) = A(M,K) · B(K,N)  in row-major.
 * 
 * Row-major trick: C^T = B^T · A^T (column-major).
 * cuBLAS sees row-major data as transposed column-major, so:
 *   cublas(N, N, M, N, K, B_data, N, A_data, K, C_data, N)
 */
static void gemm_nn(cublasHandle_t h, bool use_tc,
                     const float* A, int M, int K,
                     const float* B, int N,
                     float* C,
                     float alpha = 1.0f, float beta = 0.0f)
{
    if (use_tc) {
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, CUDA_R_32F, N,
            A, CUDA_R_32F, K,
            &beta,
            C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, N, A, K, &beta, C, N);
    }
}

/*
 * Row-major: C(M,N) = A(M,K) · B(N,K)^T
 * 
 * C^T(N,M) = B(N,K) · A^T(K,M)   [column-major view]
 * cublas sees row-major B as B_cm = B^T_cm(K,N).
 * We need B not transposed in row-major = B^T in col-major = need CUBLAS_OP_T on B_cm.
 * 
 * Actually, using the full derivation:
 *   C_rm^T = (A_rm · B_rm^T)^T = B_rm · A_rm^T
 *   C_cm = B_rm · A_rm^T
 *   cublas sees B_rm as B_cm=B_rm^T, A_rm as A_cm=A_rm^T
 *   C_cm = B_cm^T · A_cm = op_T(B_cm) · op_N(A_cm)
 *   cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
 *               alpha, A_cm=A_data, N_a, B_cm=B_data, N_b, beta, C_data, N)
 *
 * Wait — let me use the empirical approach matching the dimension conventions.
 * 
 * For dW(out,in) = dZ(out,batch) · A_prev(in,batch)^T:
 *   cublasSgemm(h, T, N, in, out, batch,
 *               alpha, A_prev_data, batch,    // cublas sees (batch,in), transposed=(in,batch)
 *               dZ_data, batch,               // cublas sees (batch,out)
 *               beta, dW_data, in)            // result (in,out) = dW^T
 */
static void gemm_nt(cublasHandle_t h, bool use_tc,
                     const float* A, int M, int K,   // A is (M,K)
                     const float* B, int N,           // B is (N,K) — will be transposed
                     float* C)                         // C is (M,N)
{
    // C_rm(M,N) = A_rm(M,K) · B_rm(N,K)^T
    // In col-major view: C_cm = B_cm^T · A_cm, where B_cm = B_rm^T(K,N), A_cm = A_rm^T(K,M)
    // C_cm(N,M): cublasSgemm(h, T, N, N, M, K, alpha, B_data, K, A_data, K, beta, C_data, N)
    float alpha = 1.0f, beta = 0.0f;
    if (use_tc) {
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, CUDA_R_32F, K,
            A, CUDA_R_32F, K,
            &beta,
            C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, K, A, K, &beta, C, N);
    }
}

/*
 * Row-major: C(M,N) = A(K,M)^T · B(K,N)
 * 
 * C_cm(N,M) = B_cm^... let me just derive:
 * C_rm^T = (A_rm^T · B_rm)^T = B_rm^T · A_rm = A_cm · B_cm^T... 
 * 
 * Simpler: cublasSgemm(h, N, T, N, M, K, alpha, B_data, N, A_data, M, beta, C_data, N)
 */
static void gemm_tn(cublasHandle_t h, bool use_tc,
                     const float* A, int K, int M,    // A is (K,M) — will be transposed to (M,K)
                     const float* B, int N,            // B is (K,N)
                     float* C)                          // C is (M,N)
{
    // C_rm(M,N) = A_rm(K,M)^T · B_rm(K,N)
    // C_cm(N,M) = B_cm · A_cm^T, where B_cm=B_rm^T(N,K), A_cm=A_rm^T(M,K)
    // cublas: op_N(B_cm(N,K)) · op_T(A_cm(M,K)) = (N,K)·(K,M) = (N,M) ✓
    // cublasSgemm(h, N, T, N, M, K, alpha, B_data, N, A_data, M, beta, C_data, N)
    float alpha = 1.0f, beta = 0.0f;
    if (use_tc) {
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, &alpha,
            B, CUDA_R_32F, N,
            A, CUDA_R_32F, M,
            &beta,
            C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K, &alpha,
            B, N, A, M, &beta, C, N);
    }
}


class MlpContext {
    bool use_tensor_;
    cublasHandle_t handle_;
    int n_linear_;                       // number of linear layers

    // Cached on device between forward and backward
    std::vector<GpuTensor*> d_Z_;        // pre-activations  [n_linear_]
    std::vector<GpuTensor*> d_A_;        // activations       [n_linear_ + 1]  (d_A_[0] = input)
    std::vector<std::string> act_types_; // "relu" or "none" per linear layer
    int batch_;                          // current batch size

    void free_cache() {
        for (auto* p : d_Z_) delete p;
        for (auto* p : d_A_) delete p;
        d_Z_.clear();
        d_A_.clear();
    }

public:
    MlpContext(bool use_tensor_cores)
        : use_tensor_(use_tensor_cores), batch_(0), n_linear_(0)
    {
        handle_ = get_handle();
    }

    ~MlpContext() {
        free_cache();
    }

    /*
     * forward(X, weights, activations)
     *
     * X:           numpy float32 (input_size, batch_size)
     * weights:     list of tuples [(W1,b1), (W2,b2), ...]  each numpy float32
     * activations: list of strings ["relu", "relu", ..., "none"]
     *
     * Returns: numpy float32 (output_size, batch_size) — logits (no softmax)
     */
    py::array_t<float> forward(py::array_t<float> X,
                                py::list weights,
                                py::list activations)
    {
        auto x_buf = X.request();
        int input_rows = x_buf.shape[0];
        batch_ = x_buf.shape[1];
        n_linear_ = weights.size();

        // Parse activation types
        act_types_.clear();
        for (int i = 0; i < n_linear_; i++) {
            act_types_.push_back(activations[i].cast<std::string>());
        }

        // Free old cache
        free_cache();

        // Upload input → d_A_[0]
        d_A_.push_back(new GpuTensor(input_rows, batch_));
        d_A_[0]->upload((float*)x_buf.ptr);

        // Forward through each linear layer
        for (int i = 0; i < n_linear_; i++) {
            py::tuple wb = weights[i].cast<py::tuple>();
            auto W_np = wb[0].cast<py::array_t<float>>();
            auto b_np = wb[1].cast<py::array_t<float>>();
            auto w_buf = W_np.request();
            auto b_buf = b_np.request();

            int out_size = w_buf.shape[0];
            int in_size  = w_buf.shape[1];

            // Upload W and b to temp GPU tensors
            GpuTensor d_W(out_size, in_size);
            GpuTensor d_b(out_size, 1);
            d_W.upload((float*)w_buf.ptr);
            d_b.upload((float*)b_buf.ptr);

            // Z = W · A_prev  (out_size × batch)
            GpuTensor* d_Zi = new GpuTensor(out_size, batch_);
            gemm_nn(handle_, use_tensor_,
                    d_W.data, out_size, in_size,
                    d_A_[i]->data, batch_,
                    d_Zi->data);

            // Z += b (broadcast bias)
            launch_add_bias(d_Zi->data, d_b.data, out_size, batch_);

            d_Z_.push_back(d_Zi);

            // Activation
            GpuTensor* d_Ai = new GpuTensor(out_size, batch_);
            if (act_types_[i] == "relu") {
                launch_relu_forward(d_Zi->data, d_Ai->data, out_size * batch_);
            } else {
                // "none": just copy Z → A
                d_Ai->copy_from(*d_Zi);
            }
            d_A_.push_back(d_Ai);
        }

        // Download final activations → CPU
        int out_rows = d_A_.back()->rows;
        auto result = py::array_t<float>({out_rows, batch_});
        d_A_.back()->download((float*)result.request().ptr);

        return result;
    }

    /*
     * backward(grad, weights)
     *
     * grad:    numpy float32 (output_size, batch_size) — dL/dZ_last from loss
     * weights: same list of (W,b) tuples as forward
     *
     * Returns: list of tuples [(dW1,db1), (dW2,db2), ...] numpy float32
     */
    py::list backward(py::array_t<float> grad_np,
                      py::list weights)
    {
        auto g_buf = grad_np.request();
        int grad_rows = g_buf.shape[0];
        int grad_cols = g_buf.shape[1];

        // Upload gradient
        GpuTensor d_grad(grad_rows, grad_cols);
        d_grad.upload((float*)g_buf.ptr);

        // We'll accumulate results
        py::list result_list;

        // Temporary storage for current gradient flowing backward
        float* curr_grad = d_grad.data;
        int curr_rows = grad_rows;

        // Backward through layers in reverse
        // We need to manage ownership of intermediate grad tensors
        std::vector<GpuTensor*> temp_grads;

        for (int i = n_linear_ - 1; i >= 0; i--) {
            py::tuple wb = weights[i].cast<py::tuple>();
            auto W_np = wb[0].cast<py::array_t<float>>();
            auto w_buf = W_np.request();
            int out_size = w_buf.shape[0];
            int in_size  = w_buf.shape[1];

            // Upload W to GPU (needed for gradient propagation)
            GpuTensor d_W(out_size, in_size);
            d_W.upload((float*)w_buf.ptr);

            // Apply activation backward (ReLU)
            GpuTensor* d_dZ = new GpuTensor(out_size, batch_);
            if (act_types_[i] == "relu") {
                // dZ = grad ⊙ relu'(Z)  where relu'(Z) = (Z > 0)
                launch_relu_backward(d_Z_[i]->data, curr_grad, d_dZ->data,
                                     out_size * batch_);
            } else {
                // "none" (last layer): dZ = grad (pass-through)
                cudaMemcpy(d_dZ->data, curr_grad,
                           out_size * batch_ * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }

            // dW = dZ · A_prev^T   — (out_size, batch) · (in_size, batch)^T = (out_size, in_size)
            GpuTensor d_dW(out_size, in_size);
            gemm_nt(handle_, use_tensor_,
                    d_dZ->data, out_size, batch_,    // A = dZ (out_size, batch)
                    d_A_[i]->data, in_size,          // B = A_prev (in_size, batch), will be transposed
                    d_dW.data);

            // db = sum_cols(dZ)   — (out_size, 1)
            GpuTensor d_db(out_size, 1);
            launch_sum_cols(d_dZ->data, d_db.data, out_size, batch_);

            // Download dW, db to CPU
            auto dW_np = py::array_t<float>({out_size, in_size});
            auto db_np = py::array_t<float>({out_size, 1});
            d_dW.download((float*)dW_np.request().ptr);
            d_db.download((float*)db_np.request().ptr);

            result_list.append(py::make_tuple(dW_np, db_np));

            // Propagate gradient to previous layer: dA_prev = W^T · dZ
            if (i > 0) {
                GpuTensor* d_grad_prev = new GpuTensor(in_size, batch_);
                // dA_prev(in_size, batch) = W(out_size, in_size)^T · dZ(out_size, batch)
                gemm_tn(handle_, use_tensor_,
                        d_W.data, out_size, in_size,   // A = W (out_size, in_size), transposed
                        d_dZ->data, batch_,            // B = dZ (out_size, batch)
                        d_grad_prev->data);
                curr_grad = d_grad_prev->data;
                curr_rows = in_size;
                temp_grads.push_back(d_grad_prev);
            }

            delete d_dZ;
        }

        // Clean up temporary gradient tensors
        for (auto* p : temp_grads) delete p;

        // Result is in reverse order (last layer first), reverse it
        py::list ordered;
        for (int i = py::len(result_list) - 1; i >= 0; i--) {
            ordered.append(result_list[i]);
        }

        return ordered;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  augment_images (backward compatible)
// ═══════════════════════════════════════════════════════════════════════════════

py::array_t<float> augment_images(py::array_t<float> images, float alpha, float sigma, float rot, float scale) {
    auto buf = images.request();

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

// ═══════════════════════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(cuda_ops, m_py) {
    m_py.def("matmul", &matmul, "GPU matrix multiply (row-major compatible)",
          py::arg("A"), py::arg("B"), py::arg("use_tensor_cores") = false);

    m_py.def("augment_images", &augment_images, "Deforma imágenes en GPU (Afín + Elástica)",
          py::arg("images"), py::arg("alpha"), py::arg("sigma"), py::arg("rot"), py::arg("scale"));

    py::class_<MlpContext>(m_py, "MlpContext", "GPU-resident forward/backward for MLP")
        .def(py::init<bool>(), py::arg("use_tensor_cores") = false)
        .def("forward", &MlpContext::forward,
             "Full forward pass on GPU. Returns logits (pre-softmax).",
             py::arg("X"), py::arg("weights"), py::arg("activations"))
        .def("backward", &MlpContext::backward,
             "Full backward pass on GPU. Returns list of (dW, db) tuples.",
             py::arg("grad"), py::arg("weights"));
}
