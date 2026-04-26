#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "../gpu_tensor.cuh"

// Constants for 28x28 images
#define IMG_W 28
#define IMG_H 28

// Kernel para inicializar los estados de curand
__global__ void init_curand_kernel(curandState *state, unsigned long long seed, int total_threads) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_threads) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Genera los parámetros afines (un hilo por imagen)
__global__ void generate_random_fields_kernel(curandState *state, float *rot, float *scale_x, float *scale_y,
                                              int m, float rot_range, float scale_range) {
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (img_idx >= m) return;

    // Usamos el estado del primer píxel de la imagen
    curandState local_state = state[img_idx * IMG_W * IMG_H];
    
    // Rotación en radianes
    float rand_rot = curand_uniform(&local_state) * 2.0f - 1.0f; // [-1, 1]
    rot[img_idx] = rand_rot * rot_range * (M_PI / 180.0f);
    
    // Escala
    float rand_sx = curand_uniform(&local_state) * 2.0f - 1.0f;
    float rand_sy = curand_uniform(&local_state) * 2.0f - 1.0f;
    scale_x[img_idx] = 1.0f + rand_sx * scale_range;
    scale_y[img_idx] = 1.0f + rand_sy * scale_range;
    
    state[img_idx * IMG_W * IMG_H] = local_state;
}

// Vamos a rediseñar curand para que sea per-pixel para el campo
__global__ void generate_fields_per_pixel_kernel(curandState *state, float *dx, float *dy, int m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (px < IMG_W && py < IMG_H && img_idx < m) {
        int flat_idx = img_idx * (IMG_W * IMG_H) + py * IMG_W + px;
        curandState local_state = state[flat_idx];
        dx[flat_idx] = curand_uniform(&local_state) * 2.0f - 1.0f; // [-1, 1]
        dy[flat_idx] = curand_uniform(&local_state) * 2.0f - 1.0f; // [-1, 1]
        state[flat_idx] = local_state;
    }
}

// Filtro Gaussiano 2D simple (no separable por simplicidad, la ventana es chica)
__global__ void gaussian_blur_kernel(const float *src_dx, const float *src_dy, 
                                     float *dst_dx, float *dst_dy,
                                     float alpha, float sigma, int m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (px < IMG_W && py < IMG_H && img_idx < m) {
        int radius = (int)ceilf(3.0f * sigma);
        float sum_dx = 0.0f;
        float sum_dy = 0.0f;
        float weight_sum = 0.0f;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int nx = px + j;
                int ny = py + i;
                
                if (nx >= 0 && nx < IMG_W && ny >= 0 && ny < IMG_H) {
                    float w = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
                    int n_idx = img_idx * (IMG_W * IMG_H) + ny * IMG_W + nx;
                    sum_dx += src_dx[n_idx] * w;
                    sum_dy += src_dy[n_idx] * w;
                    weight_sum += w;
                }
            }
        }

        int out_idx = img_idx * (IMG_W * IMG_H) + py * IMG_W + px;
        // Normalizar y escalar por alpha
        dst_dx[out_idx] = (sum_dx / weight_sum) * alpha;
        dst_dy[out_idx] = (sum_dy / weight_sum) * alpha;
    }
}

// Aplicar deformación afín y elástica usando interpolación bilineal
__global__ void apply_deformation_kernel(const float *src_img, float *dst_img,
                                         const float *dx, const float *dy,
                                         const float *rot, const float *scale_x, const float *scale_y,
                                         int m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (px < IMG_W && py < IMG_H && img_idx < m) {
        int flat_idx = img_idx * (IMG_W * IMG_H) + py * IMG_W + px;
        
        // Coordenadas normalizadas al centro [-14, 14]
        float cx = px - IMG_W / 2.0f;
        float cy = py - IMG_H / 2.0f;

        // Parámetros afines
        float theta = rot[img_idx];
        float sx = scale_x[img_idx];
        float sy = scale_y[img_idx];

        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        // Inversa de la transformación afín (queremos saber de dónde viene el píxel)
        // 1. Deshacer rotación
        float rx = cx * cos_t + cy * sin_t;
        float ry = -cx * sin_t + cy * cos_t;
        
        // 2. Deshacer escala
        rx /= sx;
        ry /= sy;

        // Volver a coordenadas de imagen
        float src_x = rx + IMG_W / 2.0f;
        float src_y = ry + IMG_H / 2.0f;

        // 3. Añadir deformación elástica (inversa)
        src_x -= dx[flat_idx];
        src_y -= dy[flat_idx];

        // Interpolación bilineal
        float val = 0.0f;
        if (src_x >= 0 && src_x < IMG_W - 1 && src_y >= 0 && src_y < IMG_H - 1) {
            int x0 = (int)floorf(src_x);
            int y0 = (int)floorf(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float tx = src_x - x0;
            float ty = src_y - y0;

            int img_offset = img_idx * (IMG_W * IMG_H);
            float p00 = src_img[img_offset + y0 * IMG_W + x0];
            float p10 = src_img[img_offset + y0 * IMG_W + x1];
            float p01 = src_img[img_offset + y1 * IMG_W + x0];
            float p11 = src_img[img_offset + y1 * IMG_W + x1];

            float c0 = p00 * (1.0f - tx) + p10 * tx;
            float c1 = p01 * (1.0f - tx) + p11 * tx;
            
            val = c0 * (1.0f - ty) + c1 * ty;
        }

        dst_img[flat_idx] = val;
    }
}

// Función principal de C++ que orquesta los kernels
void gpu_augment(const float *d_src_img, float *d_dst_img, 
                 int m, float alpha, float sigma, float rot_range, float scale_range) {
    
    int total_pixels = m * IMG_W * IMG_H;
    
    // Alocar memoria temporal
    curandState *d_state;
    float *d_dx_raw, *d_dy_raw, *d_dx, *d_dy;
    float *d_rot, *d_scale_x, *d_scale_y;

    cudaMalloc(&d_state, total_pixels * sizeof(curandState));
    cudaMalloc(&d_dx_raw, total_pixels * sizeof(float));
    cudaMalloc(&d_dy_raw, total_pixels * sizeof(float));
    cudaMalloc(&d_dx, total_pixels * sizeof(float));
    cudaMalloc(&d_dy, total_pixels * sizeof(float));
    cudaMalloc(&d_rot, m * sizeof(float));
    cudaMalloc(&d_scale_x, m * sizeof(float));
    cudaMalloc(&d_scale_y, m * sizeof(float));

    // Configuración de grid
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    // Inicializar curand
    // Generamos un seed aleatorio simple basado en el tiempo
    unsigned long long seed = (unsigned long long)clock();
    init_curand_kernel<<<grid_size, block_size>>>(d_state, seed, total_pixels);

    // Grid 3D para imágenes
    dim3 threads(16, 16, 1);
    dim3 blocks((IMG_W + threads.x - 1) / threads.x, 
                (IMG_H + threads.y - 1) / threads.y, 
                m);

    // 1. Generar campos per pixel
    generate_fields_per_pixel_kernel<<<blocks, threads>>>(d_state, d_dx_raw, d_dy_raw, m);

    // 2. Generar parámetros afines (un hilo por imagen)
    int affine_threads = 256;
    int affine_blocks = (m + affine_threads - 1) / affine_threads;
    
    // Usamos el estado del primer pixel de cada imagen
    generate_random_fields_kernel<<<affine_blocks, affine_threads>>>(d_state,
                                                                     d_rot, d_scale_x, d_scale_y,
                                                                     m, rot_range, scale_range);

    // 3. Suavizado Gaussiano del campo elástico
    gaussian_blur_kernel<<<blocks, threads>>>(d_dx_raw, d_dy_raw, d_dx, d_dy, alpha, sigma, m);

    // 4. Aplicar deformación final
    apply_deformation_kernel<<<blocks, threads>>>(d_src_img, d_dst_img, d_dx, d_dy, 
                                                  d_rot, d_scale_x, d_scale_y, m);

    // Limpiar memoria temporal
    cudaFree(d_state);
    cudaFree(d_dx_raw);
    cudaFree(d_dy_raw);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_rot);
    cudaFree(d_scale_x);
    cudaFree(d_scale_y);
}
