#pragma once
#include <cuda_runtime.h>
#include <cstdio>

struct GpuTensor {
    float * data;
    int rows;
    int cols;

    GpuTensor(int r, int c) : rows(r), cols(c), data(nullptr) {
        cudaMalloc(&data, r * c * sizeof(float));
    }

    ~GpuTensor() {
        if (data) cudaFree(data);
    }

    // No copy (evitar double-free)
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    // Move OK
    GpuTensor(GpuTensor&& o) noexcept : data(o.data), rows(o.rows), cols(o.cols) {
        o.data = nullptr;
    }

    int size() const { return rows * cols; }

    void upload(const float * cpu_ptr){
        cudaMemcpy(data, cpu_ptr, size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void download(float * cpu_ptr) const {
        cudaMemcpy(cpu_ptr, data, size() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void zero() {
        cudaMemset(data, 0, size() * sizeof(float));
    }

    // Device-to-device copy
    void copy_from(const GpuTensor& src) {
        cudaMemcpy(data, src.data, size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
};