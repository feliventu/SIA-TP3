struct GpuTensor {
    float * data;
    int rows;
    int cols;

    GpuTensor(int r, int c) : rows(r), cols(c) {
        cudaMalloc(&data, r * c * sizeof(float));
    }

    ~GpuTensor() {
        cudaFree(data);
    }

    void upload(const float * cpu_ptr){
        cudaMemcpy(data, cpu_ptr, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    void download(float * cpu_ptr){
        cudaMemcpy(cpu_ptr, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    }
};