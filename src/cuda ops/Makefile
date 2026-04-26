# Makefile para compilar cuda_ops
# Uso: make (desde src/cuda ops/)

PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE = $(shell python3 -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

CUDA_HOME ?= /usr/local/cuda-12.6
NVCC = $(CUDA_HOME)/bin/nvcc

# RTX 2080 Super = sm_75
ARCH = -arch=sm_75

TARGET = cuda_ops$(EXT_SUFFIX)

all: $(TARGET)
	@echo "Compilado: $(TARGET)"
	@echo "Verificar: python3 -c \"import cuda_ops; print('OK')\""

$(TARGET): bindings.cpp kernels/matmul.cu kernels/elementwise.cu kernels/augmentation.cu gpu_tensor.cuh
	$(NVCC) -shared -o $@ \
		bindings.cpp kernels/matmul.cu kernels/elementwise.cu kernels/augmentation.cu \
		-I$(PYTHON_INCLUDE) \
		-I$(PYBIND11_INCLUDE) \
		-I. \
		-lcublas -lcudart -lcurand \
		$(ARCH) \
		-Xcompiler -fPIC \
		--std=c++17 \
		-O3

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) $(shell python3 -c "import site; print(site.getsitepackages()[0])")/
	@echo "Instalado en site-packages"

.PHONY: all clean install
