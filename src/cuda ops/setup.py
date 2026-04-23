"""
Setup para compilar cuda_ops.
Requiere: CUDA Toolkit, pybind11, g++.

Compilar:   cd "src/cuda ops" && python setup.py install
Verificar:  python -c "import cuda_ops; print('OK')"
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def find_cuda():
    """Encuentra la instalación de CUDA (Linux/WSL o Windows)."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if cuda_home is None:
        if sys.platform == "linux":
            # WSL / Linux
            candidates = [
                "/usr/local/cuda",
                "/usr/local/cuda-12.6",
                "/usr/local/cuda-12.0",
                "/usr/local/cuda-11.8",
            ]
        else:
            # Windows
            candidates = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
            ]
        for path in candidates:
            if os.path.exists(path):
                cuda_home = path
                break

    if cuda_home is None:
        raise RuntimeError(
            "CUDA Toolkit no encontrado.\n"
            "WSL: sudo apt install cuda-toolkit-12-6\n"
            "Windows: descargar desde developer.nvidia.com/cuda-downloads\n"
            "O setear CUDA_HOME manualmente."
        )

    print(f"CUDA encontrado en: {cuda_home}")
    return cuda_home


class CUDABuildExt(build_ext):
    """Custom build que compila archivos .cu con nvcc."""

    def build_extensions(self):
        # Detectar compilador
        if self.compiler.compiler_type == "unix":
            # En Linux/WSL, los .cu se compilan aparte con nvcc
            self.compiler.set_executable("compiler_so", "nvcc")
            self.compiler.set_executable("compiler_cxx", "nvcc")
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "--compiler-options", "-fPIC",
                    "-arch=sm_75",   # RTX 2080 Super
                    "-O3",
                    "--std=c++17",
                ]
        build_ext.build_extensions(self)


def get_pybind11_include():
    import pybind11
    return pybind11.get_include()


cuda_home = find_cuda()

# Determinar lib dir según plataforma
if sys.platform == "linux":
    lib_dir = os.path.join(cuda_home, "lib64")
else:
    lib_dir = os.path.join(cuda_home, "lib", "x64")

ext = Extension(
    name="cuda_ops",
    sources=[
        "bindings.cpp",
        "kernels/matmul.cu",
        "kernels/elementwise.cu",
    ],
    include_dirs=[
        get_pybind11_include(),
        os.path.join(cuda_home, "include"),
        ".",  # para encontrar gpu_tensor.cuh
    ],
    library_dirs=[lib_dir],
    libraries=["cublas", "cudart"],
    language="c++",
)

setup(
    name="cuda_ops",
    version="1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": CUDABuildExt},
)