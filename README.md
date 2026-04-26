# SIA-TP3

Perceptrón Multicapa desde cero con NumPy, con aceleración GPU opcional (CUDA/Tensor Cores).

## Requisitos

```bash
pip install numpy pandas pyyaml matplotlib
```

## Correr Ejercicio 2 (Dígitos)

```bash
python3 main.py --config config.yaml --data data/digits.csv --test test_data/digits_test.csv
```

## Correr Ejercicio 3 (≥98% accuracy)

```bash
python3 main.py --config config.yaml --data data/more_digits.csv --test test_data/digits_test.csv
```

## Graficar resultados

```bash
python3 plot_results.py
```

## Configuración

Todo se configura en `config.yaml`. Ejemplo:

```yaml
experiments:
  - experiment_name: mi_experimento
    architecture: [784, 128, 64, 10]
    activation: relu # relu / tanh
    output_activation: softmax # softmax / same / none
    loss: cross_entropy # cross_entropy / mse
    optimizer: adam # adam / sgd / sgd_momentum
    learning_rate: 0.001
    batch_size: 64
    epochs: 50
    train_val_split: 0.8
    seed: 42
    backend: cpu # cpu / cuda / tensor
    augmentation: # WARNING: ONLY GPU SUPPORTED
      enabled: false 
      alpha: 36.0 # Escala de la deformación elástica
      sigma: 5.0 # Suavizado de la deformación elástica (Gaussian blur)
      rotation_range: 15.0 # Rango de rotación en grados (ej: -15 a 15)
      scale_range: 0.15 # Variación de escala (ej: 0.85 a 1.15)
```

## Aceleración GPU (opcional)

Para correr las multiplicaciones de matrices en la GPU (CUDA cores o Tensor Cores de NVIDIA):

### 1. Requisitos GPU

- GPU NVIDIA con soporte CUDA (ej: RTX 2080 Super)
- Driver NVIDIA actualizado en Windows
- WSL2

### 2. Instalar CUDA Toolkit en WSL

```bash
# Verificar que WSL ve la GPU
nvidia-smi

# Descargar el instalador (desde developer.nvidia.com/cuda-downloads → Linux → x86_64 → WSL-Ubuntu → runfile)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Instalar solo el toolkit (NO el driver)
sudo sh cuda_12.6.0_560.28.03_linux.run --toolkit --silent --override

# Configurar PATH
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar
nvcc --version

# Borrar el instalador
rm cuda_12.6.0_560.28.03_linux.run
```

### 3. Instalar pybind11

```bash
pip install pybind11
```

### 4. Compilar el módulo CUDA

```bash
cd "src/cuda ops"
make
make install    # (o: sudo make install)
```

Verificar:

```bash
python3 -c "import cuda_ops; print('OK')"
```

### 5. Usar GPU

Cambiar en `config.yaml`:

```yaml
backend: cuda    # CUDA Cores (float32)
backend: tensor  # Tensor Cores (float16 → float32, más rápido en RTX 20xx+)
```
