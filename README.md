# SIA-TP3

Perceptrón Multicapa desde cero con NumPy, con aceleración GPU opcional (CUDA/Tensor Cores).

## Requisitos

```bash
pip install numpy pandas pyyaml matplotlib
```

---

## Ejercicio 2 — Clasificación de dígitos (digits.csv)

### Correr experimentos

```bash
# Todos los experimentos definidos en config_ej2.yaml
python main.py --config config_ej2.yaml --test test_data/digits_test.csv

# Un solo experimento o los que esten en config.yaml
python main.py --config config.yaml --test test_data/digits_test.csv

# Equivalente: --data tiene default data/digits.csv, se puede omitir
python main.py --config config.yaml --data data/digits.csv --test test_data/digits_test.csv
```

### Evaluar modelo guardado (sin re-entrenar)

```bash
python main.py --config config_ej2.yaml --test test_data/digits_test.csv --test-only
```

---

## Ejercicio 3 — Alta precisión (more_digits.csv)

```bash
python main.py --config config_ej3.yaml --data data/more_digits.csv --test test_data/digits_test.csv
```

---

## Gráficos

### Comparación por grupo (arquitecturas / LRs / optimizadores)
Genera un PNG por grupo con curvas de val_acc + tabla de resultados (val y test accuracy).

```bash
python plot_results.py
python plot_results.py --results-dir results/
```

Salida: `results/compare_arquitectura.png`, `compare_learning_rate.png`, `compare_optimizador.png`

---

### Gráfico de un solo experimento (loss + accuracy + tiempo por época)
Muestra la línea de test accuracy guardada en el `_config.json`.

```bash
# Por nombre (busca results/<name>.csv y results/<name>_config.json)
python plot_single.py --name digit_lr_0001_cpu

# Por ruta directa
python plot_single.py --csv results/arch_large.csv

# Con título personalizado para la presentación
python plot_single.py --name arch_large --title "Configuración óptima: [784,512,256,10] Adam"
```

Salida: `results/<name>_plot.png`

---

### Funciones de activación (ReLU / Tanh / Softmax)

```bash
python plot_activations.py
python plot_activations.py --out results/activations.png
```

---

### Distribución de clases en el dataset

```bash
python plot_digit_frequency.py
python plot_digit_frequency.py --data data/digits.csv --out results/digit_frequency.png
```

---

## Configuración

Todo se configura en un archivo YAML. Ejemplo completo:

```yaml
experiments:
  - experiment_name: mi_experimento
    architecture: [784, 512, 256, 10]
    activation: relu              # relu / tanh
    output_activation: softmax    # softmax / same / none
    loss: cross_entropy           # cross_entropy / mse
    optimizer: adam               # adam / sgd / sgd_momentum
    learning_rate: 0.001
    momentum: 0.9                 # solo para sgd_momentum
    batch_size: 256
    epochs: 50
    train_val_split: 0.8
    seed: 42
    dropout: 0.0                  # Inverted Dropout (0.0 = desactivado)
    early_stopping_patience: 10   # épocas sin mejora para parar (omitir = desactivado)
    backend: cpu                  # cpu / cuda / tensor
    augmentation:                 # WARNING: ONLY GPU SUPPORTED
      enabled: false
      alpha: 36.0                 # Escala de la deformación elástica
      sigma: 5.0                  # Suavizado (Gaussian blur)
      rotation_range: 15.0        # Rango de rotación en grados
      scale_range: 0.15           # Variación de escala (±15%)
```

---

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
